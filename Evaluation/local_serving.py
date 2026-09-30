"""Owned local vLLM replicas and a standard-library round-robin API proxy."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def gpu_groups(gpus="auto", tensor_parallel_size=1):
    if tensor_parallel_size < 1:
        raise ValueError("tensor-parallel-size must be positive")
    if gpus == "auto":
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            gpus = visible
        else:
            try:
                result = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
                gpus = ",".join(result.split())
            except (FileNotFoundError, subprocess.CalledProcessError) as error:
                raise ValueError("No GPUs detected; use --gpus or connect to an existing --base-url") from error
    devices = [value.strip() for value in gpus.split(",") if value.strip()]
    if not devices or "-1" in devices or len(set(devices)) != len(devices) or len(devices) % tensor_parallel_size:
        raise ValueError("GPU IDs must be unique, nonempty, and divisible by tensor-parallel-size")
    return [devices[index:index + tensor_parallel_size] for index in range(0, len(devices), tensor_parallel_size)]


def models_ready(base_url, model, api_key="EMPTY"):
    request = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.load(response)
        return any(row.get("id") == model for row in payload.get("data", []))
    except (OSError, ValueError, TypeError):
        return False


def start_proxy(backends):
    """Bind only on localhost; forward non-streaming evaluation requests."""
    counter, lock = itertools.cycle(backends), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def forward(self):
            if not self.path.startswith("/v1/"):
                self.send_error(404)
                return
            with lock:
                backend = next(counter)
            headers = {key: value for key, value in self.headers.items()
                       if key.lower() not in {"host", "connection", "content-length"}}
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))) if self.command == "POST" else None
            request = urllib.request.Request(backend.rstrip("/") + self.path, data=body, headers=headers, method=self.command)
            try:
                with urllib.request.urlopen(request, timeout=3600) as response:
                    content, status = response.read(), response.status
            except urllib.error.HTTPError as error:
                content, status = error.read(), error.code
            except OSError:
                content, status = b'{"error":{"message":"Local model replica is unavailable"}}', 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        do_GET = forward
        do_POST = forward

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


def stop_owned(processes):
    # Every process here was started by this invocation in a new session.
    # Never use pkill, killall, or a port/GPU-based process lookup.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process in processes:
        try:
            process.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # The launcher may have exited while its GPU workers still exist.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


@contextmanager
def local_services(*, model_path, model_name, output, gpus="auto", tensor_parallel_size=1,
                   context_length=13312, startup_timeout=1800, port_base=8000,
                   trust_remote_code=False, server_args=()):
    groups = gpu_groups(gpus, tensor_parallel_size)
    services = Path(output)
    services.mkdir(parents=True, exist_ok=True)
    processes, logs, backends = [], [], []
    server = worker = None
    try:
        for index, devices in enumerate(groups):
            port = port_base + index
            # Fail rather than reuse an unrelated process listening on this port.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                       "--model", model_path, "--served-model-name", model_name,
                       "--host", "127.0.0.1", "--port", str(port),
                       "--tensor-parallel-size", str(tensor_parallel_size),
                       "--gpu-memory-utilization", "0.90", "--max-model-len", str(context_length),
                       "--max-num-seqs", "32", "--dtype", "bfloat16", "--enable-prefix-caching", "--disable-log-requests"]
            if trust_remote_code:
                command.append("--trust-remote-code")
            command.extend(server_args)
            cache = services / f"replica-{index:02d}"
            cache.mkdir(exist_ok=True)
            environment = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(devices),
                                       "VLLM_CACHE_ROOT": str(cache / "vllm"),
                                       "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                                       "TRITON_CACHE_DIR": str(cache / "triton")}
            log = (services / f"replica-{index:02d}.log").open("a")
            logs.append(log)
            processes.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            backends.append(f"http://127.0.0.1:{port}")
            print(f"[service] replica={index} GPUs={','.join(devices)} port={port}", flush=True)
        deadline = time.monotonic() + startup_timeout
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError(f"A vLLM replica exited; inspect {services}")
            if all(models_ready(backend + "/v1", model_name) for backend in backends):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Model startup timed out; inspect {services}")
            time.sleep(1)
        server, worker = start_proxy(backends)
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            worker.join()
        stop_owned(processes)
        for log in logs:
            log.close()
