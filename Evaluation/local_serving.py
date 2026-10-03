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


def prepare_vllm_request(payload, tokenizer):
    """Honor CodeAct's exact prompt budget with the training truncation policy.

    vLLM does not implement the SGLang/AReaL chat-template budget extension.
    Encode the same template locally, preserve the system/task prefix, discard
    oldest complete interaction pairs, then use vLLM's token-ID completion API.
    The caller converts the completion response back to the chat API schema.
    """
    template_kwargs = dict(payload.get("chat_template_kwargs") or {})
    budget = template_kwargs.pop("platoon_max_prompt_tokens", None)
    if budget is None:
        return payload, False
    if tokenizer is None:
        raise ValueError("A local tokenizer is required to honor the CodeAct prompt budget")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("CodeAct prompt budget must be positive")
    if payload.get("stream") or payload.get("tools"):
        raise ValueError("Budgeted CodeAct requests must be non-streaming and tool-free")
    messages = list(payload["messages"])

    def encode(turns, *, add_generation_prompt=True):
        return tokenizer.apply_chat_template(turns, tokenize=True,
                                             add_generation_prompt=add_generation_prompt,
                                             **template_kwargs)

    tokens = encode(messages)
    prefix, recent = messages[:2], messages[2:]
    while len(tokens) > budget and len(recent) >= 2:
        recent = recent[2:]
        tokens = encode(prefix + recent)
    if len(tokens) > budget:
        prefix_tokens = encode(prefix, add_generation_prompt=False)[:budget]
        tail_budget = budget - len(prefix_tokens)
        tokens = prefix_tokens + (tokens[-tail_budget:] if tail_budget > 0 else [])
    completion = dict(payload)
    for key in ("messages", "chat_template_kwargs", "max_completion_tokens"):
        completion.pop(key, None)
    completion["prompt"] = tokens
    completion["max_tokens"] = payload.get("max_completion_tokens") or payload.get("max_tokens", 512)
    return completion, True


def completion_as_chat(payload):
    """Preserve finish reasons, token usage and response IDs for LiteLLM."""
    response = dict(payload)
    response["object"] = "chat.completion"
    response["choices"] = [dict(index=choice["index"],
                                 message={"role": "assistant", "content": choice["text"]},
                                 finish_reason=choice.get("finish_reason"),
                                 logprobs=choice.get("logprobs"))
                           for choice in payload["choices"]]
    return response


def start_proxy(backends, *, tokenizer=None, tokenizer_path=None, trust_remote_code=False):
    """Bind only on localhost; forward non-streaming evaluation requests."""
    counter, lock = itertools.cycle(backends), threading.Lock()
    tokenizer_lock = threading.Lock()

    def budget_tokenizer(payload):
        nonlocal tokenizer
        if (payload.get("chat_template_kwargs") or {}).get("platoon_max_prompt_tokens") is None:
            return None
        with tokenizer_lock:
            if tokenizer is None and tokenizer_path is not None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                         trust_remote_code=trust_remote_code,
                                                         local_files_only=True)
            return tokenizer

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
            path, budgeted = self.path, False
            if path == "/v1/chat/completions" and body:
                try:
                    payload = json.loads(body)
                    payload, budgeted = prepare_vllm_request(payload, budget_tokenizer(payload))
                    if budgeted:
                        body = json.dumps(payload).encode()
                        path = "/v1/completions"
                except (ValueError, KeyError, TypeError) as error:
                    self.send_error(400, str(error))
                    return
            request = urllib.request.Request(backend.rstrip("/") + path, data=body, headers=headers, method=self.command)
            try:
                with urllib.request.urlopen(request, timeout=3600) as response:
                    content, status = response.read(), response.status
                if budgeted:
                    content = json.dumps(completion_as_chat(json.loads(content))).encode()
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
    # On cancellation, clients can disconnect while upstream requests remain
    # active. Do not block server_close on those handlers before stopping the
    # owned model processes; otherwise teardown can wait an hour.
    server.daemon_threads = True
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
        server, worker = start_proxy(backends, tokenizer_path=model_path,
                                     trust_remote_code=trust_remote_code)
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            worker.join()
        stop_owned(processes)
        for log in logs:
            log.close()
