"""Credential-safe startup checks for required OpenAI-compatible external models."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from Runtime.rubric.scoring import extract_json_object, parse_policy_score, parse_rubric


def external_request_error(error: BaseException | None) -> str:
    """Safe artifact text; third-party exception messages can contain response secrets."""
    if error is None:
        return "external request failed"
    status = getattr(error, "status_code", getattr(error, "code", None))
    return f"{type(error).__name__}: HTTP {status}" if isinstance(status, int) else type(error).__name__


@dataclass(frozen=True)
class ExternalModelSpec:
    role: str
    endpoint: str
    model: str
    endpoint_setting: str = "endpoint"
    model_setting: str = "model"
    api_key_env: str = "KIMI_API_KEY"
    output_kind: str = "binary"
    temperature: float = 0.0
    max_tokens: int = 512
    completion_token_field: str = "max_tokens"
    min_criteria: int = 2
    extra_body: dict = field(default_factory=dict)

    def validate(self, *, dry_run: bool = False) -> None:
        prefix = f"External model ({self.role})"
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError(f"{prefix}: model is required; configure {self.model_setting}")
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ValueError(f"{prefix}: api_key_env must name the credential environment variable")
        if not isinstance(self.endpoint, str):
            raise ValueError(f"{prefix}: endpoint must be an HTTP(S) API base URL")
        if self.endpoint:
            try:
                url = urllib.parse.urlsplit(self.endpoint)
                valid = (url.scheme in {"http", "https"} and bool(url.hostname)
                         and url.username is None and url.password is None
                         and not url.query and not url.fragment)
                url.port  # Reject malformed ports without displaying the input URL.
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{prefix}: use an HTTP(S) endpoint without embedded credentials or query parameters")
        elif not dry_run:
            raise ValueError(f"{prefix}: endpoint is missing; configure {self.endpoint_setting}")
        if self.output_kind not in {"binary", "rubric", "score"}:
            raise ValueError(f"{prefix}: unsupported output kind")
        if self.completion_token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"{prefix}: invalid completion token field")
        if not math.isfinite(self.temperature) or self.temperature < 0 or min(self.max_tokens, self.min_criteria) < 1:
            raise ValueError(f"{prefix}: invalid sampling or output budget")
        if not dry_run and not os.getenv(self.api_key_env, "").strip():
            raise ValueError(f"{prefix}: API key is missing; export {self.api_key_env} on this worker")


def _probe_messages(spec: ExternalModelSpec) -> list[dict]:
    if spec.output_kind == "binary":
        task = ('Assigned goal: inspect the room. Evidence: the agent inspected the room and returned its contents. '
                'Judge completion. Return only JSON with a boolean success and a string reason.')
    elif spec.output_kind == "score":
        task = ('Score this completed inspection: the agent inspected the room and returned its contents. '
                'Rubric: completion and useful reporting, equally weighted. '
                'Return only JSON with final_score between 0 and 1 and a boolean success. '
                'A failed trajectory must receive 0; a successful one must receive a positive score.')
    else:
        task = (f'Generate a brief rubric for inspecting a room and reporting its contents. Return only JSON '
                f'with at least {spec.min_criteria} rubric_items, each containing name, description and '
                'a non-negative weight; the weights must have a positive total.')
    return [{"role": "system", "content": "You are an evaluator. Follow the requested JSON schema exactly."},
            {"role": "user", "content": task}]


def _validate_response(spec: ExternalModelSpec, body: dict) -> None:
    text = body["choices"][0]["message"]["content"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty completion")
    if spec.output_kind == "binary":
        payload = extract_json_object(text.rsplit("</think>", 1)[-1])
        if not isinstance(payload.get("success"), bool) or not isinstance(payload.get("reason", ""), str):
            raise ValueError("Invalid binary judgment")
    elif spec.output_kind == "score":
        parse_policy_score(text)
    else:
        parse_rubric(text, spec.min_criteria)


def probe_external_model(spec: ExternalModelSpec, *, timeout: float = 20.0, retries: int = 1) -> None:
    """One small completion, with bounded retries only for transient transport errors.

    Never include HTTP bodies, URLs, credentials or raw exception messages in
    errors: an external service may echo authorization headers in its response.
    """
    spec.validate()
    if not math.isfinite(timeout) or timeout <= 0 or retries < 0:
        raise ValueError("Invalid external-model preflight timeout/retries")
    payload = dict(spec.extra_body)
    payload.update(model=spec.model, messages=_probe_messages(spec), temperature=spec.temperature)
    payload[spec.completion_token_field] = min(spec.max_tokens, 512)
    request = urllib.request.Request(
        spec.endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[spec.api_key_env]}"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read(1_048_576).decode("utf-8"))
            _validate_response(spec, body)
            return
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            status = exc.code
            exc.close()
            if not retryable or attempt == retries:
                hint = ("check the API key and permissions" if status in {401, 403} else
                        "check the model name and API base URL" if status == 404 else
                        "check service availability and API compatibility")
                raise RuntimeError(f"External model ({spec.role}) preflight failed: HTTP {status}; {hint}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries:
                raise RuntimeError(f"External model ({spec.role}) preflight failed: connection/timeout; check worker network access") from None
        except (ValueError, TypeError, KeyError, IndexError):
            raise RuntimeError(f"External model ({spec.role}) preflight failed: invalid {spec.output_kind} completion/JSON schema") from None
        time.sleep(min(2 ** attempt, 2))


def preflight_external_models(specs, *, dry_run: bool = False) -> None:
    """Check all configuration first, then probe required services; no policy fallback."""
    specs = list(specs)
    for spec in specs:
        spec.validate(dry_run=dry_run)
    if dry_run:
        if specs:
            print("Dry run: external-model configuration structure checked; credentials and service availability NOT verified.", file=sys.stderr)
        return
    checked = set()
    for spec in specs:
        identity = (spec.endpoint, spec.model, spec.api_key_env, spec.output_kind,
                    spec.temperature, spec.max_tokens, spec.completion_token_field,
                    spec.min_criteria, json.dumps(spec.extra_body, sort_keys=True))
        if identity not in checked:
            probe_external_model(spec)
            checked.add(identity)
