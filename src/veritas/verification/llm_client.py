"""Common LLM client interface and backends for VERITAS verification.

Supports:
- Local model backend (Hugging Face Transformers / local inference).
- API backend (OpenAI, Anthropic, or OpenAI-compatible endpoints) via standard library HTTP.
- Mock backend for deterministic, zero-cost unit testing.
- Strict secret hygiene: Never log or persist API keys or authorization headers.
- Trace recording for raw model output preservation and replay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.error
import urllib.request
import uuid

logger = logging.getLogger("veritas.verification.llm_client")


def mask_secret(secret: str | None) -> str:
    """Mask a secret string for safe logging and string representation."""
    if not secret:
        return "<none>"
    s = str(secret).strip()
    if len(s) <= 8:
        return "********"
    return f"{s[:3]}...{s[-4:]}"


def sanitize_sensitive_headers(headers: dict[str, str]) -> dict[str, str]:
    """Sanitize headers dictionary to remove any auth tokens."""
    sanitized: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in {"authorization", "api-key", "x-api-key", "token"}:
            sanitized[k] = mask_secret(v)
        else:
            sanitized[k] = v
    return sanitized


@dataclass
class LLMResponse:
    """Standardized response from an LLM call.

    Attributes:
        content: The text generation returned by the model.
        model: Identifier of the model that generated the completion.
        latency_seconds: Wall-clock duration of the call in seconds.
        raw_output: Unparsed raw payload from the backend provider.
        usage: Token counts (prompt_tokens, completion_tokens, total_tokens) if available.
        metadata: Provider-specific metadata.
    """

    content: str
    model: str
    latency_seconds: float = 0.0
    raw_output: Any = None
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary for serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "usage": self.usage,
            "metadata": self.metadata,
        }


class TraceRecorder:
    """Records raw model inputs, prompts, outputs, and metadata for analysis and replay."""

    def __init__(self, trace_dir: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.enabled and self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        prompt: str,
        system_prompt: str | None,
        response: LLMResponse,
        claim: str | None = None,
        claim_date: str | None = None,
        supplied_evidence_ids: list[str] | None = None,
        attempt: int = 1,
        parsed_result: dict[str, Any] | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a single interaction trace to JSONL.

        Returns:
            The trace record dictionary.
        """
        trace_id = str(uuid.uuid4())
        record = {
            "trace_id": trace_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "claim": claim,
            "claim_date": claim_date,
            "supplied_evidence_ids": supplied_evidence_ids or [],
            "attempt": attempt,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": response.model,
            "raw_response": response.content,
            "latency_seconds": response.latency_seconds,
            "usage": response.usage,
            "parsed_result": parsed_result,
            "error": error,
            "extra": extra or {},
        }

        if self.enabled and self.trace_dir:
            try:
                date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
                trace_file = self.trace_dir / f"verification_traces_{date_str}.jsonl"
                with trace_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Failed to save trace record: %s", exc)

        return record


class BaseLLMClient(ABC):
    """Abstract base class establishing common client interface for verification models."""

    def __init__(self, model_name: str, backend: str) -> None:
        self.model_name = model_name
        self.backend = backend

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute text generation.

        Args:
            prompt: User prompt content.
            system_prompt: Optional system instruction prompt.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            Standardized LLMResponse.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backend={self.backend!r}, model={self.model_name!r})"


class MockLLMClient(BaseLLMClient):
    """In-memory mocked LLM client for deterministic testing without external calls."""

    def __init__(
        self,
        canned_responses: list[str | dict[str, Any] | Exception] | None = None,
        response_generator: Callable[[str, str | None], str] | None = None,
        model_name: str = "mock-model",
    ) -> None:
        super().__init__(model_name=model_name, backend="mock")
        self.responses: list[str | dict[str, Any] | Exception] = list(canned_responses or [])
        self.response_generator = response_generator
        self.history: list[dict[str, Any]] = []

    def set_responses(self, responses: list[str | dict[str, Any] | Exception]) -> None:
        """Replace the list of queued responses."""
        self.responses = list(responses)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        call_record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        }
        self.history.append(call_record)

        # Generate response
        if self.response_generator:
            content = self.response_generator(prompt, system_prompt)
        elif self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, dict):
                content = json.dumps(item)
            else:
                content = str(item)
        else:
            # Default fallback mock response
            content = json.dumps(
                {
                    "label": "Unverified",
                    "confidence": 0.5,
                    "evidence_ids": [],
                    "explanation": "Default mock fallback response.",
                }
            )

        latency = time.perf_counter() - start_time
        return LLMResponse(
            content=content,
            model=self.model_name,
            latency_seconds=latency,
            raw_output={"mock": True},
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": len(content) // 4},
        )


class APILLMClient(BaseLLMClient):
    """Generic HTTP API client supporting OpenAI-compatible and Anthropic endpoints.

    Uses Python standard library urllib to avoid requiring heavy third-party SDKs.
    Strictly safeguards API keys from logging or exposure.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        api_key_env_var: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        provider: str = "openai",  # 'openai' or 'anthropic'
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, backend=f"api_{provider}")
        self.api_base = api_base.rstrip("/")
        self.provider = provider.lower().strip()
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

        # Load API key safely from explicit argument or environment
        loaded_key = api_key or os.environ.get(api_key_env_var) or os.environ.get("VERITAS_API_KEY")
        self._api_key = str(loaded_key).strip() if loaded_key else None
        self._api_key_env_var = api_key_env_var

    def __repr__(self) -> str:
        return (
            f"APILLMClient(provider={self.provider!r}, model={self.model_name!r}, "
            f"api_base={self.api_base!r}, key_set={bool(self._api_key)})"
        )

    def _get_api_key(self) -> str:
        """Retrieve the API key, raising a secure error if missing."""
        if not self._api_key:
            raise ValueError(
                f"API key not found. Please set environment variable '{self._api_key_env_var}' "
                f"or 'VERITAS_API_KEY', or pass api_key to client initialization."
            )
        return self._api_key

    def _build_request_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> tuple[str, dict[str, str], bytes]:
        """Construct the endpoint URL, headers, and request body bytes."""
        api_key = self._get_api_key()

        if self.provider == "anthropic":
            url = f"{self.api_base}/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            body: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                body["system"] = system_prompt
        else:
            # Default OpenAI-compatible Chat Completions
            url = f"{self.api_base}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            body = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        body.update(kwargs)
        payload_bytes = json.dumps(body).encode("utf-8")
        return url, headers, payload_bytes

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        url, headers, payload_bytes = self._build_request_payload(
            prompt, system_prompt, temperature, max_tokens, **kwargs
        )

        start_time = time.perf_counter()
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                data = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            # Sanitize any trace of API key or sensitive headers in error message
            safe_err = re.sub(r"sk-[a-zA-Z0-9_\-]+", "sk-***", err_body)
            logger.error(
                "API call failed with HTTP %d: %s (url=%s)",
                exc.code,
                safe_err[:300],
                url,
            )
            raise RuntimeError(f"LLM API request failed with status {exc.code}: {safe_err}") from exc
        except Exception as exc:
            logger.error("API connection error to %s: %s", url, exc)
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        latency = time.perf_counter() - start_time

        # Extract content based on provider
        if self.provider == "anthropic":
            content_blocks = data.get("content", [])
            content = content_blocks[0].get("text", "") if content_blocks else ""
            usage = data.get("usage", {})
        else:
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", self.model_name),
            latency_seconds=latency,
            raw_output=data,
            usage=usage,
            metadata={"provider": self.provider, "api_base": self.api_base},
        )


class LocalLLMClient(BaseLLMClient):
    """Local inference client utilizing Hugging Face Transformers.

    Loads tokenizer and pipeline lazily to avoid heavy compute/memory on initialization.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        pipeline_instance: Any = None,
    ) -> None:
        super().__init__(model_name=model_name, backend="local_transformers")
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipeline = pipeline_instance
        self._is_loaded = pipeline_instance is not None

    def _ensure_loaded(self) -> None:
        """Lazily load the Hugging Face text-generation pipeline."""
        if self._is_loaded and self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "Local inference requires 'torch' and 'transformers'. "
                "Install them or use MockLLMClient/APILLMClient."
            ) from exc

        logger.info("Loading local verification model '%s' on device '%s'...", self.model_name, self.device)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype if self.torch_dtype != "auto" else "auto",
        )
        self._is_loaded = True

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        self._ensure_loaded()

        start_time = time.perf_counter()

        # Format input for chat template if pipeline tokenizer supports it
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 1e-4) if temperature > 0 else 0.0,
            "do_sample": temperature > 0.0,
        }
        gen_kwargs.update(kwargs)

        try:
            outputs = self._pipeline(messages, **gen_kwargs)
            # Pipeline returns list of generated messages
            if isinstance(outputs, list) and outputs:
                generated = outputs[0]
                if isinstance(generated, dict) and "generated_text" in generated:
                    gen_text = generated["generated_text"]
                    if isinstance(gen_text, list) and gen_text:
                        content = gen_text[-1].get("content", "")
                    else:
                        content = str(gen_text)
                else:
                    content = str(generated)
            else:
                content = str(outputs)
        except Exception as exc:
            # Fallback to direct prompt generation
            logger.warning("Chat template generation failed, falling back to raw prompt: %s", exc)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            outputs = self._pipeline(full_prompt, **gen_kwargs)
            if isinstance(outputs, list) and outputs:
                content = outputs[0].get("generated_text", "")
            else:
                content = str(outputs)

        latency = time.perf_counter() - start_time
        return LLMResponse(
            content=content,
            model=self.model_name,
            latency_seconds=latency,
            raw_output={"local": True},
        )


def create_llm_client(
    backend: str = "mock",
    model_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> BaseLLMClient:
    """Factory creating an LLM client backend based on configuration.

    Args:
        backend: Backend identifier ('mock', 'api', 'local', 'openai', 'anthropic').
        model_name: Optional override model identifier.
        config: Configuration dictionary.

    Returns:
        Concrete BaseLLMClient instance.
    """
    cfg = config or {}
    backend_norm = backend.lower().strip()

    if backend_norm == "mock":
        return MockLLMClient(
            model_name=model_name or cfg.get("model", "mock-model"),
        )
    elif backend_norm in {"api", "openai", "anthropic"}:
        provider = "anthropic" if backend_norm == "anthropic" else cfg.get("provider", "openai")
        return APILLMClient(
            model_name=model_name or cfg.get("model", "gpt-4o-mini"),
            api_base=cfg.get("api_base", "https://api.openai.com/v1" if provider == "openai" else "https://api.anthropic.com/v1"),
            api_key_env_var=cfg.get("api_key_env_var", "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"),
            api_key=cfg.get("api_key"),
            provider=provider,
            timeout=float(cfg.get("timeout", 60.0)),
        )
    elif backend_norm in {"local", "transformers"}:
        return LocalLLMClient(
            model_name=model_name or cfg.get("model", "meta-llama/Llama-3-8B-Instruct"),
            device=cfg.get("device", "auto"),
            torch_dtype=cfg.get("torch_dtype", "auto"),
        )
    else:
        raise ValueError(f"Unsupported LLM backend: '{backend}'. Must be 'mock', 'api', or 'local'.")
