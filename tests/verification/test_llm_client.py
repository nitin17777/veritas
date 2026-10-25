"""Unit tests for verification LLM clients, mock backends, and secret hygiene."""

from __future__ import annotations

import io
import json
import os
from unittest.mock import MagicMock, patch
import pytest

from veritas.verification.llm_client import (
    APILLMClient,
    LocalLLMClient,
    MockLLMClient,
    TraceRecorder,
    create_llm_client,
    mask_secret,
    sanitize_sensitive_headers,
)


def test_mask_secret():
    assert mask_secret(None) == "<none>"
    assert mask_secret("") == "<none>"
    assert mask_secret("short") == "********"
    assert mask_secret("sk-proj-1234567890abcdef") == "sk-...cdef"


def test_sanitize_sensitive_headers():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-proj-1234567890abcdef",
        "x-api-key": "secret-api-key-9999",
    }
    sanitized = sanitize_sensitive_headers(headers)
    assert sanitized["Content-Type"] == "application/json"
    assert sanitized["Authorization"] == "Bea...cdef"
    assert sanitized["x-api-key"] == "sec...9999"


def test_mock_llm_client_queued_responses():
    client = MockLLMClient(
        canned_responses=[
            '{"label": "True", "confidence": 0.9}',
            '{"label": "False", "confidence": 0.8}',
        ]
    )
    resp1 = client.generate("Prompt 1")
    assert "True" in resp1.content
    assert resp1.model == "mock-model"

    resp2 = client.generate("Prompt 2")
    assert "False" in resp2.content

    # Falls back to default when empty
    resp3 = client.generate("Prompt 3")
    assert "Unverified" in resp3.content

    assert len(client.history) == 3
    assert client.history[0]["prompt"] == "Prompt 1"


def test_mock_llm_client_generator_callable():
    def custom_gen(prompt: str, system: str | None) -> str:
        return f"Echo: {prompt}"

    client = MockLLMClient(response_generator=custom_gen)
    resp = client.generate("Check this statement.")
    assert resp.content == "Echo: Check this statement."


def test_mock_llm_client_simulate_exception():
    client = MockLLMClient(canned_responses=[RuntimeError("Simulated LLM connection error")])
    with pytest.raises(RuntimeError, match="Simulated LLM connection error"):
        client.generate("Test prompt")


def test_api_client_secret_safety(monkeypatch):
    monkeypatch.setenv("TEST_VERITAS_API_KEY", "sk-secret-key-12345678")
    client = APILLMClient(api_key_env_var="TEST_VERITAS_API_KEY")

    # Repr must NEVER expose the actual key
    repr_str = repr(client)
    assert "sk-secret-key-12345678" not in repr_str
    assert "key_set=True" in repr_str


def test_api_client_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VERITAS_API_KEY", raising=False)
    client = APILLMClient(api_key_env_var="NONEXISTENT_KEY")
    with pytest.raises(ValueError, match="API key not found"):
        client.generate("Test prompt")


@patch("urllib.request.urlopen")
def test_api_client_mocked_http_openai(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": '{"label": "True", "confidence": 0.95}'}}],
            "usage": {"total_tokens": 42},
        }
    ).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = APILLMClient(api_key="sk-test-dummy-key-999", provider="openai")
    resp = client.generate("Is water wet?", system_prompt="You are fact checker.")

    assert resp.content == '{"label": "True", "confidence": 0.95}'
    assert resp.model == "gpt-4o-mini"
    assert resp.usage["total_tokens"] == 42


def test_local_llm_client_mock_pipeline():
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"generated_text": [{"content": '{"label": "True"}'}]}]

    client = LocalLLMClient(
        model_name="test-local-model",
        pipeline_instance=mock_pipeline,
    )
    resp = client.generate("Verify statement.")
    assert resp.content == '{"label": "True"}'
    assert mock_pipeline.called


def test_create_llm_client_factory():
    mock_c = create_llm_client("mock")
    assert isinstance(mock_c, MockLLMClient)

    api_c = create_llm_client("api", config={"api_key": "dummy-key"})
    assert isinstance(api_c, APILLMClient)

    local_c = create_llm_client("local")
    assert isinstance(local_c, LocalLLMClient)

    with pytest.raises(ValueError, match="Unsupported LLM backend"):
        create_llm_client("unknown_backend")


def test_trace_recorder(tmp_path):
    recorder = TraceRecorder(trace_dir=tmp_path, enabled=True)
    resp = MockLLMClient().generate("Test prompt")
    record = recorder.record(
        prompt="Test prompt",
        system_prompt="System prompt",
        response=resp,
        claim="Earth is spherical.",
        claim_date="2020-01-01",
        supplied_evidence_ids=["ev_1"],
        attempt=1,
        parsed_result={"label": "True"},
    )

    assert record["claim"] == "Earth is spherical."
    assert record["attempt"] == 1
    assert record["parsed_result"] == {"label": "True"}

    # Check written file
    files = list(tmp_path.glob("verification_traces_*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert data["claim"] == "Earth is spherical."
    assert data["prompt"] == "Test prompt"
