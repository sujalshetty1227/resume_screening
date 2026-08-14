"""Thin, provider-agnostic LLM client.

Deliberately raw `requests` instead of a vendor SDK: three of the four
supported backends speak an OpenAI-compatible chat schema (Anthropic is the
exception, handled inline), so one small module removes four heavyweight SDK
dependencies and lets a reviewer switch provider with one env var.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests

from .config import (LLM_MAX_RETRIES, LLM_PROVIDER, LLM_TEMPERATURE,
                     LLM_TIMEOUT, PROVIDERS)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = (provider or LLM_PROVIDER).lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{self.provider}'. "
                           f"Choose one of: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        self.url = cfg["url"]
        self.model = os.getenv(cfg["model_env"] or "", "") or cfg["default_model"]
        self.api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else None
        if cfg["key_env"] and not self.api_key:
            raise LLMError(
                f"{cfg['key_env']} is not set. Add it to .env, or run the agent "
                f"with `--mode offline` which needs no API key at all."
            )

    # -- request shaping ---------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        if self.provider == "anthropic":
            return {"x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"}
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, system: str, user: str) -> Dict[str, Any]:
        if self.provider == "anthropic":
            return {"model": self.model, "max_tokens": 2048,
                    "temperature": LLM_TEMPERATURE, "system": system,
                    "messages": [{"role": "user", "content": user}]}
        return {"model": self.model, "temperature": LLM_TEMPERATURE,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}

    @staticmethod
    def _content(provider: str, body: Dict[str, Any]) -> str:
        if provider == "anthropic":
            return "".join(b.get("text", "") for b in body.get("content", []))
        return body["choices"][0]["message"]["content"]

    # -- public API --------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        """Single completion, retrying only genuinely transient failures.

        429 and 5xx are retried with exponential backoff; other 4xx responses
        (invalid key, unknown model) are raised immediately, since retrying a
        request the server has already rejected on its merits cannot help.
        """
        last: Optional[Exception] = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                resp = requests.post(self.url, headers=self._headers(),
                                     json=self._payload(system, user),
                                     timeout=LLM_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return self._content(self.provider, resp.json())
            except requests.HTTPError as exc:
                # 4xx other than 429 is our fault (bad key, bad model name).
                # Retrying wastes 3 seconds and cannot succeed.
                raise LLMError(f"{self.provider} rejected the request: {exc}") from exc
            except (requests.RequestException, LLMError, KeyError, ValueError) as exc:
                last = exc
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)          # 1s, 2s, 4s
        raise LLMError(f"LLM call failed after {LLM_MAX_RETRIES} attempts: {last}")

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        """Completion parsed as JSON, tolerant of markdown-fenced output.

        Models wrap JSON in ```json fences often enough that stripping them is
        cheaper and more reliable than begging the prompt not to.
        """
        raw = self.complete(system, user)
        return parse_json_loose(raw)


def parse_json_loose(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMError(f"model did not return JSON. Got: {raw[:300]}")
        return json.loads(match.group(0))
