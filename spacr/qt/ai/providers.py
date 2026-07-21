"""
Provider abstraction — one class per AI vendor, all exposing a
`stream_chat(messages, system)` generator that yields text chunks.

The vendor SDKs (`anthropic`, `openai`, `google-generativeai`) are
optional. If a provider's SDK isn't installed the provider still
appears in `list_providers()` (so the UI can render a stub with an
install hint) but `is_configured()` returns False and `stream_chat()`
raises RuntimeError.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional

from . import keys


class ChatProvider(ABC):
    name: str = ""            # short id: "anthropic" / "openai" / "google"
    label: str = ""           # human-readable label
    default_model: str = ""
    install_hint: str = ""    # e.g. "pip install anthropic"

    def is_configured(self) -> bool:
        """True when a key is available AND the vendor SDK imports."""
        if not self.is_sdk_available():
            return False
        return bool(keys.get_key(self.name))

    def is_sdk_available(self) -> bool:
        try:
            self._import_sdk()
            return True
        except Exception:
            return False

    def source_of_key(self) -> str:
        return keys.source_of(self.name)

    @abstractmethod
    def _import_sdk(self):
        ...

    @abstractmethod
    def stream_chat(self, messages: List[Dict], system: str = "",
                     model: Optional[str] = None) -> Iterator[str]:
        """Yield text chunks streaming from the vendor. `messages` is a
        list of {role: 'user'/'assistant', content: str} dicts."""


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicProvider(ChatProvider):
    name = "anthropic"
    label = "Anthropic (Claude)"
    default_model = "claude-sonnet-5"
    install_hint = "pip install anthropic"

    def _import_sdk(self):
        import anthropic  # noqa
        return anthropic

    def stream_chat(self, messages, system="", model=None):
        anthropic = self._import_sdk()
        api_key = keys.get_key(self.name)
        if not api_key:
            raise RuntimeError("No Anthropic API key configured.")
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = dict(
            model=model or self.default_model,
            max_tokens=2048,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


# ---------------------------------------------------------------------------
# OpenAI (ChatGPT)
# ---------------------------------------------------------------------------

class OpenAIProvider(ChatProvider):
    name = "openai"
    label = "OpenAI (GPT)"
    default_model = "gpt-4o-mini"
    install_hint = "pip install openai"

    def _import_sdk(self):
        import openai  # noqa
        return openai

    def stream_chat(self, messages, system="", model=None):
        openai = self._import_sdk()
        api_key = keys.get_key(self.name)
        if not api_key:
            raise RuntimeError("No OpenAI API key configured.")
        client = openai.OpenAI(api_key=api_key)
        combined = []
        if system:
            combined.append({"role": "system", "content": system})
        combined.extend(messages)
        stream = client.chat.completions.create(
            model=model or self.default_model,
            messages=combined,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(ChatProvider):
    name = "google"
    label = "Google Gemini"
    default_model = "gemini-2.0-flash-exp"
    install_hint = "pip install google-genai"

    def _import_sdk(self):
        from google import genai  # noqa  — the new supported SDK
        return genai

    def stream_chat(self, messages, system="", model=None):
        genai = self._import_sdk()
        api_key = keys.get_key(self.name)
        if not api_key:
            raise RuntimeError("No Google API key configured.")
        client = genai.Client(api_key=api_key)

        # Convert chat messages into the new SDK's Content format.
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        cfg = None
        try:
            from google.genai import types as gtypes
            cfg = gtypes.GenerateContentConfig(system_instruction=system) \
                if system else None
        except Exception:
            cfg = None

        stream = client.models.generate_content_stream(
            model=model or self.default_model,
            contents=contents,
            config=cfg,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PROVIDERS: List[ChatProvider] = [
    AnthropicProvider(),
    OpenAIProvider(),
    GeminiProvider(),
]


def list_providers() -> List[ChatProvider]:
    return list(_PROVIDERS)


def configured_providers() -> List[ChatProvider]:
    return [p for p in _PROVIDERS if p.is_configured()]


def get_provider(name: str) -> Optional[ChatProvider]:
    for p in _PROVIDERS:
        if p.name == name:
            return p
    return None
