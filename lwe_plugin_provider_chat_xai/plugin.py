"""LWE chat xai provider integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_xai import ChatXAI
from typing_extensions import override

from lwe.core.provider import PresetValue, Provider

XAI_DEFAULT_MODEL = "grok-4.7"


class CustomChatXai(ChatXAI):

    def __init__(self, **kwargs: Any) -> None:
        if "model" not in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = XAI_DEFAULT_MODEL
        super().__init__(**kwargs)

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat_xai"


class ProviderChatXai(Provider):
    """
    Access to chat xAI models
    """

    @property
    @override
    def capabilities(self) -> dict[str, bool]:
        return {
            "chat": True,
            "validate_models": True,
        }

    @property
    @override
    def default_model(self) -> str:
        return XAI_DEFAULT_MODEL

    @property
    @override
    def static_models(self) -> dict[str, dict[str, int]]:
        return {
            "grok-4.7": {
                "max_tokens": 500000,
            },
            "grok-4.6": {
                "max_tokens": 500000,
            },
            "grok-4.5": {
                "max_tokens": 500000,
            },
            "grok-4.5-latest": {
                "max_tokens": 500000,
            },
            "grok-build-latest": {
                "max_tokens": 500000,
            },
            "grok-4.3": {
                "max_tokens": 1000000,
            },
            "grok-4.3-latest": {
                "max_tokens": 1000000,
            },
            "grok-4.20-0309-reasoning": {
                "max_tokens": 1000000,
            },
            "grok-4.20": {
                "max_tokens": 1000000,
            },
            "grok-4.20-0309": {
                "max_tokens": 1000000,
            },
            "grok-4.20-reasoning": {
                "max_tokens": 1000000,
            },
            "grok-4.20-reasoning-latest": {
                "max_tokens": 1000000,
            },
            "grok-4.20-0309-non-reasoning": {
                "max_tokens": 1000000,
            },
            "grok-4.20-non-reasoning": {
                "max_tokens": 1000000,
            },
            "grok-4.20-non-reasoning-latest": {
                "max_tokens": 1000000,
            },
            "grok-build-0.1": {
                "max_tokens": 256000,
            },
        }

    @override
    def prepare_messages_method(
        self,
    ) -> Callable[[list[dict[str, Any]]], list[BaseMessage]]:
        return self.prepare_messages_for_llm_chat

    @override
    def llm_factory(self) -> type[CustomChatXai]:
        return CustomChatXai

    @override
    def customization_config(self) -> dict[str, Any]:
        return {
            "verbose": PresetValue(bool),
            "model_name": PresetValue(str, options=self.available_models),
            "temperature": PresetValue(float, min_value=0.0, max_value=2.0),
            "reasoning_effort": PresetValue(
                str,
                options=["none", "low", "medium", "high", "xhigh"],
                include_none=True,
            ),
            "xai_api_base": PresetValue(str, include_none=True),
            "xai_api_key": PresetValue(str, include_none=True, private=True),
            "request_timeout": PresetValue(int),
            "max_retries": PresetValue(int, 1, 10),
            "n": PresetValue(int, 1, 10),
            "max_tokens": PresetValue(int, include_none=True),
            "top_p": PresetValue(float, min_value=0.0, max_value=1.0),
            "presence_penalty": PresetValue(float, min_value=-2.0, max_value=2.0),
            "frequency_penalty": PresetValue(float, min_value=-2.0, max_value=2.0),
            "seed": PresetValue(int, include_none=True),
            "logprobs": PresetValue(bool, include_none=True),
            "top_logprobs": PresetValue(
                int, min_value=0, max_value=20, include_none=True
            ),
            "logit_bias": dict,
            "stop": PresetValue(str, include_none=True),
            "tools": None,
            "tool_choice": None,
            "model_kwargs": {
                "response_format": dict,
                "user": PresetValue(str),
            },
        }

    def format_streaming_chunk(self, chunk: AIMessageChunk, mode: str = "") -> str:
        output = ""
        if "reasoning_content" in chunk.additional_kwargs:
            if mode != "thinking":
                output += "[THINKING]\n\n"
            output += chunk.additional_kwargs["reasoning_content"]
        else:
            if mode != "text":
                output += "\n\n[ANSWER]\n\n"
            output += chunk.text
        return output

    def handle_non_streaming_response(self, response: AIMessage) -> AIMessage:
        content = ""
        if "reasoning_content" in response.additional_kwargs:
            content = f"""
[THINKING]

{response.additional_kwargs['reasoning_content']}

[ANSWER]

"""
        content += response.text
        response.content = content
        return response

    def handle_streaming_chunk(
        self,
        chunk: AIMessageChunk | str,
        previous_chunks: list[AIMessageChunk | str],
    ) -> str:
        if isinstance(chunk, str):
            return chunk
        mode = self.get_last_streaming_mode(previous_chunks)
        output = self.format_streaming_chunk(chunk, mode)
        return output

    def get_last_streaming_mode(
        self, previous_chunks: list[AIMessageChunk | str]
    ) -> str:
        try:
            last_chunk = previous_chunks[-1]
            if (
                not isinstance(last_chunk, str)
                and "reasoning_content" in last_chunk.additional_kwargs
            ):
                return "thinking"
            return "text"
        except Exception:
            return "text"
