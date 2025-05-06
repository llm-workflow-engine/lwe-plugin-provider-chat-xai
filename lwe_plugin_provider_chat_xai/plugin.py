import json
from langchain_xai import ChatXAI
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import Field

from lwe.core.provider import Provider, PresetValue
from lwe.core import util

XAI_DEFAULT_MODEL = 'grok-3-mini'


class CustomChatXai(ChatXAI):

    def __init__(self, **kwargs):
        if "model" not in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = XAI_DEFAULT_MODEL
        super().__init__(**kwargs)

    @property
    def _llm_type(self):
        """Return type of llm."""
        return "chat_xai"


class ProviderChatXai(Provider):
    """
    Access to chat xAI models
    """

    @property
    def capabilities(self):
        return {
            "chat": True,
            'validate_models': True,
        }

    @property
    def default_model(self):
        return XAI_DEFAULT_MODEL

    @property
    def static_models(self):
        return {
            'grok-2-1212': {
                'max_tokens': 131072,
            },
            'grok-3-mini': {
                'max_tokens': 131072,
            },
            'grok-3-mini-beta': {
                'max_tokens': 131072,
            },
            'grok-3-mini-fast-beta': {
                'max_tokens': 131072,
            },
            'grok-3': {
                'max_tokens': 131072,
            },
            'grok-3-beta': {
                'max_tokens': 131072,
            },
            'grok-3-fast-beta': {
                'max_tokens': 131072,
            },
            'grok-beta': {
                'max_tokens': 131072,
            },
        }

    def prepare_messages_method(self):
        return self.prepare_messages_for_llm_chat

    def llm_factory(self):
        return CustomChatXai

    def customization_config(self):
       return {
           "verbose": PresetValue(bool),
           "model_name": PresetValue(str, options=self.available_models),
           "temperature": PresetValue(float, min_value=0.0, max_value=2.0),
           "reasoning_effort": PresetValue(str, options=["low", "high"], include_none=True),
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
           "top_logprobs": PresetValue(int, min_value=0, max_value=20, include_none=True),
           "logit_bias": dict,
           "stop": PresetValue(str, include_none=True),
           "tools": None,
           "tool_choice": None,
           "model_kwargs": {
               "response_format": dict,
               "user": PresetValue(str),
           },
       }

    def format_streaming_chunk(self, chunk: AIMessageChunk, mode: str = ""):
        output = ""
        if 'reasoning_content' in chunk.additional_kwargs:
            if mode != "thinking":
                output += "[THINKING]\n\n"
            output += chunk.additional_kwargs['reasoning_content']
        else:
            if mode != "text":
                output += "\n\n[ANSWER]\n\n"
            output += chunk.content
        return output

    def handle_non_streaming_response(self, response: AIMessage):
        content = ""
        if 'reasoning_content' in response.additional_kwargs:
            content = f"""
[THINKING]

{response.additional_kwargs['reasoning_content']}

[ANSWER]

"""
        content += response.content
        response.content = content
        return response

    def handle_streaming_chunk(self, chunk: AIMessageChunk | str, previous_chunks: list[AIMessageChunk | str]) -> str:
        if isinstance(chunk, str):
            return chunk
        mode = self.get_last_streaming_mode(previous_chunks)
        output = self.format_streaming_chunk(chunk, mode)
        return output

    def get_last_streaming_mode(self, previous_chunks: list[AIMessageChunk | str]) -> str:
        try:
            last_chunk = previous_chunks[-1]
            if 'reasoning_content' in last_chunk.additional_kwargs:
                return "thinking"
            return "text"
        except Exception:
            return "text"
