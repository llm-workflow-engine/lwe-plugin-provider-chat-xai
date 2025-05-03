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

    # def format_thinking_content(self, content: list[dict[str, str]], mode: str = ""):
    #     output = ""
    #     for part in content:
    #         if 'thinking' in part:
    #             if mode != "thinking":
    #                 mode = "thinking"
    #                 output += "[THINKING]\n\n"
    #             output += part['thinking']
    #         elif 'text' in part:
    #             if mode != "text":
    #                 mode = "text"
    #                 output += "\n\n[ANSWER]\n\n"
    #             output += part['text']
    #     return output
    #
    # def handle_non_streaming_response(self, response: AIMessage):
    #     if isinstance(response.content, list):
    #         response.content = self.format_thinking_content(response.content)
    #     return response
    #
    # def handle_streaming_chunk(self, chunk: AIMessageChunk | str, previous_chunks: list[AIMessageChunk | str]) -> str:
    #     if isinstance(chunk, str):
    #         return chunk
    #     elif isinstance(chunk.content, str):
    #         return chunk.content
    #     return self.handle_streaming_thinking_chunk(chunk, previous_chunks)
    #
    # def get_last_streaming_mode(self, previous_chunks: list[AIMessageChunk | str]) -> str:
    #     for chunk in reversed(previous_chunks):
    #         if isinstance(chunk, AIMessageChunk):
    #             content = chunk.content
    #             if isinstance(content, list):
    #                 for part in reversed(content):
    #                     if part['type'] in ['thinking', 'text']:
    #                         return part['type']
    #     return ""
    #
    # def handle_streaming_thinking_chunk(self, chunk: AIMessageChunk, previous_chunks: list[AIMessageChunk | str]) -> str:
    #     mode = self.get_last_streaming_mode(previous_chunks)
    #     output = self.format_thinking_content(chunk.content, mode)
    #     return output
