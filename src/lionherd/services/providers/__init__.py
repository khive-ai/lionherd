# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Provider-specific endpoint implementations.

Provides specialized Endpoint subclasses for:
- Anthropic Claude Messages API
- OpenAI Chat Completions API
- OpenAI-compatible providers: OpenRouter, Groq, Nvidia NIM
"""

from .anthropic_messages import AnthropicMessagesEndpoint, create_anthropic_config
from .oai_compatible import (
    GroqChatEndpoint,
    NvidiaNimChatEndpoint,
    OpenRouterChatEndpoint,
    create_groq_config,
    create_nvidia_nim_config,
    create_openrouter_config,
)
from .openai_chat import OpenAIChatEndpoint, create_openai_config

__all__ = (
    "AnthropicMessagesEndpoint",
    "GroqChatEndpoint",
    "NvidiaNimChatEndpoint",
    "OpenAIChatEndpoint",
    "OpenRouterChatEndpoint",
    "create_anthropic_config",
    "create_groq_config",
    "create_nvidia_nim_config",
    "create_openai_config",
    "create_openrouter_config",
)
