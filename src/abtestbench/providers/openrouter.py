"""OpenRouter LLM provider using OpenAI-compatible API."""

import os
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from ..config import OpenRouterConfig
from ..models.response import LLMResponse, ToolCall


# Default models available on OpenRouter
AVAILABLE_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large",
]


class OpenRouterProvider:
    """OpenRouter API provider with tool use support."""

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        self.config = config or OpenRouterConfig()
        if not self.config.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY env var.")

        # Configure HTTP client with proxy if available
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url:
            http_client = httpx.AsyncClient(proxy=proxy_url)
        else:
            http_client = None

        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            http_client=http_client,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: Optional[list[dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a completion with optional tool use."""
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)

        return self._parse_response(response, model)

    def _parse_response(self, response: Any, model: str) -> LLMResponse:
        """Parse OpenAI-format response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        tool_calls = None

        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                # Parse arguments - they come as a JSON string
                import json

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model,
            provider="openrouter",
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            stop_reason=choice.finish_reason,
        )
