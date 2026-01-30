"""OpenRouter LLM provider using OpenAI-compatible API."""

import asyncio
import os
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from ..config import OpenRouterConfig
from ..models.response import LLMResponse, ToolCall, ToolResult


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

    async def complete_with_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]],
        tool_executor: "ToolExecutor",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ) -> LLMResponse:
        """Complete with automatic tool execution loop."""
        current_messages = messages.copy()
        all_tool_results: list[ToolResult] = []

        for _ in range(max_iterations):
            response = await self.complete(
                messages=current_messages,
                model=model,
                tools=tools,
                system_prompt=system_prompt,
            )

            if not response.tool_calls:
                response.all_tool_results = all_tool_results
                return response

            # Add assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content or ""}

            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": str(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]

            current_messages.append(assistant_msg)

            # Execute each tool call
            for tool_call in response.tool_calls:
                result = await tool_executor.execute(tool_call.name, tool_call.arguments)

                tool_result = ToolResult(
                    tool_name=tool_call.name,
                    tool_input=tool_call.arguments,
                    output=result.output if hasattr(result, "output") else str(result),
                    success=result.success if hasattr(result, "success") else True,
                    error=result.error if hasattr(result, "error") else None,
                )
                all_tool_results.append(tool_result)

                # Add tool result message
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result.output,
                    }
                )

        # Max iterations reached
        final_response = await self.complete(
            messages=current_messages,
            model=model,
            tools=None,  # No more tools
            system_prompt=system_prompt,
        )
        final_response.all_tool_results = all_tool_results
        return final_response

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


class ToolExecutor:
    """Executes tools called by the LLM."""

    def __init__(self):
        self.tools: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        """Register a tool handler."""
        self.tools[name] = handler

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        if name not in self.tools:
            from ..models.response import ToolResult

            return ToolResult(
                tool_name=name,
                tool_input=arguments,
                output=f"Unknown tool: {name}",
                success=False,
                error=f"Tool '{name}' not found",
            )

        handler = self.tools[name]

        if asyncio.iscoroutinefunction(handler.execute):
            return await handler.execute(arguments)
        else:
            return handler.execute(arguments)
