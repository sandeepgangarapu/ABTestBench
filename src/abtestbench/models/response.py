"""LLM response models."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    tool_input: dict[str, Any]
    output: str
    success: bool
    error: Optional[str] = None


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str
    tool_calls: Optional[list[ToolCall]] = None
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)
    stop_reason: Optional[str] = None
    all_tool_results: list[ToolResult] = Field(default_factory=list)
