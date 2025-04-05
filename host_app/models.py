from enum import StrEnum
from typing import Annotated, Any, Protocol

import reflex as rx
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import add_messages
from pydantic import BaseModel


class InputState(BaseModel):
    """State required to run the graph."""

    question: str
    conversation_id: str | None = None


class FullGraphState(BaseModel):
    """Full state used by and returned by graph."""

    question: str
    previous_messages: list[BaseMessage] = []
    response_messages: Annotated[list[AnyMessage], add_messages]
    tools: list[BaseTool] = []
    conversation_id: str | None = None


class UpdateTypes(StrEnum):
    """THe types of update that are sent back for the frontend to display."""

    preprocess = "preprocess"
    graph_start = "graph-start"
    ai_message_start = "ai-message-start"
    ai_stream = "ai-delta"
    ai_stream_tool_call = "ai-tool-call-delta"
    ai_message_end = "ai-message-end"
    tool_start = "tool-start"
    tool_end = "tool-end"
    graph_end = "graph-end"
    value_update = "value-update"


class GraphUpdate(Protocol):
    """General protocol for updates."""

    type_: UpdateTypes


class GeneralUpdate(rx.Base):
    """General update for the graph."""

    type_: UpdateTypes
    data: Any | None = None


class AIStartUpdate(rx.Base):
    """Update for start of AI messages."""

    type_ = UpdateTypes.ai_message_start
    delta: str
    metadata: dict[str, Any]


class AIStreamUpdate(rx.Base):
    """Update for streaming AI messages."""

    type_ = UpdateTypes.ai_stream
    delta: str
    metadata: dict[str, Any]


class AIEndUpdate(rx.Base):
    """Update for end of AI messages."""

    type_ = UpdateTypes.ai_message_end
    response: AIMessage


class ToolCallInfo(rx.Base):
    """Info for each tool call."""

    name: str
    args: dict[str, Any]
    id: str | None


class ToolStartUpdate(rx.Base):
    type_ = UpdateTypes.tool_start
    calls: list[ToolCallInfo]


class ToolEndUpdate(rx.Base):
    type_ = UpdateTypes.tool_end
    tool_responses: list[ToolMessage]


class ToolUse(rx.Base):
    tool_names: list[str]


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    tool_uses: list[ToolUse] = []
    answer: str


class ToolInfo(rx.Base):
    """Info for each MCP tool."""

    name: str
    description: str


class McpServerInfo(rx.Base):
    """Information about an MCP server."""

    name: str
    tools: list[ToolInfo]
