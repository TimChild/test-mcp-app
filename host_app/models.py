from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

import reflex as rx
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import add_messages
from pydantic import BaseModel


class InputState(BaseModel):
    question: str
    conversation_id: str | None = None


class FullGraphState(BaseModel):
    question: str
    previous_messages: list[BaseMessage] = []
    response_messages: Annotated[list[AnyMessage], add_messages]
    tools: list[BaseTool] = []
    conversation_id: str | None = None


class UpdateTypes(StrEnum):
    start = "start"
    preprocess = "preprocess"
    graph_start = "graph-start"
    ai_delta = "ai-delta"
    ai_message_end = "ai-message-end"
    tool_start = "tool-start"
    tool_end = "tool-end"
    graph_end = "graph-end"
    end = "end"


@dataclass
class GraphUpdate:
    type_: UpdateTypes
    delta: str = ""
    name: str = ""
    data: Any = None


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


class ToolInfo(rx.Base):
    name: str
    description: str


class McpServerInfo(rx.Base):
    """Information about an MCP server."""

    name: str
    tools: list[ToolInfo]
