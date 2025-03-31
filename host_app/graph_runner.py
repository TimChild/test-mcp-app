"""Regular graph version of langgraph."""

import logging
import uuid
from typing import AsyncIterator

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)
from langchain_core.runnables.schema import EventData
from langgraph.graph.graph import CompiledGraph
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application

from .graph import make_graph
from .models import GraphUpdate, UpdateTypes


class GraphRunner:
    def __init__(
        self,
        graph: CompiledGraph | None = None,
        mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
    ) -> None:
        self.graph = graph or make_graph()
        self.mcp_client = mcp_client

    async def astream_events(
        self, input: BaseModel, thread_id: str | None = None
    ) -> AsyncIterator[GraphUpdate]:
        """Run the graph, yield events converted to GraphUpdates."""
        thread_id = thread_id or str(uuid.uuid4())
        yield GraphUpdate(type_=UpdateTypes.graph_start, data=thread_id)
        async for event in self.graph.astream_events(
            input=input,
            config={"configurable": {"thread_id": thread_id}},
        ):
            event_type = event["event"]
            event_data: EventData = event["data"]
            match event_type:
                case "on_chat_model_stream":
                    chunk = event_data.get("chunk", None)
                    if chunk:
                        assert isinstance(chunk, AIMessageChunk)
                        content = chunk.content
                        assert isinstance(content, str)
                        yield GraphUpdate(type_=UpdateTypes.ai_delta, delta=content)
                case "on_chat_model_end":
                    chunk = event_data.get("output", None)
                    assert isinstance(chunk, AIMessage)
                    yield GraphUpdate(type_=UpdateTypes.ai_message_end)
                case "on_tool_start":
                    chunk = event_data.get("input", None)
                    assert isinstance(chunk, dict)
                    yield GraphUpdate(type_=UpdateTypes.tool_start, name=event["name"], data=chunk)
                case "on_tool_end":
                    chunk = event_data.get("output", None)
                    assert isinstance(chunk, ToolMessage)
                    yield GraphUpdate(
                        type_=UpdateTypes.tool_end, name=event["name"], data=str(chunk.content)
                    )
                case _:
                    logging.debug(f"Ignoring event: {event_type}")
        yield GraphUpdate(type_=UpdateTypes.graph_end)
