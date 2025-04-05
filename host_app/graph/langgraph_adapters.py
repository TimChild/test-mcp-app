"""Regular graph version of langgraph."""

import logging
import uuid
from typing import Any, AsyncIterator, Iterator, Literal, Protocol, TypeGuard

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application
from host_app.models import (
    AIEndUpdate,
    AIStartUpdate,
    AIStreamUpdate,
    FullGraphState,
    GeneralUpdate,
    GraphMetadata,
    GraphUpdate,
    ToolCallInfo,
    ToolEndUpdate,
    ToolsStartUpdate,
    UpdateTypes,
)

from .functional_implementation import make_graph as make_functional_graph
from .graph_implementation import make_graph as make_standard_graph


class LgEvent(BaseModel):
    """Event emitted by the graph."""

    mode: Literal["values", "messages"]
    data: Any


class GraphUpdateError(Exception):
    pass


class EventsToUpdatesHandlerProtocol(Protocol):
    def handle_stream_event(self, event: LgEvent) -> Iterator[GraphUpdate]:
        """Handle a stream event from the graph."""
        raise NotImplementedError("handle_stream_event not implemented")


class FunctionalAdapter:
    pass


class GraphAdapter:
    """Adapter between langgraph graph and rest of app.

    Use this to call langgraph graphs and convert events/values to a representation
    used by the rest of the app (e.g. rx.Base models).

    This is primarily to protect against the rapidly changing langchain ecosystem.
    Avoid relying on langgraph/langchain throughout app so that this is a centralized
    location to update code on breaking changes.
    """

    def __init__(
        self,
        graph: CompiledGraph | None = None,
        mcp_client: MultiMCPClient = Provide[Application.mcp_client],
        use_functional_graph: bool = Provide[Application.config.use_functional_graph],
    ) -> None:
        if graph is not None:
            self.graph = graph
        else:
            if use_functional_graph:
                self.graph = make_functional_graph()
            else:
                self.graph = make_standard_graph()

        self.mcp_client = mcp_client

    def _make_runnable_config(self, thread_id: str | None = None) -> RunnableConfig:
        return RunnableConfig(
            configurable={"thread_id": thread_id or str(uuid.uuid4())},
        )

    async def ainvoke(self, input: BaseModel, thread_id: str | None = None) -> FullGraphState:
        result = await self.graph.ainvoke(input=input, config=self._make_runnable_config(thread_id))
        return FullGraphState.model_validate(result)

    async def astream_updates(
        self,
        input: BaseModel,
        thread_id: str | None = None,
        events_to_updates_handler: EventsToUpdatesHandlerProtocol | None = None,
    ) -> AsyncIterator[GraphUpdate]:
        """Run the graph, yield events converted to GraphUpdates.

        Updates:
            - GeneralUpdate: Graph Start
            - AIStartUpdate: AI message start
            - AIStreamUpdate/AIStreamToolUpdate: AI message stream
            - AIEndUpdate: AI message end
            [if tool calls]
            - ToolStartUpdate: Tool start
            - ToolEndUpdate: Tool end
            [back to AI updates]
            [possible loop back to tool calls]
            - GeneralUpdate: Graph End
        """
        yield GeneralUpdate(type_=UpdateTypes.graph_start, data=thread_id)

        stream_handler = events_to_updates_handler or MessagesStreamHandler()

        async for event in self.graph.astream(
            input=input,
            config=self._make_runnable_config(thread_id),
            stream_mode=["messages", "values"],  # otherwise defaults to only "values"
        ):
            assert isinstance(event, tuple)
            assert len(event) == 2
            event = LgEvent(mode=event[0], data=event[1])
            for update in stream_handler.handle_stream_event(event):
                yield update

        yield GeneralUpdate(type_=UpdateTypes.graph_end)


class MessagesStreamHandler(EventsToUpdatesHandlerProtocol):
    """Convert a stream of message chunk events to updates."""

    def __init__(self) -> None:
        self.streaming_messages: dict[str, AIMessageChunk] = {}

    def handle_stream_event(self, event: LgEvent) -> Iterator[GraphUpdate]:
        """Handle a stream event from the graph."""
        if event.mode != "messages":
            return

        # Extract data from message event
        m: AIMessageChunk | ToolMessage = event.data[0]

        # Ensure it has an id present
        if m.id is None:
            m.id = str(uuid.uuid4())

        m_id: str = m.id

        if self.is_ai_message(m):
            if self.is_new_ai_message(m_id):
                yield AIStartUpdate(
                    m_id=m_id, metadata=GraphMetadata(node=event.data[1]["langgraph_node"])
                )
                self.streaming_messages[m_id] = m

            self.update_streaming_message(m_id, m)

            if self.has_content_chunk(m):
                content = self.ensure_content_is_str(m.content)
                yield AIStreamUpdate(m_id=m_id, delta=content)

            if self.has_tool_call_chunk(m):
                # Not streaming tool call part for now
                pass

            if self.is_message_finish(m):
                full_message = self.streaming_messages.pop(m_id)
                yield AIEndUpdate(m_id=m_id, response=full_message)
                if self.has_tool_calls(full_message):
                    yield self.make_tool_start_update(full_message)
        elif self.is_tool_message(m):
            yield ToolEndUpdate(
                tool_response=m,
            )
        else:
            logging.error(f"Ignoring unexpected message update: {m}")

    @staticmethod
    def is_ai_message(m: AnyMessage) -> TypeGuard[AIMessageChunk]:
        return isinstance(m, AIMessageChunk)

    @staticmethod
    def is_tool_message(m: AnyMessage) -> TypeGuard[ToolMessage]:
        return isinstance(m, ToolMessage)

    def is_new_ai_message(self, m_id: str) -> bool:
        return m_id not in self.streaming_messages

    def update_streaming_message(self, m_id: str, m: AIMessageChunk) -> None:
        self.streaming_messages[m_id] += m  # pyright: ignore[reportArgumentType]

    @staticmethod
    def has_content_chunk(m: AIMessageChunk) -> bool:
        return True if m.content else False

    @staticmethod
    def ensure_content_is_str(content: str | list[str | dict]) -> str:
        if not isinstance(content, str):
            raise GraphUpdateError(f"Expected str content, got {type(content)}")
        return content

    @staticmethod
    def has_tool_call_chunk(m: AIMessageChunk) -> bool:
        return True if m.tool_call_chunks else False

    @staticmethod
    def is_message_finish(m: AIMessageChunk) -> bool:
        return m.response_metadata.get("finish_reason", None) is not None

    @staticmethod
    def has_tool_calls(m: AIMessage) -> bool:
        return True if m.tool_calls else False

    @staticmethod
    def make_tool_start_update(full_message: AIMessage) -> ToolsStartUpdate:
        return ToolsStartUpdate(
            calls=[
                ToolCallInfo(
                    name=call["name"],
                    args=call["args"],
                    id=call["id"],
                )
                for call in full_message.tool_calls
            ]
        )
