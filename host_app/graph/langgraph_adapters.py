"""Adapter from langgraph to updates usable by the app.

This is effectively a buffer layer between the very fast changing langchain/langgraph
ecosystem and the rest of the app.

This makes it easier to address breaking changes.

This also translates the lg events into more useful updates for triggering UI events.
"""

import logging
import uuid
from typing import Any, AsyncIterator, Iterator, Literal, Protocol, TypeGuard

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.pregel import Pregel
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application
from host_app.graph.functional_implementation import OutputState
from host_app.models import (
    AIEndUpdate,
    AIStartUpdate,
    AIStreamUpdate,
    GeneralUpdate,
    GraphMetadata,
    GraphUpdate,
    ToolCallInfo,
    ToolEndUpdate,
    ToolsStartUpdate,
    UpdateTypes,
)


class LgEvent(BaseModel):
    """Structure of event emitted by langgraph."""

    mode: Literal["values", "messages"]
    data: Any


class GraphUpdateError(Exception):
    """Base for any errors raised during conversion of events to updates."""

    pass


class EventsToUpdatesHandlerProtocol(Protocol):
    """Basic protocol for converting events to updates."""

    def handle_stream_event(self, event: LgEvent) -> Iterator[GraphUpdate]:
        """Take an event and return an iterable of updates."""
        raise NotImplementedError("handle_stream_event not implemented")

    def reset(self) -> None:
        """Reset the handler for a new stream."""
        raise NotImplementedError("reset not implemented")


class GraphRunAdapter:
    """Adapter for running a langgraph graph and returning custom updates instead of langgraph events."""

    def __init__(
        self,
        graph: Pregel,
        stream_handler: EventsToUpdatesHandlerProtocol | None = None,
        mcp_client: MultiMCPClient = Provide[Application.mcp_client],
    ) -> None:
        self.graph = graph
        self.stream_handler = stream_handler or MessagesStreamHandler(
            listen_nodes=["call_tools", "call_llm", "graph"]
        )
        self.mcp_client = mcp_client

    async def ainvoke(self, input: BaseModel, thread_id: str | None = None) -> OutputState:
        """Run the graph and only return the final output."""
        result = await self.graph.ainvoke(input=input, config=self._make_runnable_config(thread_id))
        return OutputState.model_validate(result)

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

        stream_handler = events_to_updates_handler or self.stream_handler
        stream_handler.reset()

        async for event in self.graph.astream(
            input=input,
            config=self._make_runnable_config(thread_id),
            stream_mode=[
                "messages",
                "values",
            ],  # otherwise defaults to only "values" but we want message chunks
        ):
            assert isinstance(event, tuple)
            assert len(event) == 2
            event = LgEvent(mode=event[0], data=event[1])
            for update in stream_handler.handle_stream_event(event):
                yield update

        yield GeneralUpdate(type_=UpdateTypes.graph_end)

    def _make_runnable_config(self, thread_id: str | None = None) -> RunnableConfig:
        return RunnableConfig(
            configurable={"thread_id": thread_id or str(uuid.uuid4())},
        )


class MessagesStreamHandler(EventsToUpdatesHandlerProtocol):
    """Convert a stream of message chunk events to updates."""

    def __init__(self, listen_nodes: list[str]) -> None:
        self.listen_nodes = listen_nodes
        self.streaming_messages: dict[str, AIMessageChunk] = {}

    def reset(self) -> None:
        """Reset the handler for a new stream."""
        self.streaming_messages = {}

    def handle_stream_event(self, event: LgEvent) -> Iterator[GraphUpdate]:
        """Handle a stream event from the graph.

        Yields:
            - AIStartUpdate: On new AI message
            - AIStreamUpdate: With delta content for AI message
            - AIEndUpdate: On AI message end
            - ToolStartUpdate: After AI has made tool calls (single update for multiple calls)
            - ToolEndUpdate: With response from tool (an update per tool response)
        """
        if event.mode != "messages":
            # Ignore non-message events
            return
        lg_metadata = event.data[1]
        if lg_metadata["langgraph_node"] not in self.listen_nodes:
            # Ignore events from nodes we are not listening to
            return

        # Extract data from message event
        m: AIMessageChunk | ToolMessage = event.data[0]
        assert isinstance(m, BaseMessage)

        # Ensure it has an id present
        if m.id is None:
            m.id = str(uuid.uuid4())

        m_id: str = m.id

        if self.is_ai_message(m):
            if self.is_new_ai_message(m_id):
                yield AIStartUpdate(
                    m_id=m_id, metadata=GraphMetadata(node=lg_metadata["langgraph_node"])
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
                    # update now because no other notification of tool calls until tool responses returned
                    yield self.make_tool_start_update(full_message)

        elif self.is_tool_message(m):
            yield ToolEndUpdate(
                tool_response=m,
            )
        else:
            node = lg_metadata["langgraph_node"]
            logging.warning(f"Ignoring unexpected message update from: {node=}, {m.type=}")

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
