"""Regular graph version of langgraph."""

import logging
import uuid
from typing import Any, AsyncIterator, Iterator, Literal, Protocol

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    ToolMessageChunk,
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
    GraphUpdate,
    ToolCallInfo,
    ToolEndUpdate,
    ToolStartUpdate,
    UpdateTypes,
)

from .functional_langgraph import make_graph as make_functional_graph
from .graph import make_graph


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
                self.graph = make_graph()

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

        stream_handler = events_to_updates_handler or StreamHandler()

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


class StreamHandler(EventsToUpdatesHandlerProtocol):
    def __init__(self) -> None:
        self.current_message_type: None | Literal["ai", "tool"] = None

    def handle_stream_event(self, event: LgEvent) -> Iterator[GraphUpdate]:
        """Handle a stream event from the graph."""
        # Handle the event here
        for update, message_type in get_update(event, self.current_message_type):
            logging.info("iterating update")
            if message_type:
                if message_type == "clear":
                    self.current_message_type = None
                else:
                    self.current_message_type = message_type
            if update:
                yield update


def get_update(
    event: LgEvent, current_message_type: Literal["tool", "ai"] | None
) -> Iterator[tuple[GraphUpdate | None, Literal["tool", "ai", "clear"] | None]]:
    """Convert event into a GraphUpdate."""
    match event.mode:
        case "values":
            logging.info("Values update")
            # Updates at end of graph nodes
            value_update = event.data
            if not isinstance(value_update, dict):
                raise GraphUpdateError(f"Expected dict, got {type(value_update)}")
            yield from get_value_update(value_update, current_message_type)
        case "messages":
            logging.info("Messages update")
            # Streaming messages
            assert isinstance(event.data, tuple)
            assert len(event.data) == 2
            chunk, metadata = event.data
            assert isinstance(metadata, dict)
            yield from get_messages_update(chunk, metadata, current_message_type)


def get_value_update(
    value_dict: dict, current_message_type: Literal["tool", "ai"] | None
) -> Iterator[tuple[GraphUpdate, Literal["tool", "ai", "clear"] | None]]:
    def get_tool_update() -> Iterator[
        tuple[ToolStartUpdate | ToolEndUpdate, Literal["tool", "clear"]]
    ]:
        # This is the end of the tool call
        if "response_messages" not in value_dict:
            raise GraphUpdateError(
                f"Expected dict with key 'response_messages', got {value_dict.keys()}"
            )

        whole_messages = value_dict["response_messages"]
        # Need all of the latest ToolResponse messages
        tool_responses = []
        for msg in reversed(whole_messages):
            if isinstance(msg, ToolMessage):
                tool_responses.append(msg)
            else:
                print(type(msg))
                break
        tool_responses.reverse()
        if len(tool_responses) == 0:
            raise GraphUpdateError("Expected at least one new tool message")
        logging.info("Yielding ToolEnd")
        yield (
            ToolEndUpdate(
                tool_responses=tool_responses,
            ),
            "clear",
        )

    def get_ai_update() -> Iterator[tuple[GraphUpdate, Literal["ai", "tool", "clear"] | None]]:
        # This is the end of the AI message
        if "response_messages" not in value_dict:
            raise GraphUpdateError(
                f"Expected dict with key 'response_messages', got {value_dict.keys()}"
            )
        whole_messages = value_dict["response_messages"]
        # The latest message is the new AI message
        ai_message = whole_messages[-1]
        assert isinstance(ai_message, AIMessage)

        if ai_message.tool_calls:
            logging.info("Yielding AIEnd")
            yield (
                AIEndUpdate(response=ai_message),
                None,
            )
            logging.info("Yielding ToolStart")
            yield (
                ToolStartUpdate(
                    calls=[
                        ToolCallInfo(
                            name=call["name"],
                            args=call["args"],
                            id=call["id"],
                        )
                        for call in ai_message.tool_calls
                    ],
                ),
                "tool",
            )
        else:
            logging.info("Yielding final AIEnd")
            yield (
                AIEndUpdate(
                    response=ai_message,
                ),
                "clear",
            )

    # Dict of updates from each graph node
    logging.info("Value update")
    match current_message_type:
        case None:
            logging.info("General update")
            yield GeneralUpdate(type_=UpdateTypes.value_update, data=value_dict), None
        case "tool":
            logging.info("Tool update")
            yield from get_tool_update()
        case "ai":
            logging.info("AI update")
            yield from get_ai_update()


def get_messages_update(
    chunk: AIMessage | ToolMessageChunk,
    metadata: dict[str, Any],
    current_message_type: Literal["tool", "ai"] | None,
) -> Iterator[tuple[GraphUpdate | None, Literal["tool", "ai"] | None]]:
    """Convert langchain message to GraphUpdate."""
    logging.info("Message update")
    if isinstance(chunk, AIMessageChunk):
        logging.info("AI message chunk")
        if not isinstance(chunk.content, str):
            raise TypeError(f"Expected str, got {type(chunk.content)}")

        if current_message_type is None:
            yield (
                AIStartUpdate(
                    delta=chunk.content,
                    metadata=metadata,
                ),
                "ai",
            )
        else:
            yield (
                AIStreamUpdate(
                    delta=chunk.content,
                    metadata=metadata,
                ),
                None,
            )
        # if chunk.response_metadata.get("finish_reason", None):
        #     logging.info("AI message end")
        #     yield (
        #         AIEndUpdate(
        #             response=chunk.response_metadata,
        #         ),
        #         "clear",
        #     )
    elif isinstance(chunk, ToolMessage):
        logging.info("Tool message -- Do nothing with message event")
        # Handled in the values mode
        return
    else:
        logging.error(f"Unknown chunk type: {type(chunk)}")
        # raise TypeError(f"Expected AIMessageChunk, got {type(chunk)}")
        return
