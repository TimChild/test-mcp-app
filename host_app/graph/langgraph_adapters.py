"""Regular graph version of langgraph."""

import uuid
from typing import Any, AsyncIterator, Literal

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
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
    GraphUpdate,
    ToolCallInfo,
    ToolEndUpdate,
    ToolStartUpdate,
    UpdateTypes,
)

from .graph import make_graph


class LgEvent(BaseModel):
    """Event emitted by the graph."""

    mode: Literal["values", "messages"]
    data: Any


class GraphUpdateError(Exception):
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
    ) -> None:
        self.graph = graph or make_graph()
        self.mcp_client = mcp_client

    def _make_runnable_config(self, thread_id: str | None = None) -> RunnableConfig:
        return RunnableConfig(
            configurable={"thread_id": thread_id or str(uuid.uuid4())},
        )

    async def ainvoke(self, input: BaseModel, thread_id: str | None = None) -> FullGraphState:
        result = await self.graph.ainvoke(input=input, config=self._make_runnable_config(thread_id))
        return FullGraphState.model_validate(result)

    async def astream_updates(
        self, input: BaseModel, thread_id: str | None = None
    ) -> AsyncIterator[GraphUpdate]:
        """Run the graph, yield events converted to GraphUpdates."""
        yield GeneralUpdate(type_=UpdateTypes.graph_start, data=thread_id)

        current_message_type: None | Literal["ai", "tool"] = None

        async for event in self.graph.astream(
            input=input,
            config=self._make_runnable_config(thread_id),
            stream_mode=["messages", "values"],  # defaults to "values"
        ):
            assert isinstance(event, tuple)
            assert len(event) == 2
            event = LgEvent(mode=event[0], data=event[1])
            match event.mode:
                case "values":
                    value_update = event.data
                    assert isinstance(value_update, dict)
                    # Dict of updates from each graph node
                    match current_message_type:
                        case None:
                            yield GeneralUpdate(type_=UpdateTypes.value_update, data=value_update)
                        case "tool":
                            # This is the end of the tool call
                            current_message_type = None
                            if "response_messages" not in value_update:
                                raise GraphUpdateError(
                                    f"Expected dict with key 'response_messages', got {value_update.keys()}"
                                )
                            whole_messages = value_update["response_messages"]
                            if not all(isinstance(msg, ToolMessage) for msg in whole_messages):
                                raise GraphUpdateError(
                                    f"Expected all messages to be ToolMessage, got {whole_messages}"
                                )
                            yield ToolEndUpdate(
                                tool_responses=whole_messages,
                            )
                        case "ai":
                            # This is the end of the AI message
                            if "response_messages" not in value_update:
                                raise GraphUpdateError(
                                    f"Expected dict with key 'response_messages', got {value_update.keys()}"
                                )
                            whole_messages = value_update["response_messages"]
                            if not len(whole_messages) == 1:
                                raise GraphUpdateError(
                                    f"Expected one new ai message, got {len(whole_messages)}"
                                )
                            ai_message = whole_messages[-1]
                            assert isinstance(ai_message, AIMessage)

                            if ai_message.tool_calls:
                                current_message_type = "tool"
                                yield ToolStartUpdate(
                                    calls=[
                                        ToolCallInfo(
                                            name=call["name"],
                                            args=call["args"],
                                            id=call["id"],
                                        )
                                        for call in ai_message.tool_calls
                                    ],
                                )
                            else:
                                current_message_type = None

                            yield AIEndUpdate(
                                response=ai_message,
                            )
                case "messages":
                    assert isinstance(event.data, tuple)
                    assert len(event.data) == 2
                    chunk, metadata = event.data
                    assert isinstance(metadata, dict)
                    if isinstance(chunk, AIMessageChunk):
                        if not isinstance(chunk.content, str):
                            raise TypeError(f"Expected str, got {type(chunk.content)}")

                        if current_message_type is None:
                            current_message_type = "ai"
                            yield AIStartUpdate(
                                delta=chunk.content,
                                metadata=metadata,
                            )
                        else:
                            yield AIStreamUpdate(
                                delta=chunk.content,
                                metadata=metadata,
                            )
                    elif isinstance(chunk, ToolMessage):
                        # Handled in the values mode
                        pass
                    else:
                        raise TypeError(f"Expected AIMessageChunk, got {type(chunk)}")
        yield GeneralUpdate(type_=UpdateTypes.graph_end)

        # async for event in self.graph.astream_events(
        #     input=input,
        #     config=self._make_runnable_config(thread_id),
        # ):
        #     event_type = event["event"]
        #     event_data: EventData = event["data"]
        #     match event_type:
        #         case "on_chat_model_stream":
        #             chunk = event_data.get("chunk", None)
        #             if chunk:
        #                 assert isinstance(chunk, AIMessageChunk)
        #                 content = chunk.content
        #                 assert isinstance(content, str)
        #                 yield GraphUpdate(type_=UpdateTypes.ai_delta, delta=content)
        #         case "on_chat_model_end":
        #             chunk = event_data.get("output", None)
        #             assert isinstance(chunk, AIMessage)
        #             yield GraphUpdate(type_=UpdateTypes.ai_message_end)
        #         case "on_tool_start":
        #             chunk = event_data.get("input", None)
        #             assert isinstance(chunk, dict)
        #             yield GraphUpdate(type_=UpdateTypes.tool_start, name=event["name"], data=chunk)
        #         case "on_tool_end":
        #             chunk = event_data.get("output", None)
        #             assert isinstance(chunk, ToolMessage)
        #             yield GraphUpdate(
        #                 type_=UpdateTypes.tool_end, name=event["name"], data=str(chunk.content)
        #             )
        #         case _:
        #             logging.debug(f"Ignoring event: {event_type}")
        # yield GraphUpdate(type_=UpdateTypes.graph_end)
