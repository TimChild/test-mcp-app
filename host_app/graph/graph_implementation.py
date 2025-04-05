"""Regular graph version of langgraph."""

import logging
from typing import Literal, Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application
from host_app.models import FullGraphState, InputState


class InitializeOutput(BaseModel):
    previous_messages: list[BaseMessage] = []


@inject
async def initialize(
    state: InputState,
    store: BaseStore,
) -> InitializeOutput:
    question = state.question
    logging.debug(f"Processing question: {question}")

    previous_messages: Sequence[BaseMessage] = []
    logging.debug(f"Conversation ID: {state.conversation_id}")
    if state.conversation_id:
        found = await store.aget(namespace=("messages",), key=state.conversation_id)
        logging.debug(f"Found: {found}")
        if found:
            previous_messages = messages_from_dict(found.value["messages"])
    else:
        previous_messages = []
    return InitializeOutput(
        previous_messages=previous_messages,
    )


class CallLLMOutput(BaseModel):
    response_messages: list[BaseMessage] = []


@inject
async def call_llm(
    state: FullGraphState,
    store: BaseStore,
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
    chat_model: BaseChatModel = Provide[Application.main_model],
    system_prompt: str = Provide[Application.config.system_prompt],
) -> Command[Literal["tool_node", "__end__"]]:
    if not state.tools:
        async with mcp_client as client:
            tools = await client.get_tools()
    else:
        tools = state.tools

    model = chat_model.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(system_prompt),
        *state.previous_messages,
        HumanMessage(state.question),
        *state.response_messages,
    ]
    response: BaseMessage = await model.ainvoke(input=messages)
    assert isinstance(response, AIMessage)
    update = CallLLMOutput(response_messages=[response])

    if response.tool_calls:
        return Command(update=update, goto="tool_node")

    if state.conversation_id:
        logging.debug(f"Saving messages for conversation ID: {state.conversation_id}")
        await store.aput(
            namespace=("messages",),
            key=state.conversation_id,
            value={"messages": messages_to_dict(messages + [response])},
        )
    return Command(update=update, goto=END)


class ToolNodeInput(BaseModel):
    response_messages: list[BaseMessage]
    tools: list[BaseTool]


class ToolNodeOutput(BaseModel):
    response_messages: list[BaseMessage]


@inject
async def call_tool(
    state: ToolNodeInput,
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
) -> ToolNodeOutput:
    async with mcp_client as client:
        tools = await client.get_tools()
        logging.debug("Calling tools")
        messages_state = await ToolNode(tools=tools, name="tool_node").ainvoke(
            input={"messages": state.response_messages.copy()}
        )
    results = messages_state["messages"]
    return ToolNodeOutput(response_messages=results)


@inject
def make_graph(
    checkpointer: BaseCheckpointSaver | None = Provide[Application.checkpointer],
    store: BaseStore | None = Provide[Application.store],
    debug_mode: bool = Provide[Application.config.debug_mode],
) -> CompiledGraph:
    checkpointer = checkpointer or MemorySaver()
    store = store or InMemoryStore()

    graph = StateGraph(state_schema=FullGraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tool_node", call_tool)

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "call_llm")
    graph.add_edge("tool_node", "call_llm")
    # call_llm directs to tool_node or __end__

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=None,
        interrupt_after=None,
        debug=debug_mode,
    )
    return compiled_graph
