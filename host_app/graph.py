"""Regular graph version of langgraph."""

import logging
from typing import Annotated, Literal, Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
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
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application

from .models import InputState, OutputState


class FullState(BaseModel):
    question: str
    previous_messages: list[BaseMessage] = []
    response_messages: Annotated[list[AnyMessage], add_messages]
    tools: list[BaseTool] = []
    conversation_id: str | None = None


SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


class ToolNodeInput(BaseModel):
    response_messages: list[BaseMessage] = []


class ToolNodeOutput(BaseModel):
    response_messages: list[BaseMessage] = []


@inject
async def call_tool(
    state: ToolNodeInput,
    mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
) -> ToolNodeOutput:
    async with mcp_client as client:
        tools = await client.get_tools()
        logging.debug("Calling tools")
        messages_state = await ToolNode(tools=tools, name="tool_node").ainvoke(
            input={"messages": state.response_messages}
        )
    results = messages_state["messages"]
    logging.debug("Got tool responses")
    return ToolNodeOutput(response_messages=results)


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
    tools: list[BaseTool] = []


@inject
async def call_llm(
    state: FullState,
    store: BaseStore,
    mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
    chat_model: BaseChatModel = Provide[Application.llms.main_model],
) -> Command[Literal["tool_node", "__end__"]]:
    if not state.tools:
        async with mcp_client as client:
            tools = await client.get_tools()
    else:
        tools = state.tools

    model = chat_model.bind_tools(tools)
    messages: list[BaseMessage] = [
        SystemMessage(SYSTEM_PROMPT),
        *state.previous_messages,
        HumanMessage(state.question),
    ]
    response: BaseMessage = await model.ainvoke(input=messages)
    assert isinstance(response, AIMessage)
    update = OutputState(response_messages=[response])

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


@inject
def make_graph(
    checkpointer: BaseCheckpointSaver | None = Provide[Application.graph.checkpointer],
    store: BaseStore | None = Provide[Application.graph.store],
    debug_mode: bool = Provide[Application.config.debug_mode],
) -> CompiledGraph:
    checkpointer = checkpointer or MemorySaver()
    store = store or InMemoryStore()

    graph = StateGraph(state_schema=FullState)
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
