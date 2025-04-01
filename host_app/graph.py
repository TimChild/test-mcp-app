"""Regular graph version of langgraph."""

import logging
from typing import Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application

from .models import InputState, OutputState


class FullState(BaseModel):
    question: str
    # response_messages: list[ToolMessage | AIMessage] = []
    response_messages: list[BaseMessage] = []
    conversation_id: str | None = None


SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


@inject
async def process(
    state: InputState,
    store: BaseStore,
    mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
    chat_model: BaseChatModel = Provide[Application.llms.main_model],
) -> OutputState:
    responses: list[AIMessage | ToolMessage] = []
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

    async with mcp_client as client:
        tools = await client.get_tools()
        model = chat_model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            *previous_messages,
            HumanMessage(question),
        ]
        response: BaseMessage = await model.ainvoke(input=messages)
        assert isinstance(response, AIMessage)
        responses.append(response)
        messages.append(response)
        logging.debug("Got initial response")

        assert isinstance(response, AIMessage)
        if response.tool_calls:
            logging.debug("Calling tools")
            messages_state = await ToolNode(tools=tools, name="tool_node").ainvoke(
                input={"messages": messages}
            )
            results = messages_state["messages"]
            responses.extend(results)
            messages.extend(results)
            logging.debug("Got tool responses")
            try:
                response = await model.ainvoke(input=messages)
            except Exception as e:
                logging.error(f"Error invoking model: {e}")
                logging.error(f"Messages: {messages}")
                raise e
            assert isinstance(response, AIMessage)
            responses.append(response)

        logging.debug("Returning responses")
        # return responses
    if state.conversation_id:
        logging.debug(f"Saving messages for conversation ID: {state.conversation_id}")
        await store.aput(
            namespace=("messages",),
            key=state.conversation_id,
            value={"messages": messages_to_dict(messages)},
        )
    return OutputState(response_messages=responses)
    # return Command(update=update, goto=["tool_caller_node", "sub_assistant_caller_node"])


@inject
def make_graph(
    checkpointer: BaseCheckpointSaver | None = Provide[Application.graph.checkpointer],
    store: BaseStore | None = Provide[Application.graph.store],
) -> CompiledGraph:
    checkpointer = checkpointer or MemorySaver()
    store = store or InMemoryStore()

    graph = StateGraph(state_schema=FullState)
    graph.add_node("process", process)
    # graph.add_node("tool_node", ToolNode)
    graph.set_entry_point("process")

    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=None,
        interrupt_after=None,
        debug=True,
    )
    return compiled_graph
