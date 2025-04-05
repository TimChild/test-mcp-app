"""Using the new Functional API for langgraph."""

import asyncio
import logging
from typing import Sequence

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    messages_from_dict,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.func import entrypoint, task
from langgraph.pregel import Pregel
from langgraph.store.base import BaseStore
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application
from host_app.models import InputState


class InitializeOutput(BaseModel):
    previous_messages: list[BaseMessage] = []


@task
async def load_previous_messages(
    conversation_id: str | None,
    store: BaseStore,
) -> list[BaseMessage]:
    previous_messages: list[BaseMessage] = []
    if conversation_id:
        loaded = await store.aget(namespace=("messages",), key=conversation_id)
        if loaded:
            previous_messages = messages_from_dict(loaded.value["messages"])
    return previous_messages


class CallToolsInput(BaseModel):
    response_messages: list[BaseMessage]
    tools: list[BaseTool]


class CallToolsOutput(BaseModel):
    response_messages: list[BaseMessage]


@task
async def call_tools(tool_calls: list[ToolCall], tools: Sequence[BaseTool]) -> list[ToolMessage]:
    def missing_message(tool_call: ToolCall) -> ToolMessage:
        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Error: Missing ToolMessage from tool {tool_call['name']}",
        )

    logging.debug(f"Calling tools: {[tc['name'] for tc in tool_calls]}")

    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    response_tasks = []
    async with asyncio.TaskGroup() as tg:
        for tool_call in tool_calls:
            if tool_call["name"] not in tools_by_name:
                response_tasks.append(missing_message(tool_call))
            tool = tools_by_name[tool_call["name"]]
            response_tasks.append(tg.create_task(tool.ainvoke(tool_call)))

    tool_responses = await asyncio.gather(*response_tasks)
    assert all(isinstance(tool_response, ToolMessage) for tool_response in tool_responses)
    return tool_responses


class OutputState(BaseModel):
    response_messages: Sequence[AnyMessage]


@inject
def make_graph(
    checkpointer: BaseCheckpointSaver = Provide[Application.checkpointer],
    store: BaseStore = Provide[Application.store],
    system_prompt: str = Provide[Application.config.system_prompt],
    mcp_client: MultiMCPClient = Provide[Application.mcp_client],
    chat_model: BaseChatModel = Provide[Application.main_model],
    max_iterations: int = 10,
) -> Pregel:
    """Create a graph with the given checkpointer and store."""

    @entrypoint(checkpointer=checkpointer, store=store)
    async def graph(
        inputs: InputState,
        store: BaseStore,
    ) -> OutputState:
        responses: list[AIMessage | ToolMessage] = []
        question = inputs.question
        logging.debug(f"Processing question: {question}")

        async with mcp_client as client:
            tools = await client.get_tools()

            model = chat_model.bind_tools(tools)

            previous_messages = await load_previous_messages(
                conversation_id=inputs.conversation_id, store=store
            )

            message_history: list[BaseMessage] = [
                SystemMessage(system_prompt),
                *previous_messages,
                HumanMessage(question),
            ]

            # Loop calling ai -> tools -> ai ... until no more tool calls or max iterations
            for i in range(max_iterations):
                logging.debug(f"Iteration {i}")

                ai_message: BaseMessage = await model.ainvoke(input=message_history)
                assert isinstance(ai_message, AIMessage)
                message_history.append(ai_message)
                responses.append(ai_message)

                if not ai_message.tool_calls:
                    break

                tool_responses: list[ToolMessage] = await call_tools(
                    ai_message.tool_calls, tools=tools
                )
                message_history.extend(tool_responses)
                responses.extend(tool_responses)

        return OutputState(response_messages=responses)

    return graph
