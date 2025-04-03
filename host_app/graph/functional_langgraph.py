"""Using the new Functional API for langgraph.

2025-03-30 -- Unfortunately, it has a bug that prevents it from working in `.astream_events` mode.
@task's return None instead of their values.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Sequence

from dependency_injector.wiring import Provide
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Application
from host_app.models import InputState

checkpointer = MemorySaver()

SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


@asynccontextmanager
async def connect_client(
    mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
) -> AsyncIterator[MultiMCPClient]:
    async with mcp_client:
        yield mcp_client


@task
async def call_tool(tool_call: ToolCall, tools: Sequence[BaseTool]) -> ToolMessage:
    logging.debug(f"Calling tool: {tool_call}")
    tool = next(tool for tool in tools if tool.name == tool_call["name"])
    tool_call_result = await tool.ainvoke(tool_call)
    assert isinstance(tool_call_result, ToolMessage)
    return tool_call_result


class OutputState(BaseModel):
    response_messages: Sequence[AnyMessage]


@entrypoint(checkpointer=checkpointer)
async def process(inputs: InputState) -> OutputState:
    responses: list[AIMessage | ToolMessage] = []
    question = inputs.question
    model = ChatOpenAI(model="gpt-4o")
    logging.debug(f"Processing question: {question}")

    async with connect_client() as client:
        tools = await client.get_tools()
        model = model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
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
            futures = [call_tool(tool_call, tools) for tool_call in response.tool_calls]
            results = await asyncio.gather(*futures)
            if any(not isinstance(result, ToolMessage) for result in results):
                logging.error(f"Got invalid tool response: {results}")
                results = [
                    r
                    if isinstance(r, ToolMessage)
                    else ToolMessage(
                        tool_call_id=call["id"], content="Error: Missing ToolMessage from tool"
                    )
                    for r, call in zip(results, response.tool_calls, strict=True)
                ]
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
        return OutputState(response_messages=responses)
