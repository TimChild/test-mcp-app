import logging
import uuid
from typing import AsyncIterator

from .functional_langgraph import InputState
from .langgraph_adapters import GraphAdapter
from .models import QA, GraphUpdate, UpdateTypes


async def get_response_updates(
    question: str,
    message_history: list[QA],
    conversation_id: str | None = None,
    thread_id: str | None = None,
) -> AsyncIterator[GraphUpdate]:
    thread_id = thread_id or str(uuid.uuid4())
    yield GraphUpdate(type_=UpdateTypes.start, data=f"Question: {question}\n\n")
    yield GraphUpdate(
        type_=UpdateTypes.preprocess, data=f"Length History: {len(message_history)}\n\n"
    )
    logging.debug(f"calling with Conversation ID: {conversation_id}")
    async for update in GraphAdapter().astream_events(
        input=InputState(question=question, conversation_id=conversation_id),
        thread_id=thread_id,
    ):
        yield update

    yield GraphUpdate(type_=UpdateTypes.end, data="\n\n!!! End of response updates !!!\n\n")


# async def get_response_updates(
#     question: str, message_history: list[QA]
# ) -> AsyncIterator[GraphUpdate]:
#     """Get the response updates for a question.
#
#     Args:
#         question: The question to get the response updates for.
#         message_history: The message history.
#
#     Returns:
#         The response updates.
#     """
#     yield GraphUpdate(type_=UpdateTypes.start, data=f"Question: {question}\n\n")
#     yield GraphUpdate(
#         type_=UpdateTypes.preprocess, data=f"Length History: {len(message_history)}\n\n"
#     )
#
#     async for event in process.astream_events(
#         input=InputState(question=question),
#         config={"configurable": {"thread_id": str(uuid.uuid4())}},
#     ):
#         event_type = event["event"]
#         event_data: EventData = event["data"]
#         match event_type:
#             case "on_chat_model_stream":
#                 chunk = event_data.get("chunk", None)
#                 if chunk:
#                     assert isinstance(chunk, AIMessageChunk)
#                     content = chunk.content
#                     assert isinstance(content, str)
#                     yield GraphUpdate(type_=UpdateTypes.ai_delta, delta=content)
#             case "on_chat_model_end":
#                 chunk = event_data.get("output", None)
#                 assert isinstance(chunk, AIMessage)
#                 yield GraphUpdate(type_=UpdateTypes.ai_message_end)
#             case "on_tool_start":
#                 chunk = event_data.get("input", None)
#                 assert isinstance(chunk, dict)
#                 yield GraphUpdate(type_=UpdateTypes.tool_start, name=event["name"], data=chunk)
#             case "on_tool_end":
#                 chunk = event_data.get("output", None)
#                 assert isinstance(chunk, ToolMessage)
#                 yield GraphUpdate(
#                     type_=UpdateTypes.tool_end, name=event["name"], data=str(chunk.content)
#                 )
#             case _:
#                 print(f"Ignoring event: {event_type}")
#
#     yield GraphUpdate(type_=UpdateTypes.end, data="\n\n!!! End of response updates !!!\n\n")
