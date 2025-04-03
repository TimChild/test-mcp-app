import logging
import uuid
from typing import AsyncIterator

from .graph import GraphAdapter
from .models import QA, GraphUpdate, InputState, UpdateTypes


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
