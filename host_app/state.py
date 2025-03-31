import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator

import reflex as rx
from dotenv import load_dotenv
from openai import OpenAI
from reflex.event import EventType

from .models import QA, GraphUpdate, UpdateTypes
from .process import get_response_updates

load_dotenv()

SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""

# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    test_var: int = 0

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = DEFAULT_CHATS

    # The current chat name.
    current_chat = "Intros"

    # The current question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The current status.
    current_status: str = ""

    # The name of the new chat.
    new_chat_name: str = ""

    # New chat modal open state.
    modal_open: bool = False

    @rx.event
    def set_new_chat_name(self, name: str) -> None:
        """Set the name of the new chat.

        Args:
            form_data: A dict with the new chat name.
        """
        self.new_chat_name = name

    @rx.event
    def toggle_modal(self) -> None:
        """Toggle the modal."""
        self.modal_open = not self.modal_open

    @rx.event
    def create_chat(self) -> None:
        """Create a new chat."""
        # Add the new chat to the list of chats.
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    @rx.event
    def delete_chat(self) -> None:
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    @rx.event
    def set_chat(self, chat_name: str) -> None:
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name

    @rx.var(cache=True)
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    @rx.event
    async def test_handler(self) -> AsyncIterator[EventType]:
        t = time.time()
        while time.time() - t < 5:
            self.test_var += 1
            yield rx.toast.info(f"Test var: {self.test_var}")
            await asyncio.sleep(0.5)

    @rx.event
    async def process_question(self, form_data: dict[str, Any]) -> AsyncIterator[EventType | None]:
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

        # # The reflex default
        # question_processor = self.openai_process_question

        # My implementation
        question_processor = self.general_process_question

        async for event in question_processor(question):
            yield event

    async def general_process_question(self, question: str) -> AsyncIterator[EventType | None]:
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        self.processing = True
        yield

        async for update in get_response_updates(
            question=question,
            message_history=self.chats[self.current_chat][:-1],
            conversation_id=self.current_chat,
        ):
            update: GraphUpdate
            match update.type_:
                case UpdateTypes.start:
                    logging.debug("Start update")
                    self.current_status = f"Starting: {update.data}"
                case UpdateTypes.ai_delta:
                    logging.debug("AI delta update")
                    self.chats[self.current_chat][-1].answer += update.delta
                    self.chats = self.chats
                case UpdateTypes.ai_message_end:
                    logging.debug("AI message end update")
                    pass
                case UpdateTypes.tool_start:
                    logging.debug("Tool start update")
                    self.chats[self.current_chat][
                        -1
                    ].answer += f"\n\n---\n\nStarting tool: {update.name} -- ({update.data})\n\n..."
                    self.chats = self.chats
                    self.current_status = f"Starting tool: {update.name} -- ({update.data})"
                case UpdateTypes.tool_end:
                    logging.debug("Tool end update")
                    self.chats[self.current_chat][
                        -1
                    ].answer += f"\n\nEnding tool: {update.name} -- ({update.data})\n\n---\n\n"
                    self.current_status = f"Ending tool: {update.name} -- ({update.data})"
                case UpdateTypes.end:
                    logging.debug("End update")
                    self.current_status = f"Ending: {update.data}"
                case _:
                    logging.debug(f"Unknown update type: {update.type_}")
                    self.current_status = f"Unknown update type: {update.type_}"
            yield

        self.current_status = ""
        self.processing = False

    async def openai_process_question(self, question: str) -> AsyncIterator[None]:
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """
        # NOTE: This is the Reflex default implementation

        # Add the question to the list of questions.
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        # Clear the input and start the processing.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Start a new session to answer the question.
        session = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,  # type: ignore
            stream=True,
        )  # type: ignore

        # Stream the results, yielding after every word.
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer_text = item.choices[0].delta.content
                # Ensure answer_text is not None before concatenation
                if answer_text is not None:
                    self.chats[self.current_chat][-1].answer += answer_text
                else:
                    # Handle the case where answer_text is None, perhaps log it or assign a default value
                    # For example, assigning an empty string if answer_text is None
                    answer_text = ""
                    self.chats[self.current_chat][-1].answer += answer_text
                self.chats = self.chats
                yield

        # Toggle the processing flag.
        self.processing = False
