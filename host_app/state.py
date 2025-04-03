import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Mapping, Sequence

import reflex as rx
from dependency_injector.wiring import Provide, inject
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from mcp_client import MultiMCPClient
from reflex.event import EventType

from host_app.containers import Application

from .models import QA, GraphUpdate, McpServerInfo, ToolInfo, UpdateTypes
from .process import get_response_updates

load_dotenv()

# Checking if the API key is set properly
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("Please set OPENAI_API_KEY environment variable.")


DEFAULT_CHATS = {
    "Intros": [],
}


class State(rx.State):
    """The app state."""

    test_var: int = 0

    chats: dict[str, list[QA]] = DEFAULT_CHATS
    """A dict from the chat name to the list of questions and answers."""

    current_chat = "Intros"
    """The current chat name."""

    question: str
    """The current question."""

    processing: bool = False
    """Whether we are processing the question."""

    current_status: str = ""
    """The current status."""

    new_chat_name: str = ""
    """The name of the new chat."""

    modal_open: bool = False
    """The new chat modal open state."""

    mcp_servers: list[McpServerInfo] = []
    """The connected MCP servers."""

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

    @rx.event
    @inject
    async def on_load(self, mcp_client: MultiMCPClient = Provide[Application.mcp_client]) -> None:
        """Load the state."""
        server_tools: Mapping[str, Sequence[BaseTool]] = await mcp_client.get_tools_by_server()
        self.mcp_servers = []
        for server_name, tools in server_tools.items():
            tool_infos = [ToolInfo(name=tool.name, description=tool.description) for tool in tools]
            self.mcp_servers.append(McpServerInfo(name=server_name, tools=tool_infos))

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
    async def handle_send_click(self, form_data: dict[str, Any]) -> AsyncIterator[EventType | None]:
        # Get the question from the form
        question = form_data["question"]

        # Check if the question is empty
        if question == "":
            return

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
