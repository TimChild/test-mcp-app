import asyncio
import uuid
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import BaseStore, Item
from mcp_client import MultiMCPClient

from host_app.containers import Application, config_option_to_connections
from host_app.graph import FullState, InputState, OutputState, make_graph
from host_app.graph_runner import GraphRunner
from host_app.models import GraphUpdate, UpdateTypes

EXAMPLE_SERVER_CONFIG = {
    "command": "uv",
    "args": ["run", "tests/example_server.py"],
}
MISSING_STDIO_SERVER_CONFIG = {
    "command": "uv",
    "args": ["run", "non-existent-server.py"],
}
MISSING_SSE_SERVER_CONFIG = {
    "url": "https://missing-server.com",
}


class FakeChatModel(FakeMessagesListChatModel):
    # NOTE: list-list in-case it's called multiple times
    tools_bound: list[list[BaseTool]] = []

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: Optional[Union[str, Literal["any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        _, _ = tool_choice, kwargs
        base_tools: list[BaseTool] = []
        for tool in tools:
            assert isinstance(tool, BaseTool)
            base_tools.append(tool)
        self.tools_bound.append(base_tools)
        return self


@pytest.fixture
def fake_chat_model() -> FakeChatModel:
    return FakeChatModel(
        responses=[
            AIMessage("First response"),
            AIMessage("Second response"),
            AIMessage("Third response"),
        ]
    )


class NotSetModel:
    pass


@pytest.fixture(autouse=True, scope="session")
def container() -> Iterator[Application]:
    container = Application()
    container.config.from_yaml("config.yml")
    with container.config.adapters.mcp_servers.override({"example_server": EXAMPLE_SERVER_CONFIG}):
        with container.llms.main_model.override(NotSetModel):
            container.wire(
                modules=[
                    "host_app.graph",
                    "host_app.graph_runner",
                ]
            )
            yield container
            container.unwire()


def test_container(container: Application):
    """Check the container is set up correctly."""
    conf = container.config.adapters.mcp_servers()["example_server"]
    assert isinstance(conf, dict)
    assert conf["command"] == "uv"
    assert conf["args"] == ["run", "tests/example_server.py"]
    connections = container.adapters.mcp_client().connections
    assert "example_server" in connections
    # NOTE: connections is a dict[name, SSEConnection | StdioConnection]
    assert "tests/example_server.py" in str(connections["example_server"]), (
        "Should include the path somewhere"
    )


@pytest.fixture
def mock_chat_model(
    container: Application, fake_chat_model: FakeChatModel
) -> Iterator[FakeChatModel]:
    with container.llms.main_model.override(fake_chat_model):
        yield fake_chat_model


@pytest.fixture
def basic_runnable_config() -> RunnableConfig:
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
    }


def test_compile_graph():
    graph = make_graph()
    assert isinstance(graph, CompiledGraph)


@pytest.fixture()
def graph(mock_chat_model: FakeChatModel) -> CompiledGraph:
    _ = mock_chat_model
    return make_graph()


async def test_invoke_graph(graph: CompiledGraph, basic_runnable_config: RunnableConfig):
    result: dict[str, Any] = await graph.ainvoke(
        input=InputState(question="Hello"), config=basic_runnable_config
    )

    validated: OutputState = OutputState.model_validate(result)
    assert validated.response_messages[0].content == "First response"


def test_init_graph_runner():
    runner = GraphRunner()
    assert isinstance(runner, GraphRunner)
    connections = runner.mcp_client.connections
    assert isinstance(connections, dict)
    assert list(connections.keys()) == ["example_server"]


@pytest.fixture
def graph_runner() -> GraphRunner:
    return GraphRunner()


@pytest.mark.usefixtures("mock_chat_model")
async def test_astream_graph_runner(graph_runner: GraphRunner):
    updates: list[GraphUpdate] = []
    async for update in graph_runner.astream_events(input=InputState(question="Hello")):
        assert isinstance(update, GraphUpdate)
        updates.append(update)

    assert len(updates) > 0
    assert updates[0].type_ == UpdateTypes.graph_start
    assert updates[-1].type_ == UpdateTypes.graph_end


async def test_memory_store_standalone(container: Application):
    store = container.graph.store()
    before = await store.aget(namespace=("testing",), key="test")
    assert ("testing",) in store.list_namespaces(), "Should have been created on attempted access"
    assert before is None

    # Test retrieving with same `store`
    await store.aput(namespace=("testing",), key="test", value={"value": "value"})
    after = await store.aget(namespace=("testing",), key="test")
    assert isinstance(after, Item)
    assert after.value == {"value": "value"}

    # Check that `store` is a singleton
    store2 = container.graph.store()
    after2 = await store2.aget(namespace=("testing",), key="test")
    assert isinstance(after2, Item)
    assert after2.value == {"value": "value"}


async def test_graph_binds_tools(graph_runner: GraphRunner, mock_chat_model: FakeChatModel):
    _ = await graph_runner.ainvoke(input=InputState(question="Hello"))

    assert len(mock_chat_model.tools_bound) == 1, "Should bind tools once"
    assert mock_chat_model.tools_bound[0][0].name == "test-tool", (
        "Should bind test-tool during tests"
    )


@pytest.mark.usefixtures("mock_chat_model")
async def test_graph_has_memory(graph_runner: GraphRunner, container: Application):
    _ = await graph_runner.ainvoke(
        input=InputState(question="Hello", conversation_id="test-conv-id"),
    )

    store: BaseStore = container.graph.store()
    assert ("messages",) in store.list_namespaces()
    value = store.get(namespace=("messages",), key="test-conv-id")
    assert value is not None


async def test_mcp_client_with_missing_server(container: Application):
    with container.config.adapters.mcp_servers.override(
        {
            "example_server": EXAMPLE_SERVER_CONFIG,
            "missing_server": MISSING_STDIO_SERVER_CONFIG,
        }
    ):
        # mcp_client = container.adapters.mcp_client()
        conns = config_option_to_connections(container.config.adapters.mcp_servers())
        mcp_client = MultiMCPClient(connections=conns)
        mcp_client.set_connection_timeout(0.5)

        async def func() -> None:
            async with mcp_client:
                pass

        try:
            await asyncio.wait_for(func(), timeout=1)
        except TimeoutError:
            pytest.fail("Should not hang on missing server")


@pytest.mark.usefixtures("mock_chat_model")
async def test_graph_runs_with_missing_mcp_server(
    graph_runner: GraphRunner, container: Application
):
    """Should still be able to run the graph even if one of the servers is down."""
    with container.config.adapters.mcp_servers.override(
        {
            "example_server": EXAMPLE_SERVER_CONFIG,
            "missing_server": MISSING_SSE_SERVER_CONFIG,
        }
    ):
        mcp_client = container.adapters.mcp_client()
        mcp_client.set_connection_timeout(0.5)
        response: FullState = await graph_runner.ainvoke(input=InputState(question="Hello"))
        assert response.response_messages[0].content == "First response"


async def test_graph_with_single_tool_call(
    graph_runner: GraphRunner, mock_chat_model: FakeChatModel
):
    tool_call = ToolCall(id="test-call-id", name="test-tool", args={})
    mock_chat_model.responses = [
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage("Response after tool call"),
    ]
    response: FullState = await graph_runner.ainvoke(input=InputState(question="Hello"))

    first_message = response.response_messages[0]
    assert isinstance(first_message, AIMessage)
    assert first_message.content == ""
    assert len(first_message.tool_calls) == 1

    second_message = response.response_messages[1]
    assert isinstance(second_message, ToolMessage)

    third_message = response.response_messages[2]
    assert isinstance(third_message, AIMessage)
    assert third_message.content == "Response after tool call"


# async def test_graph_with_sequential_tool_calls(
#     graph_runner: GraphRunner, mock_chat_model: FakeChatModel
# ):
#     tool_call1 = ToolCall(id="test-call-id1", name="test-tool", args={})
#     tool_call2 = ToolCall(id="test-call-id2", name="test-tool", args={})
#     mock_chat_model.responses = [
#         AIMessage(content="", tool_calls=[tool_call1]),
#         AIMessage(content="", tool_calls=[tool_call2]),
#         AIMessage("Response after tool calls"),
#     ]
#     response: FullState = await graph_runner.ainvoke(input=InputState(question="Hello"))
#
#     first_message = response.response_messages[0]
#     assert isinstance(first_message, AIMessage)
#     assert first_message.content == ""
#     assert len(first_message.tool_calls) == 1
#
#     second_message = response.response_messages[1]
#     assert isinstance(second_message, ToolMessage)
#
#     third_message = response.response_messages[2]
#     assert isinstance(third_message, AIMessage)
#     assert third_message.content == ""
#     assert len(third_message.tool_calls) == 1
#
#     fourth_message = response.response_messages[3]
#     assert isinstance(fourth_message, ToolMessage)
#
#     fifth_message = response.response_messages[4]
#     assert isinstance(fifth_message, AIMessage)
#     assert fifth_message.content == "Response after tool calls"
