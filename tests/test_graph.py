import uuid
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import Item
from langgraph.store.memory import InMemoryStore

from host_app.containers import Application
from host_app.graph import InputState, OutputState, make_graph
from host_app.graph_runner import GraphRunner
from host_app.models import GraphUpdate, UpdateTypes


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


@pytest.fixture(autouse=True)
def container(fake_chat_model: FakeChatModel) -> Iterator[Application]:
    container = Application()
    container.config.from_yaml("config.yml")
    container.wire(
        modules=[
            "host_app.graph",
            "host_app.graph_runner",
        ]
    )
    with container.llms.main_model.override(fake_chat_model):
        yield container


@pytest.fixture
def basic_runnable_config() -> RunnableConfig:
    return {
        "configurable": {"thread_id": str(uuid.uuid4())},
    }


def test_compile_graph():
    graph = make_graph()
    assert isinstance(graph, CompiledGraph)


@pytest.fixture(scope="module")
def graph() -> CompiledGraph:
    return make_graph()


async def test_invoke_graph(graph: CompiledGraph, basic_runnable_config: RunnableConfig):
    result: dict[str, Any] = await graph.ainvoke(
        input=InputState(question="Hello"), config=basic_runnable_config
    )

    validated: OutputState = OutputState.model_validate(result)
    assert validated.response_messages[0].content == "Received: Hello"


def test_init_graph_runner():
    runner = GraphRunner()
    assert isinstance(runner, GraphRunner)
    connections = runner.mcp_client.connections
    assert isinstance(connections, dict)
    assert list(connections.keys()) == ["example_server"]


@pytest.fixture
def graph_runner() -> GraphRunner:
    return GraphRunner()


async def test_astream_graph_runner(graph_runner: GraphRunner):
    updates: list[GraphUpdate] = []
    async for update in graph_runner.astream_events(input=InputState(question="Hello")):
        assert isinstance(update, GraphUpdate)
        updates.append(update)

    assert len(updates) > 0
    assert updates[0].type_ == UpdateTypes.graph_start
    assert updates[-1].type_ == UpdateTypes.graph_end


async def test_memory_store_standalone():
    store = InMemoryStore()

    before = await store.aget(namespace=("testing",), key="test")
    assert before is None

    await store.aput(namespace=("testing",), key="test", value={"value": "value"})
    after = await store.aget(namespace=("testing",), key="test")
    assert isinstance(after, Item)
    assert after.value == {"value": "value"}


async def test_graph_binds_tools(
    graph_runner: GraphRunner, fake_chat_model: FakeChatModel, basic_runnable_config: RunnableConfig
):
    _ = await graph_runner.graph.ainvoke(
        input=InputState(question="Hello"), config=basic_runnable_config
    )

    assert len(fake_chat_model.tools_bound) == 1, "Should bind tools once"
    assert fake_chat_model.tools_bound[0][0].name == "test-tool", (
        "Should bind test-tool during tests"
    )
