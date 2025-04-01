import logging.config
from typing import Any

from dependency_injector import containers, providers
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from mcp_client import MultiMCPClient
from mcp_client.multi_client import SSEConnection, StdioConnection


class Core(containers.DeclarativeContainer):
    config = providers.Configuration()

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )


def config_option_to_connections(
    simple_config_dict: dict[str, dict[str, Any]],
) -> dict[str, StdioConnection | SSEConnection]:
    """Add the necessary fields for a full MCP connection based on uri."""
    connections: dict[str, StdioConnection | SSEConnection] = {}
    assert isinstance(simple_config_dict, dict)
    for name, conf in simple_config_dict.items():
        if url := conf.get("url"):
            connections[name] = SSEConnection(transport="sse", url=url)
        elif command := conf.get("command"):
            args = conf.get("args", [])
            assert isinstance(args, list)
            assert all(isinstance(arg, str) for arg in args)
            connections[name] = StdioConnection(
                transport="stdio",
                command=command,
                args=args,
                env=None,
                encoding="utf-8",
                encoding_error_handler="strict",
            )
        else:
            raise ValueError(f"Invalid connection configuration: {conf}")
    return connections


class Adapters(containers.DeclarativeContainer):
    config = providers.Configuration()

    mcp_client: providers.Factory[MultiMCPClient] = providers.Factory(
        MultiMCPClient,
        connections=providers.Factory(
            config_option_to_connections,
            config.mcp_servers,
        ),
    )


class LLMs(containers.DeclarativeContainer):
    config = providers.Configuration()

    main_model = providers.Factory(
        ChatOpenAI,
        model="gpt-4o",
    )


class Graph(containers.DeclarativeContainer):
    config = providers.Configuration()

    checkpointer = providers.Singleton(MemorySaver)
    store = providers.Singleton(InMemoryStore)


class Application(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config.yml"], strict=True)

    core = providers.Container(
        Core,
        config=config.core,
    )

    adapters: providers.Container[Adapters] = providers.Container(
        Adapters,
        config=config.adapters,
    )

    llms = providers.Container(
        LLMs,
        config=config.llms,
    )

    graph = providers.Container(
        Graph,
        config=config.graph,
    )
