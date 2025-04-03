import logging.config
from typing import Any

from dependency_injector import containers, providers
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from mcp_client import MultiMCPClient
from mcp_client.multi_client import SSEConnection, StdioConnection


def config_option_to_connections(
    simple_config_dict: dict[str, dict[str, Any]],
) -> dict[str, StdioConnection | SSEConnection]:
    """Convert the mcp config dict from yaml to Connections.

    Basically just determine which type of connection, then check that the args are specified
    correctly for that connection type.

    Args:
        - The full mcp connections config as loaded from config.yaml

    Returns:
        - Sanitized connections dict utilizing the StdioConnection and SSEConnection TypedDict classes.
    """
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


class Application(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config.yml"], strict=True)

    logging = providers.Resource(
        logging.config.dictConfig,
        config=config.logging,
    )

    mcp_client = providers.Factory(
        MultiMCPClient,
        connections=providers.Singleton(
            config_option_to_connections,
            config.mcp_servers,
        ),
    )
    main_model = providers.Factory(
        ChatOpenAI,
        model="gpt-4o",
    )
    checkpointer = providers.Singleton(MemorySaver)
    store = providers.Singleton(InMemoryStore)

    wiring_config = containers.WiringConfiguration(
        modules=[
            ".host_app",
            ".state",
        ],
        packages=[".graph"],
    )
