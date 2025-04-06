from typing import AsyncIterator

import pytest
from aiosqlite import Connection
from dependency_injector.wiring import Provide, inject
from langgraph.checkpoint.base import BaseCheckpointSaver

from host_app.containers import Application


@inject
async def get_conn(conn: Connection = Provide[Application.conn]) -> Connection:
    return conn


@inject
async def checkpoint_getter(
    checkpointer: BaseCheckpointSaver = Provide[Application.checkpointer],
) -> BaseCheckpointSaver:
    """Create a SqliteSaver instance from a connection string."""
    assert isinstance(checkpointer, BaseCheckpointSaver)
    return checkpointer


@pytest.fixture(scope="session")
async def application() -> AsyncIterator[Application]:
    """Fixture to provide a container instance."""
    container = Application()
    container.wire(modules=[__name__])
    await container.init_resources()  # pyright: ignore[reportGeneralTypeIssues]
    yield container
    await container.shutdown_resources()  # pyright: ignore[reportGeneralTypeIssues]
    container.unwire()


@pytest.mark.usefixtures("application")
async def test_get_conn():
    """Test that the connection is available in the container."""
    conn = await get_conn()
    assert conn is not None
    assert isinstance(conn, Connection)


@pytest.mark.usefixtures("application")
async def test_get_checkpointer():
    """Test that the checkpointer is available in the container."""
    checkpointer = await checkpoint_getter()
    assert checkpointer is not None
    assert isinstance(checkpointer, BaseCheckpointSaver)
