"""The main Chat app."""

import logging

import reflex as rx
import reflex_chakra as rc
from reflex.config import environment

from host_app.components import chat, navbar
from host_app.state import State

from .containers import Application

logging.basicConfig(level=logging.DEBUG)


def index() -> rx.Component:
    """The main app."""
    return rc.vstack(
        navbar(),
        chat.chat(),
        chat.action_bar(),
        background_color=rx.color("mauve", 1),
        color=rx.color("mauve", 12),
        min_height="100vh",
        align_items="stretch",
        spacing="0",
    )


def make_app() -> rx.App:
    container = Application()
    container.config.core.init_resources()
    container.config.adapters.init_resources()
    container.wire(
        modules=[
            ".graph",
            ".state",
        ]
    )

    # Add state and page to the app.
    app = rx.App(
        theme=rx.theme(
            appearance="dark",
            accent_color="indigo",
        ),
    )
    app.add_page(index, on_load=[State.on_load])
    return app


if env := environment.REFLEX_ENV_MODE.get():
    compile_context = environment.REFLEX_COMPILE_CONTEXT.get()
    print(f"Running in {env} mode. Compile context: {compile_context}")
    app = make_app()
