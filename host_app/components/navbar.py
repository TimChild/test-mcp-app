import reflex as rx

from host_app.models import McpServerInfo
from host_app.state import State


def sidebar_chat(chat: str) -> rx.Component:
    """A sidebar chat item.

    Args:
        chat: The chat item.
    """
    return rx.drawer.close(
        rx.hstack(
            rx.button(
                chat,
                on_click=lambda: State.set_chat(chat),
                width="80%",
                variant="surface",
            ),
            rx.button(
                rx.icon(
                    tag="trash",
                    on_click=State.delete_chat,
                    stroke_width=1,
                ),
                width="20%",
                variant="surface",
                color_scheme="red",
            ),
            width="100%",
        )
    )


def sidebar(trigger: rx.Component) -> rx.Component:
    """The sidebar component."""
    return rx.drawer.root(
        rx.drawer.trigger(trigger),
        rx.drawer.overlay(),
        rx.drawer.portal(
            rx.drawer.content(
                rx.vstack(
                    rx.heading("Chats", color=rx.color("mauve", 11)),
                    rx.divider(),
                    rx.foreach(State.chat_titles, lambda chat: sidebar_chat(chat)),
                    align_items="stretch",
                    width="100%",
                ),
                top="auto",
                right="auto",
                height="100%",
                width="20em",
                padding="2em",
                background_color=rx.color("mauve", 2),
                outline="none",
            )
        ),
        direction="left",
    )


def modal(trigger: rx.Component) -> rx.Component:
    """A modal to create a new chat."""
    return rx.dialog.root(
        rx.dialog.trigger(trigger),
        rx.dialog.content(
            rx.hstack(
                rx.input(
                    placeholder="Type something...",
                    on_blur=State.set_new_chat_name,
                    width=["15em", "20em", "30em", "30em", "30em", "30em"],
                ),
                rx.dialog.close(
                    rx.button(
                        "Create chat",
                        on_click=State.create_chat,
                    ),
                ),
                # background_color=rx.color("mauve", 1),
                spacing="2",
                width="100%",
            ),
        ),
    )


def connected_mcp_server_infos() -> rx.Component:
    def render_mcp_server_info(server_info: McpServerInfo) -> rx.Component:
        return rx.card(
            rx.inset(rx.heading(server_info.name), padding="1em", side="top"),
            rx.data_list.root(
                rx.data_list.item(
                    rx.data_list.label("Tools"),
                    rx.data_list.value(
                        rx.flex(
                            rx.foreach(server_info.tools, lambda tool: rx.badge(tool.name)),
                            wrap="wrap",
                            spacing="1",
                        )
                    ),
                )
            ),
        )

    return rx.dialog.root(
        rx.dialog.trigger(rx.button("MCP Servers")),
        rx.dialog.content(
            rx.vstack(
                rx.foreach(State.mcp_servers, render_mcp_server_info),
            )
        ),
    )


def navbar() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.avatar(fallback="RC", variant="solid"),
                rx.heading("Reflex Chat"),
                rx.desktop_only(
                    rx.badge(
                        State.current_chat,
                        rx.tooltip(
                            rx.icon("info", size=14),
                            content="The current selected chat.",
                        ),
                        variant="soft",
                    )
                ),
                rx.text(f"test: {State.test_var}"),
                rx.button("test-trigger", on_click=State.test_handler),
                align_items="center",
            ),
            rx.hstack(
                connected_mcp_server_infos(),
                modal(rx.button("+ New chat")),
                sidebar(
                    rx.button(
                        rx.icon(
                            tag="messages-square",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                rx.desktop_only(
                    rx.button(
                        rx.icon(
                            tag="sliders-horizontal",
                            color=rx.color("mauve", 12),
                        ),
                        background_color=rx.color("mauve", 6),
                    )
                ),
                align_items="center",
            ),
            justify_content="space-between",
            align_items="center",
        ),
        backdrop_filter="auto",
        backdrop_blur="lg",
        padding="12px",
        border_bottom=f"1px solid {rx.color('mauve', 3)}",
        background_color=rx.color("mauve", 2),
        position="sticky",
        top="0",
        z_index="100",
        align_items="center",
    )
