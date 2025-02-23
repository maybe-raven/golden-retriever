"""A simple application using examples/boilerplate.py as a basis."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from lib import DBHandler
from threading import Thread

import pytermgui as ptg


PALETTE_LIGHT = "#FCBA03"
PALETTE_MID = "#8C6701"
PALETTE_DARK = "#4D4940"
PALETTE_DARKER = "#242321"


def _process_arguments(argv: list[str] | None = None) -> Namespace:
    """Processes command line arguments.

    Args:
        argv: A list of command line arguments, not including the binary path
            (sys.argv[0]).
    """

    parser = ArgumentParser(description="Golden Retriever")
    parser.add_argument(
        "query",
        type=str,
        help="the query to run on the knowledge database",
    )
    parser.add_argument(
        "-e",
        "--embed-root-dir",
        type=Path,
        help=(
            "the root directory to embed; all descendant "
            "files in this directory will be embeded"
        ),
    )

    return parser.parse_args(argv)


def _create_aliases() -> None:
    """Creates all the TIM aliases used by the application.

    Aliases should generally follow the following format:

        namespace.item

    For example, the title color of an app named "myapp" could be something like:

        myapp.title
    """

    ptg.tim.alias("app.text", "#cfc7b0")

    ptg.tim.alias("app.header", f"bold @{PALETTE_MID} #d9d2bd")
    ptg.tim.alias("app.header.fill", f"@{PALETTE_LIGHT}")

    ptg.tim.alias("app.title", f"bold {PALETTE_LIGHT}")
    ptg.tim.alias("app.button.label", f"bold @{PALETTE_DARK} app.text")
    ptg.tim.alias("app.button.highlight", "inverse app.button.label")

    ptg.tim.alias("app.footer", f"@{PALETTE_DARKER}")


def _configure_widgets() -> None:
    """Defines all the global widget configurations.

    Some example lines you could use here:

        ptg.boxes.DOUBLE.set_chars_of(ptg.Window)
        ptg.Splitter.set_char("separator", " ")
        ptg.Button.styles.label = "myapp.button.label"
        ptg.Container.styles.border__corner = "myapp.border"
    """

    ptg.boxes.DOUBLE.set_chars_of(ptg.Window)
    ptg.boxes.ROUNDED.set_chars_of(ptg.Container)

    ptg.Button.styles.label = "app.button.label"
    ptg.Button.styles.highlight = "app.button.highlight"

    ptg.Slider.styles.filled__cursor = PALETTE_MID
    ptg.Slider.styles.filled_selected = PALETTE_LIGHT

    ptg.Label.styles.value = "app.text"

    ptg.Window.styles.border__corner = "#C2B280"
    ptg.Container.styles.border__corner = PALETTE_DARK

    ptg.Splitter.set_char("separator", "")


def _define_layout() -> ptg.Layout:
    """Defines the application layout.

    Layouts work based on "slots" within them. Each slot can be given dimensions for
    both width and height. Integer values are interpreted to mean a static width, float
    values will be used to "scale" the relevant terminal dimension, and giving nothing
    will allow PTG to calculate the corrent dimension.
    """

    layout = ptg.Layout()

    # A header slot with a height of 1
    layout.add_slot("Header", height=1)
    layout.add_break()

    # A body slot that will fill the entire width, and the height is remaining
    layout.add_slot("Body left", width=0.5)

    # A slot in the same row as body, using the full non-occupied height and
    # 20% of the terminal's height.
    layout.add_slot("Body right", width=0.5)

    layout.add_break()

    # A footer with a static height of 1
    layout.add_slot("Footer", height=1)

    return layout


def list_files_and_chunks(root_dir):
    """Generates a directory tree with chunks"""
    tree = []
    for dirpath, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".md", ".txt")):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = [content[i : i + 100] for i in range(0, len(content), 100)]
                file_node = [f"[blue]{file}[/]"]
                for i, chunk in enumerate(chunks):
                    file_node.append(f"  └ Chunk {i + 1}: {chunk[:50]}...")
                tree.append(file_node)
    return tree


def _confirm_quit(manager: ptg.WindowManager) -> None:
    """Creates an "Are you sure you want to quit" modal window"""

    modal = ptg.Window(
        "[app.title]Are you sure you want to quit?",
        "",
        ptg.Container(
            ptg.Splitter(
                ptg.Button("Yes", lambda *_: manager.stop()),
                ptg.Button("No", lambda *_: modal.close()),
            ),
        ),
    ).center()

    modal.select(1)
    manager.add(modal)


def main(argv: list[str] | None = None) -> None:
    """Runs the application."""

    _create_aliases()
    _configure_widgets()

    args = _process_arguments(argv)
    db = DBHandler()
    if args.embed_root_dir is not None:
        db.embed_recursive(args.embed_root_dir)
    results = db.search(args.query)
    print(type(results))
    print(results["path"])
    print(results)

    with ptg.WindowManager() as manager:
        manager.layout = _define_layout()

        # Directory Tree and colors font for Left Panel

        header = ptg.Window(
            "[app.header] Golden Retriever ",
            box="EMPTY",
            is_persistant=True,
        )

        header.styles.fill = "app.header.fill"

        # Since header is the first defined slot, this will assign to the correct place
        manager.add(header)

        footer = ptg.Window(
            ptg.Button("Quit", lambda *_: _confirm_quit(manager)),
            box="EMPTY",
        )
        footer.styles.fill = "app.footer"

        # Since the second slot, body was not assigned to, we need to manually assign
        # to "footer"
        manager.add(footer, assign="footer")

        manager.add(
            ptg.Window("My sidebar"),
            assign="body_right",
        )

        right_label = ptg.Label(
            horizontal_align=ptg.HorizontalAlignment.LEFT,
            parent_align=ptg.HorizontalAlignment.LEFT,
        )
        right_pane = ptg.Window(
            right_label,
            vertical_align=ptg.VerticalAlignment.TOP,
            overflow=ptg.Overflow.SCROLL,
        )

        def update_right_pane(path):
            print("udpated")
            right_pane.set_title(os.path.basename(path))
            with open(str(results["path"][0]), "r", encoding="utf-8") as f:
                right_label.value = f.read()
                right_label.value = right_label.value.replace("[", "\\[")

        update_right_pane(str(results["path"][0]))

        wndw = ptg.Window(
            "[app.title]Files",
            "",
            vertical_align=ptg.VerticalAlignment.TOP,
            overflow=ptg.Overflow.SCROLL,
        )
        for p in sorted(set(results["path"])):
            wndw += ptg.Button(
                p,
                onclick=lambda *_: update_right_pane(p),
                parent_align=ptg.HorizontalAlignment.LEFT,
            )
        manager.add(
            wndw,
            assign="body_left",
        )

        manager.add(
            right_pane,
            assign="body_right",
        )

        manager.run()

    ptg.tim.print(f"[{PALETTE_LIGHT}]Goodbye!")


if __name__ == "__main__":
    main(sys.argv[1:])
