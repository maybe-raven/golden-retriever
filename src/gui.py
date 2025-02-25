import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Type

from pandas import DataFrame
from textual._path import CSSPathType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.driver import Driver
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Label, ListItem, ListView, Markdown

from lib import DBHandler


class DocumentsListView(ListView):
    paths: reactive[List[Path]] = reactive([])

    BINDINGS = [
        Binding("enter", "select_cursor", "Select", show=False),
        Binding("up", "cursor_up", "Cursor up", show=False),
        Binding("k", "cursor_up", "Cursor up", show=False),
        Binding("down", "cursor_down", "Cursor down", show=False),
        Binding("j", "cursor_down", "Cursor down", show=False),
    ]

    class PathSelected(Message):
        def __init__(self, path: Path | None) -> None:
            super().__init__()
            self.path = path

    def send_msg(self):
        if len(self.paths) == 0 or self.index is None:
            self.post_message(self.PathSelected(None))
        else:
            self.post_message(self.PathSelected(self.paths[self.index]))

    def watch_paths(self, new_paths: List[Path]):
        self.clear()
        self.extend(
            [ListItem(Label(os.path.basename(p))) for p in sorted(set(new_paths))]
        )
        self.send_msg()

    def watch_index(self, old_index: int | None, new_index: int | None):
        super().watch_index(old_index, new_index)
        self.send_msg()


class DocumentView(Markdown):
    path: reactive[Path | None] = reactive(None)

    async def watch_path(self, new_path: Path | None):
        if new_path is None:
            self.update("")
        else:
            await self.load(new_path)


class GRApp(App):
    """A Textual app to manage stopwatches."""

    CSS_PATH = "gui.tcss"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        data: DataFrame,
        driver_class: Type[Driver] | None = None,
        css_path: CSSPathType | None = None,
        watch_css: bool = False,
        ansi_color: bool = False,
    ):
        self.data = data
        super().__init__(driver_class, css_path, watch_css, ansi_color)

    def on_mount(self):
        self.query_one(DocumentsListView).paths = [Path(p) for p in self.data["path"]]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Footer()
        with Horizontal():
            yield DocumentsListView(id="left", classes="column")
            yield DocumentView(classes="column", id="right")

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def on_documents_list_view_path_selected(
        self, event: DocumentsListView.PathSelected
    ):
        self.query_one(DocumentView).path = event.path


def main():
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
    args = parser.parse_args(sys.argv[1:])

    db = DBHandler()

    if args.embed_root_dir is not None:
        db.embed_recursive(args.embed_root_dir)
    results = db.search(args.query)
    print(results)

    app = GRApp(results)
    app.run()


if __name__ == "__main__":
    main()
