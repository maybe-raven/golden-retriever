import os
import sys
from argparse import ArgumentParser
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Tuple

from pandas import DataFrame, Series
from rich.text import Text
from textual import log, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content, Span
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    TabbedContent,
    Tabs,
    Tree,
)
from textual.widgets.tree import TreeNode

from lib import DBHandler, get_all_files, hash_file


class DocumentsListView(ListView):
    paths: reactive[List[Path]] = reactive([])

    BINDINGS = [
        ("k", "cursor_up", "Cursor up"),
        ("j", "cursor_down", "Cursor down"),
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
        self.extend([ListItem(Label(os.path.basename(p))) for p in new_paths])
        self.send_msg()

    def watch_index(self, old_index: int | None, new_index: int | None):
        super().watch_index(old_index, new_index)
        self.send_msg()


# class DocumentView(Markdown):
#     path: reactive[Path | None] = reactive(None)
#
#     async def watch_path(self, new_path: Path | None):
#         if new_path is None:
#             self.update("")
#         else:
#             await self.load(new_path)


class ChunksView(RichLog):
    """A widget to display chunks in its original text."""

    chunks: reactive[DataFrame | None] = reactive(None, always_update=True)
    """The chunks to display. All chunks are assumed to refer to the same file."""

    BINDINGS = [
        Binding("right", "next_hunk", "Next Hunk", show=False),
        Binding("left", "prev_hunk", "Previous Hunk", show=False),
        ("l", "next_hunk", "Next Hunk"),
        ("h", "prev_hunk", "Previous Hunk"),
        ("k", "scroll_up", "Scroll Up"),
        ("j", "scroll_down", "Scroll Down"),
        ("ctrl+b", "page_up", "Page Up"),
        ("ctrl+f", "page_down", "Page Down"),
    ]

    def __init__(
        self,
        *,
        max_lines: int | None = None,
        min_width: int = 78,
        highlight: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            max_lines=max_lines,
            min_width=min_width,
            wrap=True,
            highlight=highlight,
            markup=True,
            auto_scroll=False,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.hunk_line_numbers = []

    def watch_chunks(self, chunks: Optional[DataFrame]):
        self.clear()
        if chunks is None:
            return

        print(chunks[["offset", "text"]])
        path = chunks["path"].unique()
        assert len(path) == 1
        with open(path[0], "r", encoding="utf-8") as f:
            text = f.read().strip()

        doc_hash = hash_file(text)
        print(doc_hash)
        chunk_doc_hash = chunks["hash"].unique()
        assert len(chunk_doc_hash) == 1

        if chunk_doc_hash[0] == doc_hash:
            # same file
            spans = [Span(i, i + 1000, "green") for i in chunks["offset"]]
            content = Content(text, spans)
            self.write(content=content.markup)
            self.find_line_numbers()
            if self.hunk_line_numbers:
                self.scroll_to(y=self.hunk_line_numbers[0], animate=False)
        else:
            # content has changed since embedding
            self.hunk_line_numbers = []
            # need to subract 2 from the width because of the scrollbar
            divider = "=" * (self.size.width - 2)
            text = (
                "[bold]WARNING:[/bold] the file has changed since it was embedded. "
                "Displayed here are the embedding chunks that matched.\n" + divider
            )
            for chunk in chunks.sort_values(by="offset", ascending=True)["text"]:
                text += "[green]" + chunk.replace("[", r"\[") + "[/green]\n" + divider
            self.write(text)

    def action_next_hunk(self):
        if not self.hunk_line_numbers:
            return

        for line_number in self.hunk_line_numbers:
            if line_number > self.scroll_y:
                self.scroll_to(y=line_number, animate=False)
                return

    def action_prev_hunk(self):
        if not self.hunk_line_numbers:
            return

        for line_number in reversed(self.hunk_line_numbers):
            if line_number < self.scroll_y:
                self.scroll_to(y=line_number, animate=False)
                return

    def find_line_numbers(self):
        self.hunk_line_numbers = []
        flag = False
        for i, line in enumerate(self.lines):
            line_has_style = False
            for segment in line._segments:
                if segment.style:
                    line_has_style = True
                    break
            if line_has_style and not flag:
                flag = True
                self.hunk_line_numbers.append(0 if i <= 3 else i - 3)
            elif not line_has_style and flag:
                flag = False


class RetrievalView(Widget):
    data: reactive[Optional[DataFrame]] = reactive(None, always_update=True)

    def watch_data(self, data):
        self.query_one(DocumentsListView).paths = (
            [] if data is None else [Path(p) for p in sorted(data["path"].unique())]
        )

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield DocumentsListView(classes="column left")
            yield ChunksView(classes="column right")

    def on_documents_list_view_path_selected(
        self, event: DocumentsListView.PathSelected
    ):
        if event.path is None or self.data is None:
            chunks = None
        else:
            p = str(event.path)
            chunks = self.data[self.data["path"] == p]
            print(chunks)
        assert isinstance(chunks, Optional[DataFrame])
        self.query_one(ChunksView).chunks = chunks


class ChatView(Widget):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield ListView(classes="row top")
            yield Input(classes="row bottom")


class MainView(Widget):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield CustomDirectoryTree(classes="column left")
            yield ChatView(classes="column right")


class StagingData:
    files: List[str]
    """
    The absolute paths of files to display in the directory tree.
    They are all assumed to be descendants of the current working directory.
    """

    hashes: DataFrame
    """
    A dataframe that contains two columns: path and hash.
    The hash is used to check if the file has changed.
    """

    def __init__(self, files: List[str], hashes: DataFrame):
        self.files = files
        self.hashes = hashes


class NodeState(StrEnum):
    UNEMBEDDED = ""
    """File that have not been embedded."""
    CLEAN = "green"
    """File that have been embedded and have not changed since."""
    DIRTY = "red"
    """File that have been embedded but have changed since."""
    MULTI = "purple"
    """This file has been embedded multiple times."""


CDTDataType = Tuple[Path, NodeState]


class CustomDirectoryTree(Tree[CDTDataType]):
    selection: Set[Path] = set()

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("a", "toggle_all", "Select All"),
        Binding("space", "select_cursor", "Select"),
        Binding("enter", "toggle_node", "Toggle"),
        Binding("k", "cursor_up", "Cursor Up"),
        Binding("j", "cursor_down", "Cursor Down"),
    ]

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            ".", None, name=name, id=id, classes=classes, disabled=disabled
        )
        self.show_root = False
        self.auto_expand = False

    def build_tree(self, data: StagingData):
        self.clear()
        self.selection.clear()
        self.root.expand()

        files = data.files
        hashes = data.hashes.set_index("path")["hash"]
        assert isinstance(hashes, Series)

        root_path = Path.cwd()
        nodes: Dict[Path, TreeNode] = {root_path: self.root}

        for file in sorted(files):
            file_path = Path(file)
            relative_path = file_path.relative_to(root_path)
            parent = root_path

            for part in relative_path.parts:
                current = parent / part
                if current not in nodes:
                    nodes[current] = nodes[parent].add(
                        part, expand=not part.startswith(".")
                    )
                parent = current

            state = self.check_state(file, hashes)
            node = nodes[file_path]
            node.data = (file_path, state)  # Store the absolute path
            node.set_label(Text(file_path.name, style=str(state)))
            node.allow_expand = False

    def check_state(self, file_path: str, hashes: Series) -> NodeState:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        computed_hash = hash_file(content)
        stored_hashes = hashes[hashes.index == file_path]
        assert isinstance(stored_hashes, Series)

        log(
            {
                "path": file_path,
                "computed_hash": computed_hash,
                "nunique": stored_hashes.nunique(),
                "stored_hash": stored_hashes.iloc[0] if file_path in hashes else None,
            }
        )

        if file_path not in hashes:
            return NodeState.UNEMBEDDED
        elif stored_hashes.nunique() > 1:
            return NodeState.MULTI
        elif stored_hashes.iloc[0] == computed_hash:
            return NodeState.CLEAN
        else:
            return NodeState.DIRTY

    def action_toggle_all(self):
        self.toggle_selection(self.root)

    def toggle_selection(self, node: TreeNode[Tuple[Path, NodeState]]):
        if len(node.children) == 0:
            assert node.data is not None, "all leaf nodes should have a path"
            path, state = node.data
            if state == NodeState.CLEAN:
                # don't support re-embedding clean files atm.
                return

            if path in self.selection:
                self.selection.remove(path)
                if isinstance(node.label, str):
                    node.set_label(Text(node.label, style=str(state)))
                else:
                    node.label.style = str(state)
            else:
                self.selection.add(path)
                if isinstance(node.label, str):
                    node.set_label(Text(node.label, style=str(NodeState.CLEAN)))
                else:
                    node.label.style = str(NodeState.CLEAN)

            node.refresh()
        else:
            for child in node.children:
                self.toggle_selection(child)

    def action_select_cursor(self) -> None:
        super().action_select_cursor()
        if self.cursor_node is None:
            return
        self.toggle_selection(self.cursor_node)
        log(node=self.cursor_node, selection=self.selection)


class GRApp(App):
    CSS_PATH = "gui.tcss"
    BINDINGS = [("q", "quit", "Quit")]

    db = DBHandler()

    @work(exclusive=True, group="search")
    async def do_search(self, query: str, embed_root_dir: Optional[Path]):
        log("connecting to db for search")
        await self.db.connect()

        if embed_root_dir is not None:
            await self.db.embed_recursive(embed_root_dir)
        log("doing search..........")
        result = await self.db.search(query)
        log(result)
        self.query_one(RetrievalView).data = result

    @work(exclusive=True, group="paths")
    async def check_paths(self):
        await self.db.connect()
        paths = list(get_all_files("."))
        results = await self.db.check_paths(paths)
        for _, item in results.iterrows():
            log({"path": item.iloc[0], "hash": item.iloc[1]})
        self.query_one(CustomDirectoryTree).build_tree(StagingData(paths, results))

    async def on_mount(self):
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
        self.check_paths()
        self.do_search(args.query, args.embed_root_dir)

        tabs_widget = self.query_one(Tabs)
        tabs_widget._bindings.bind("h", "previous_tab", "Previous tab")
        tabs_widget._bindings.bind("l", "next_tab", "Next tab")

    def compose(self) -> ComposeResult:
        yield Footer()
        with TabbedContent("Main", "Retrieval"):
            yield MainView()
            yield RetrievalView()


def main():
    app = GRApp()
    app.run()


if __name__ == "__main__":
    main()
