"""A simple application using examples/boilerplate.py as a basis."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from lib import DBHandler
from threading import Thread

import tkinter
import PySimpleGUI as sg

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



def main(argv: list[str] | None = None) -> None:
    """Runs the application."""

    args = _process_arguments(argv)
    db = DBHandler()
    if args.embed_root_dir is not None:
        db.embed_recursive(args.embed_root_dir)
    results = db.search(args.query)
    with open(str(results["path"][0]), "r", encoding="utf-8") as f:
        text = f.read()
    print(type(results))
    print(results.get(["path", "text"]))
    paththing = results.get(["path", "text"]).values.tolist()
    
    searchListTable = sg.Table(paththing,
                headings=["path", "text"],
                enable_events=True,
                bind_return_key=True,
                select_mode=sg.TABLE_SELECT_MODE_BROWSE,
                key='list row select')
    
    selectedListText = sg.Multiline(default_text=text, size=(None, 20))
    
    # searchListTable.get_last_clicked_position()
    
    layout = [  [searchListTable,
                selectedListText]
        ]

    # Create the Window
    window = sg.Window('Window Title', layout)

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
            
        if event == 'list row select':
            row = searchListTable.SelectedRows[0]
            with open(str(results["path"][row]), "r", encoding="utf-8") as f:
                text = f.read()
            selectedListText.update(value=text)

    window.close()


if __name__ == "__main__":
    main(sys.argv[1:])
