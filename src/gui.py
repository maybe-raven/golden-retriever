"""A simple application using examples/boilerplate.py as a basis."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from lib import DBHandler, GenAiModel
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
    genai = GenAiModel()
    
    if args.embed_root_dir is not None:
        db.embed_recursive(args.embed_root_dir)
    results = db.search(args.query)
    with open(str(results["path"][0]), "r", encoding="utf-8") as f:
        selectedText = f.read()

    print(type(results))
    print(results.get(["path", "text"]))
    
    paththing = results.get(["path", "text"]).values.tolist()
    
    searchListTable = sg.Table(paththing,
                headings=["path", "text"],
                enable_events=True,
                bind_return_key=True,
                select_mode=sg.TABLE_SELECT_MODE_BROWSE,
                key='list row select')
    
    selectedListText = sg.Multiline(default_text=selectedText, size=(None, 50))
    
    useFileBtn = sg.Button(button_text="use this file", key="use file")
    
    searchLayout = [  [searchListTable,
                selectedListText, useFileBtn]
        ]
    
    chatMultiline = sg.Multiline(size=(1, 50))
    chatInput = sg.Input()
    chatInputBtn = sg.Button('submit input', key='-user input')
    chatSelectFileBtn = sg.Button("select file", key='-open search')
    
    chatLayout = [  [chatMultiline], [chatInput], [chatInputBtn], [chatSelectFileBtn]
        ]

    # Create the Window
    chatWindow = sg.Window('Window Title', chatLayout)
    # window = sg.Window('Window Title', searchLayout)

    # Event Loop to process "events" and get the "values" of the inputs
    window = chatWindow
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
            break
        
        if event == '-user input':
            userin = chatInput.get()
            aiout = genai.generateResponse(userin)
            chatMultiline.update(aiout)
            
            print(userin)
        
        if event == '-open search':
            searchWindow = sg.Window('Window Title', searchLayout)
            window = searchWindow
        
        if event == 'list row select':
            row = searchListTable.SelectedRows[0]
            print(row, str(results["path"][row]))
            with open(str(results["path"][row]), "r", encoding="utf-8") as f:
                selectedText = f.read()
            selectedChunk = results["text"][row]
            selectedListText.update(value=selectedText)
        
        if event == 'use file':
            print('use file')

            res = genai.generateResponse(selectedChunk)
            print(res)
            window = chatWindow

           
            

    chatWindow.close()


if __name__ == "__main__":
    main(sys.argv[1:])
