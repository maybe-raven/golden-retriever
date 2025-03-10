import asyncio
import os
from hashlib import sha1
from os import path
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple

import lancedb
from lancedb.embeddings import get_registry
from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from pandas import DataFrame
from pydantic import ValidationError

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    hash: str
    path: str
    offset: int
    text: str = embeddings.SourceField()
    vector: Vector(1024) = embeddings.VectorField(default=None)


class DBHandler:
    def __init__(self) -> None:
        self.connected = False
        self._lock = asyncio.Lock()

    async def connect(self):
        await self._lock.acquire()
        if not self.connected:
            self.db = await lancedb.connect_async("~/.golden-retriever/lancedb")
            self.table = await self.db.create_table(
                "documents", schema=Documents, exist_ok=True
            )
            self.connected = True
        self._lock.release()

    # Function to generate overlapping chunks from text
    def generate_chunks(
        self, text: str, chunk_size: int = 1000, overlap: int = 100
    ) -> List[Tuple[int, str]]:
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append((start, text[start:end]))
            if end == text_length:
                break
            start += chunk_size - overlap
        return chunks

    # Function to recursively traverse directories and process files
    def process_files(self, root_dir: str | Path) -> Generator[List[Documents]]:
        for file_path in get_all_files(root_dir):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                doc_hash = hash_file(content)
                chunks = self.generate_chunks(content)
                yield [
                    Documents(
                        hash=doc_hash,
                        path=file_path,
                        offset=offset,
                        text=text,
                    )
                    for (offset, text) in chunks
                ]
            except ValidationError as e:
                raise e
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Process files and insert into the table
    async def embed_recursive(self, root_dir: str | Path):
        await self._lock.acquire()
        data = self.process_files(root_dir)
        await self.table.add(data)
        await self.table.optimize()
        await self.table.create_index("text", config=FTS(with_position=False))
        self._lock.release()

    async def check_paths(self, paths: Iterable[str]) -> DataFrame:
        paths = f"({', '.join(f"'{p.replace("'", r"''")}'" for p in paths)})"
        print(paths)
        await self._lock.acquire()
        data = await (
            self.table.query()
            .where(f"path IN {paths}")
            .select(["path", "hash"])
            .limit(1_000_000_000)
            .to_pandas()
        )
        self._lock.release()
        assert isinstance(data, DataFrame)
        data.drop_duplicates(inplace=True)
        return data

    async def search(self, query: str) -> DataFrame:
        await self._lock.acquire()
        vector_query = embeddings.compute_query_embeddings(query)[0]
        data = await (
            self.table.query()
            .nearest_to(vector_query)
            .nearest_to_text(query, "text")
            .rerank()
            # .rerank(CrossEncoderReranker(trust_remote_code=False))
            # WTF!!! ^^^this single line breaks Textual's input
            .limit(10)
            .to_pandas()
        )
        self._lock.release()
        return data


class GenAiModel:
    def __init__(self):
        self.model = os.environ.get("GR_MODEL", "llama-3.2-1b-instruct")

        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "..."),
        )
        pass

    def generateResponse(self, text: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You're helping me get information, specifically from a text that I will give you",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            model=self.model,
            # stream=Fal,
        )
        return response.choices[0].message.content


def hash_file(content: str) -> str:
    hasher = sha1(content.encode(), usedforsecurity=False)
    return hasher.hexdigest()


def get_all_files(root_dir: str | Path) -> Generator[str]:
    for dirpath, _, files in os.walk(root_dir):
        for filename in files:
            if not (filename.endswith(".md") or filename.endswith(".txt")):
                continue
            yield path.abspath(path.join(dirpath, filename))
