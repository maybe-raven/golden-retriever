import asyncio
import os
from hashlib import sha1
from os import path
from pathlib import Path
from typing import AsyncGenerator, Generator, Iterable, List, Tuple, Union

import lancedb
from lancedb.embeddings import get_registry
from lancedb.index import FTS, BTree
from lancedb.pydantic import LanceModel, Vector
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionUserMessageParam
from pandas import DataFrame
from pydantic import ValidationError
from textual import log

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()

RAG_PROMT = """## Task:
Respond to the user query using the provided context, quote texts from context wherever appropriate.

## Guidelines:
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Quote text from the context to support each of your point if possible.
- Respond in the same language as the user's query.
- If the context clearly answers the user's query, quote from the context directly.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
"""

DEFAULT_PROMT = """You're an AI. Make fun of the user or something. Go crazy."""


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

    async def create_indices(self):
        log("creating indices...")
        await self.table.create_index("text", config=FTS(with_position=False))
        await self.table.create_index("path", config=BTree())

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
    def process_files(
        self, files: Iterable[Union[Path, str]]
    ) -> Generator[Tuple[str, List[Documents]]]:
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read().strip()
                doc_hash = hash_file(content)
                chunks = self.generate_chunks(content)
                yield (
                    str(file_path),
                    [
                        Documents(
                            hash=doc_hash,
                            path=str(file_path),
                            offset=offset,
                            text=text,
                        )
                        for (offset, text) in chunks
                    ],
                )
            except ValidationError as e:
                raise e
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    async def embed_files(
        self, paths: Iterable[Union[Path, str]]
    ) -> AsyncGenerator[str]:
        log("embedding files: waiting for lock")
        await self._lock.acquire()
        log("manual upinserting...")
        for p, documents in self.process_files(paths):
            await self.table.delete(f"path = '{escape(p)}'")
            await self.table.add(documents)
            yield p
        log("optimizing...")
        await self.table.optimize()
        await self.create_indices()
        self._lock.release()
        log("embedding done...")

    async def check_paths(self, paths: Iterable[str]) -> DataFrame:
        paths = f"({', '.join(f"'{escape(p)}'" for p in paths)})"
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
        log("searching: waiting for lock")
        await self._lock.acquire()
        if not query or await self.table.count_rows() == 0:
            return DataFrame()
        log("searching: computing query embeddings...")
        vector_query = embeddings.compute_query_embeddings(query)[0]
        log("searching...")
        data = await (
            self.table.query()
            .nearest_to(vector_query)
            .nearest_to_text(query, "text")
            .rerank()
            # .rerank(CrossEncoderReranker(trust_remote_code=False))
            # WTF!!! ^^^this single line breaks Textual's input
            .select(["path", "hash", "offset", "text"])
            .limit(10)
            .to_pandas()
        )
        self._lock.release()
        log("searching: post-processing...")
        assert isinstance(data, DataFrame)
        data.drop_duplicates(["path", "hash", "offset", "text"], inplace=True)
        return data


class GenAiModel:
    def __init__(self):
        self.model = os.environ.get("GR_MODEL", "llama-3.2-1b-instruct")

        self.client = AsyncOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "..."),
        )

    async def generateResponse(
        self,
        user_msg: str,
        context: DataFrame,
        history: List[ChatCompletionUserMessageParam],
    ) -> AsyncStream[ChatCompletionChunk]:
        if not context.empty:
            context_texts = context.sort_values(by=["path", "offset"])["text"]
            user_msg = f"""## Contexts
            {"\n".join([f"### Citation {i}\n{c}\n" for i, c in enumerate(context_texts)])}

            ## User Query
            {user_msg}
            """

        return await self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": DEFAULT_PROMT if context.empty else RAG_PROMT,
                },
                *history,
                {
                    "role": "user",
                    "content": user_msg,
                },
            ],
            stream=True,
            model=self.model,
        )


def hash_file(content: str) -> str:
    hasher = sha1(content.encode(), usedforsecurity=False)
    return hasher.hexdigest()


def get_all_files(root_dir: str | Path) -> Generator[str]:
    for dirpath, _, files in os.walk(root_dir):
        for filename in files:
            if not (filename.endswith(".md") or filename.endswith(".txt")):
                continue
            yield path.abspath(path.join(dirpath, filename))


def escape(s: str) -> str:
    return s.replace("'", r"''")
