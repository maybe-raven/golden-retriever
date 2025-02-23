import os
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker
from pydantic import ValidationError
from typing import List, Tuple
from pandas import DataFrame


# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    hash: int
    path: str
    offset: int
    text: str = embeddings.SourceField()
    vector: Vector(1024) = embeddings.VectorField(default=None)


class DBHandler:
    def __init__(self) -> None:
        self.db = lancedb.connect("~/.golden-retriever/lancedb")
        self.table = self.db.create_table("documents", schema=Documents, exist_ok=True)

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
    def process_files(self, root_dir: str):
        documents = []
        for dirpath, _, files in os.walk(root_dir):
            for filename in files:
                if not (filename.endswith(".md") or filename.endswith(".txt")):
                    continue
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        doc_hash = hash(content)
                        chunks = self.generate_chunks(content)
                        documents.extend(
                            [
                                Documents(
                                    hash=doc_hash,
                                    path=file_path,
                                    offset=offset,
                                    text=text,
                                )
                                for (offset, text) in chunks
                            ]
                        )
                except ValidationError as e:
                    raise e
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        return documents

    # Process files and insert into the table
    def embed_recursive(self, root_dir: str):
        data = self.process_files(root_dir)
        self.table.add(data=data)
        self.table.create_index(
            metric="L2", vector_column_name="vector", index_type="IVF_FLAT"
        )
        self.table.create_fts_index("text", replace=True)

    def search(self, query: str) -> DataFrame:
        reranker = CrossEncoderReranker()
        return (
            self.table.search(
                query,
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .rerank(reranker)
            .limit(10)
            .to_pandas()
        )

import os
from openai import OpenAI

class GenAiModel:
    
    def __init__(self):
        self.model = os.environ.get("GR_MODEL", "llama-3.2-1b-instruct")

        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "..."),
        )
        pass
        

    def generateResponse(self, text:str) -> str:
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
                
