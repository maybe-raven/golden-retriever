import os
from typing import List, Tuple

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from pydantic import ValidationError

# connect to LanceDB
db = lancedb.connect("~/.golden-retriever/lancedb")

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    text: str = embeddings.SourceField()
    vector: Vector(1024) = embeddings.VectorField(default=None)

class Dbhandler:
    def __init__(self):
        self.docs = Documents
        
        table_name = "myTable"
        table = db.create_table(table_name, schema=Documents, mode="overwrite")
        data = [
            {"text": "rebel spaceships striking from a hidden base"},
            {"text": "have won their first victory against the evil Galactic Empire"},
            {"text": "during the battle rebel spies managed to steal secret plans"},
            {"text": "to the Empire's ultimate weapon the Death Star"},
        ]
        table.add(data=data)
        table.create_index(metric="L2", vector_column_name="vector", index_type="IVF_FLAT")
        table.create_fts_index("text", replace=True)
        self.table = table

        return
    
    def generate_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[Tuple[int, str]]:
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
                                    hash=doc_hash, path=file_path, offset=offset, text=text
                                )
                                for (offset, text) in chunks
                            ]
                        )
                except ValidationError as e:
                    raise e
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        return documents

    def addData(self, dir:str):
        data = self.process_files(dir)
        if not data:
            print("no .txt or .md files found")
            return
        self.table.add(data=data)
        self.table.create_index(metric="L2", vector_column_name="vector", index_type="IVF_FLAT")
        self.table.create_fts_index("text", replace=True)
        
    def search(self, searchTerm:str):
        # you can use table.list_indices() to make sure indices have been created
        reranker = RRFReranker()
        results = (
            self.table.search(
                searchTerm,
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .rerank(reranker)
            .limit(10)
            .to_pandas()
        )
        print(results)
        return results

    
    



