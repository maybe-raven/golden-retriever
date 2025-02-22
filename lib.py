import os

import lancedb
import openai
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from pydantic import Field, FilePath

# connect to LanceDB
db = lancedb.connect("~/.golden-retriever/lancedb")

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    text: str = embeddings.SourceField()
    vector: Vector(1024) = embeddings.VectorField()


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

# you can use table.list_indices() to make sure indices have been created
reranker = RRFReranker()
results = (
    table.search(
        "flower moon",
        query_type="hybrid",
        vector_column_name="vector",
        fts_columns="text",
    )
    .rerank(reranker)
    .limit(10)
    .to_pandas()
)
print(results)
