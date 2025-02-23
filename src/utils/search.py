import os
import lancedb
from lancedb.embeddings import get_registry
from lancedb.rerankers import RRFReranker

# connect to LanceDB
db = lancedb.connect("~/.golden-retriever/lancedb")

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()

table = db.open_table("documents")
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
