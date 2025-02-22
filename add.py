import os
import lancedb
from lancedb.embeddings import get_registry

# connect to LanceDB
db = lancedb.connect("~/.golden-retriever/lancedb")

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()

data = [
    {"text": "rebel spaceships striking from a hidden base"},
    {"text": "have won their first victory against the evil Galactic Empire"},
    {"text": "during the battle rebel spies managed to steal secret plans"},
    {"text": "to the Empire's ultimate weapon the Death Star"},
]
table = db.open_table("documents")
table.add(data=data)
table.create_index(metric="L2", vector_column_name="vector", index_type="IVF_FLAT")
table.create_fts_index("text", replace=True)
