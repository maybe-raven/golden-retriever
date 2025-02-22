import os
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# connect to LanceDB
db = lancedb.connect("~/.golden-retriever/lancedb")

# Configuring the environment variable OPENAI_API_KEY
os.environ.setdefault("OPENAI_API_KEY", "...")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
embeddings = get_registry().get("openai").create()


class Documents(LanceModel):
    text: str = embeddings.SourceField()
    vector: Vector(1024) = embeddings.VectorField()


table_name = "documents"
table = db.create_table(table_name, schema=Documents)
