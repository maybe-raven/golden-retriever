import lancedb
import pandas as pd
import pyarrow as pa


def main():
    uri = "data/test"
    db = lancedb.connect(uri)

    data = [
        {"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
        {"vector": [5.9, 26.5], "item": "bar", "price": 20.0},
    ]

    tbl = db.create_table("my_table", data=data)
    
if __name__ == "__main__":
    main()