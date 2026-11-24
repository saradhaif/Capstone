import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    backend="torch"
)

caption = json.loads(sys.stdin.read())["caption"]

embedding = model.encode(
    [caption],
    convert_to_numpy=True
).astype("float32")

print(json.dumps(embedding.tolist()))
