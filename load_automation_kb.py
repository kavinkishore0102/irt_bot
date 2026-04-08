"""
load_automation_kb.py
─────────────────────
Loads automation_categories.json into a Qdrant collection called `automation_kb`.

Each category becomes one document. The embedding is built from:
  - category name
  - description
  - all trigger phrases

Run:
    python load_automation_kb.py

To add a new category later:
    1. Add it to automation_categories.json in the same format.
    2. Re-run this script — it will recreate the collection fresh.
"""

import json, os, sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
AUTOMATION_KB_JSON = os.environ.get("AUTOMATION_KB_JSON", "./automation_categories.json")
STORAGE_DIR        = os.environ.get("QDRANT_STORAGE",     "./qdrant_storage")
COLLECTION         = "automation_kb"
EMBED_MODEL        = "all-MiniLM-L6-v2"
VECTOR_SIZE        = 384   # all-MiniLM-L6-v2 output dimension


def build_embedding_text(cat: dict) -> str:
    """
    Builds the text string that gets embedded.
    Combines category name + description + all trigger phrases
    so semantic search works naturally from user intent.
    """
    parts = [
        cat["category"],
        cat.get("description", ""),
    ]
    for phrase in cat.get("trigger_phrases", []):
        parts.append(phrase)
    return " | ".join(p.strip() for p in parts if p.strip())


def load():
    json_path = Path(AUTOMATION_KB_JSON)
    if not json_path.exists():
        print(f"❌  {AUTOMATION_KB_JSON} not found. Create it first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    print(f"📄  Loaded {len(categories)} categories from {json_path}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("⏳  Loading embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL)

    texts  = [build_embedding_text(c) for c in categories]
    print("⏳  Encoding categories …")
    vectors = embedder.encode(texts, normalize_embeddings=True)
    print(f"✅  Encoded {len(vectors)} vectors  (dim={vectors.shape[1]})")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    print("⏳  Connecting to Qdrant …")
    qclient = QdrantClient(path=STORAGE_DIR)

    # Recreate collection fresh — ensures any deleted/changed categories are gone
    existing = [c.name for c in qclient.get_collections().collections]
    if COLLECTION in existing:
        print(f"🔄  Dropping existing '{COLLECTION}' collection …")
        qclient.delete_collection(COLLECTION)

    qclient.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅  Created collection '{COLLECTION}'")

    # ── Build points ─────────────────────────────────────────────────────────
    points = []
    for i, (cat, vec) in enumerate(zip(categories, vectors)):
        points.append(PointStruct(
            id      = i,
            vector  = vec.tolist(),
            payload = cat,          # full JSON stored as payload
        ))

    qclient.upsert(collection_name=COLLECTION, points=points)
    print(f"✅  Inserted {len(points)} automation categories into '{COLLECTION}'")

    # ── Verify ────────────────────────────────────────────────────────────────
    count = qclient.count(collection_name=COLLECTION).count
    print(f"\n{'='*55}")
    print(f"  automation_kb ready — {count} categories")
    for cat in categories:
        fields_summary = ", ".join(
            f['key'] + ('*' if f['required'] else '')
            for f in cat.get('fields', [])
        )
        print(f"  ✅  {cat['category']}")
        print(f"       fields: {fields_summary}")
    print(f"{'='*55}\n")
    print("* = required field")


if __name__ == "__main__":
    load()
