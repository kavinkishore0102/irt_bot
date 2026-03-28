"""
STEP 2: Build Qdrant Knowledge Base
Reads bugs_enriched.xlsx → embeds → stores in local Qdrant

Run: python step2_build_knowledge_base.py
"""

import os, re, uuid
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()

INPUT_FILE  = "bugs_enriched.xlsx"
COLLECTION  = "irt_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free local model — no API needed
VECTOR_DIM  = 384
BATCH_SIZE  = 50
STORAGE_DIR = "./qdrant_storage"


def clean(text) -> str:
    if not isinstance(text, str) or text in ("nan", "None", ""):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def build_document(row) -> str:
    """Combine all fields into one searchable text."""
    parts = []
    summary  = str(row.get("Summary",      "")).strip()
    details  = clean(str(row.get("Details",   "")))
    solution = clean(str(row.get("Solution",  "")))
    comments = clean(str(row.get("Comments",  "")))
    category = str(row.get("Bug Category", "")).strip()
    env      = str(row.get("Environment",  "")).strip()

    if summary:                        parts.append(f"Issue: {summary}")
    if category and category != "nan": parts.append(f"Category: {category}")
    if env      and env      != "nan": parts.append(f"Environment: {env}")
    if details:                        parts.append(f"Problem: {details[:400]}")
    if solution:                       parts.append(f"Solution: {solution}")
    if comments:                       parts.append(f"Discussion: {comments[:600]}")

    return "\n".join(parts)


def main():
    print()
    print("=" * 60)
    print("  STEP 2 — Build Qdrant Knowledge Base")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found.")
        print("   Run step1_enrich_excel.py first!")
        return

    # Load data
    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded: {len(df)} rows from {INPUT_FILE}")

    # Only index useful rows.
    # Accept either "Resolution Status" (recommended) or "status" (requirement wording).
    resolution_col = "Resolution Status" if "Resolution Status" in df.columns else ("status" if "status" in df.columns else None)
    if not resolution_col:
        print("❌ Missing column: 'Resolution Status' (or 'status').")
        print("   Run step1 (irt_enrich_excel.py) first to generate it.")
        return

    useful = df[
        df[resolution_col].isin(["Fixed", "Partial"]) |
        (df.get("Status") == "Done")
    ].copy()
    print(f"  Useful rows to index (Fixed/Partial/Done): {len(useful)}")

    # Load embedding model
    print(f"\n  Loading embedding model: {EMBED_MODEL} …")
    print("  (Downloads ~90MB on first run, then cached locally)")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("  ✅ Embedding model ready")

    # Connect to local Qdrant
    print(f"\n  Connecting to local Qdrant storage: {STORAGE_DIR}")
    qclient = QdrantClient(path=STORAGE_DIR)

    # Recreate collection
    existing = [c.name for c in qclient.get_collections().collections]
    if COLLECTION in existing:
        qclient.delete_collection(COLLECTION)
        print(f"  Deleted old collection '{COLLECTION}'")

    qclient.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"  ✅ Collection '{COLLECTION}' created (dim={VECTOR_DIM}, cosine)")

    # Process in batches
    print(f"\n  Indexing {len(useful)} documents (batch size={BATCH_SIZE}) …\n")

    batch_docs = []
    batch_meta = []
    total_stored = 0

    for _, row in useful.iterrows():
        doc = build_document(row)
        if not doc.strip():
            continue

        meta = {
            "summary"           : str(row.get("Summary",           ""))[:200],
            "solution"          : str(row.get("Solution",          ""))[:500],
            "resolution_status" : str(row.get("Resolution Status", "")),
            "status"            : str(row.get("Status",            "")),
            "severity"          : str(row.get("Severity",          "")),
            "bug_category"      : str(row.get("Bug Category",      "")),
            "environment"       : str(row.get("Environment",       "")),
            "references"        : str(row.get("References",        "")),
            "team"              : str(row.get("Team/Department",   "")),
            "assignee"          : str(row.get("Assignee",          "")),
            "date_submitted"    : str(row.get("Date submitted",    "")),
            "doc_text"          : doc[:800],
        }

        batch_docs.append(doc)
        batch_meta.append(meta)

        if len(batch_docs) >= BATCH_SIZE:
            vectors = embedder.encode(batch_docs, show_progress_bar=False).tolist()
            points  = [
                PointStruct(id=str(uuid.uuid4()), vector=v, payload=m)
                for v, m in zip(vectors, batch_meta)
            ]
            qclient.upsert(collection_name=COLLECTION, points=points)
            total_stored += len(points)
            print(f"  Stored: {total_stored}/{len(useful)} documents")
            batch_docs = []
            batch_meta = []

    # Store remaining batch
    if batch_docs:
        vectors = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        points  = [
            PointStruct(id=str(uuid.uuid4()), vector=v, payload=m)
            for v, m in zip(vectors, batch_meta)
        ]
        qclient.upsert(collection_name=COLLECTION, points=points)
        total_stored += len(points)

    # Verify
    final_count = qclient.count(collection_name=COLLECTION).count

    print()
    print("=" * 60)
    print(f"  ✅ Knowledge base built!")
    print("=" * 60)
    print(f"  Collection  : {COLLECTION}")
    print(f"  Documents   : {final_count}")
    print(f"  Storage     : {STORAGE_DIR}/")
    print()
    print("  ➡️  Next: run  python step3_query.py")


if __name__ == "__main__":
    main()
