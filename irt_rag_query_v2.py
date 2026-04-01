"""
STEP 3 (v2): Query the IRT Knowledge Base

Run:
  python irt_rag_query_v2.py
  python irt_rag_query_v2.py --query "v2 dataset failed" --min-score 0.30
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION = "irt_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
STORAGE_DIR = "./qdrant_storage"
TOP_K       = 5

ai = OpenAI()


def clean(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search(query: str, embedder: SentenceTransformer, qclient: QdrantClient, top_k: int) -> list[dict[str, Any]]:
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()

    if hasattr(qclient, "search"):
        results = qclient.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
    else:
        inner = getattr(qclient, "_client", qclient)
        results = inner.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )

    out: list[dict[str, Any]] = []
    for r in results:
        out.append({
            "score"             : float(r.score),
            "summary"           : r.payload.get("summary", ""),
            "solution"          : clean(r.payload.get("solution", "")),
            "resolution_status" : r.payload.get("final_status",
                                  r.payload.get("resolution_status",
                                  r.payload.get("status", ""))),
            "severity"          : r.payload.get("severity", ""),
            "bug_category"      : r.payload.get("bug_category", ""),
            "environment"       : r.payload.get("environment", ""),
            "references"        : r.payload.get("references", "None"),
            "team"              : r.payload.get("team", ""),
        })
    return out


def generate_answer(query: str, hits: list[dict[str, Any]]) -> str:
    # Build context — include only hits with real solutions
    context = ""
    for i, h in enumerate(hits, 1):
        sol = h["solution"]
        if not sol or sol in ("nan", "None", "No solution documented."):
            sol = "(No specific solution recorded)"
        context += f"""
Case {i} (relevance: {h['score']:.2f})
  Issue      : {h['summary']}
  Category   : {h['bug_category']}
  Status     : {h['resolution_status']}
  Solution   : {sol}
  References : {h['references']}
"""

    resp = ai.responses.create(
        model=os.environ.get("OPENAI_MODEL_ANSWER", "gpt-4.1"),
        max_output_tokens=900,
        input=f"""
You are an IRT (Incident Response Team) support assistant for ConverSight.
You help IRT team members and clients resolve issues based on what has worked before.

Someone asked:
"{query}"

Here are the most relevant past cases the IRT team has handled:
{context}

Write your answer following these rules strictly:

1. Start with: "Yes, the IRT team has seen this before." if any case matches.
   OR "This looks like a new issue — please raise it with the IRT team." if none match.

2. For each relevant past case, write one short paragraph:
   "In a similar case where [plain-English description of the issue], the fix was: [plain-English description of what was done]."
   Then add: "This was a permanent fix." OR "This was a workaround."

   IMPORTANT:
   - Use the EXACT action from the Solution field — do not generalize or invent steps.
   - Write it so a non-developer can understand what was done.
     BAD:  "livy node count increased and pods restarted"
     GOOD: "The processing service was scaled up and restarted to clear the stuck job."
     BAD:  "clean up entities in Qdrant"
     GOOD: "Outdated records were cleaned up from the data store so the system could rebuild them correctly."
   - Keep IRT terms (SME publish, republish, vacuum, dataset activation, org ID) but briefly clarify if the meaning is not obvious.

3. Then write: "Here are the steps to try:"
   List 2-4 specific steps based only on the actual solutions above.
   - Write each step as a clear action a user or IRT member can follow.
   - No generic advice unless it was the recorded fix.

4. End with:
   "If these steps do not resolve the issue, please contact the IRT team and share:
   - Dataset name
   - Org ID
   - Environment (Production / Staging)
   - What you see on screen right now"

Rules:
- Do NOT say "knowledge base", "Case #1", or "previous tickets".
- Do NOT use raw technical strings like pod names, internal command syntax, or stack traces in the answer.
- Do NOT mention staging environment fixes. If staging was the root cause, say "an environment configuration issue."
- Keep under 350 words. Use bullet points for steps.
""",
    )
    return (resp.output_text or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",     "-q", type=str,   help="Single query to run")
    parser.add_argument("--top-k",          type=int,   default=TOP_K)
    parser.add_argument("--min-score",      type=float, default=0.0,
                        help="If top hit is below this, treat as not found")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY missing (export it or set it in .env)")

    print("  Loading embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("  Connecting to Qdrant …")
    qclient = QdrantClient(path=STORAGE_DIR)

    existing = [c.name for c in qclient.get_collections().collections]
    if COLLECTION not in existing:
        print(f"❌ Collection '{COLLECTION}' not found!")
        print("   Run irt_rag_build_knowledge_base_v2.py --recreate first.")
        return

    count = qclient.count(collection_name=COLLECTION).count
    print(f"  ✅ Knowledge base ready: {count} documents\n")

    def run_one(q: str) -> None:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print("=" * 60)

        hits = search(q, embedder, qclient, args.top_k)
        if not hits or (args.min_score and hits[0]["score"] < args.min_score):
            print("  ❌ No similar issues found in knowledge base.")
            return

        print("  🤖 AI Answer:\n")
        print(generate_answer(q, hits))
        print()

    if args.query:
        run_one(args.query)
        return

    print("=" * 60)
    print("  🐛 IRT Knowledge Base — Interactive Query (v2)")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            q = input("❓ Question: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            run_one(q)
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()