"""
STEP 3: Query the IRT Knowledge Base
Search similar issues and get AI-generated answers.

Usage:
  python step3_query.py                             # interactive mode
  python step3_query.py --query "v2 dataset failed" # single query
"""

import os, re, argparse
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION  = "irt_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
STORAGE_DIR = "./qdrant_storage"
TOP_K       = 5

ai = OpenAI()


def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search(query: str, embedder, qclient) -> list:
    vec     = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    results = qclient.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=TOP_K,
        with_payload=True,
    )
    return [
        {
            "score"             : round(r.score, 3),
            "summary"           : r.payload.get("summary", ""),
            "solution"          : clean(r.payload.get("solution", "")),
            "resolution_status" : r.payload.get("resolution_status", ""),
            "severity"          : r.payload.get("severity", ""),
            "bug_category"      : r.payload.get("bug_category", ""),
            "environment"       : r.payload.get("environment", ""),
            "references"        : r.payload.get("references", "None"),
            "team"              : r.payload.get("team", ""),
        }
        for r in results
    ]


def generate_answer(query: str, hits: list) -> str:
    context = ""
    for i, h in enumerate(hits, 1):
        context += f"""
Issue #{i} (relevance: {h['score']})
  Summary    : {h['summary']}
  Category   : {h['bug_category']}
  Severity   : {h['severity']}
  Environment: {h['environment']}
  Status     : {h['resolution_status']}
  Solution   : {h['solution']}
  References : {h['references']}
  Team       : {h['team']}
"""

    resp = ai.responses.create(
        model=os.environ.get("OPENAI_MODEL_ANSWER", "gpt-4.1"),
        max_output_tokens=900,
        input=f"""
You are an expert IRT (Incident Response Team) support assistant for ConverSight.

A team member asked:
"{query}"

Relevant past issues from our knowledge base:
{context}

Instructions:
- If this is a known issue, clearly state what the root cause was and how it was fixed
- Provide specific step-by-step actions if applicable
- Mention any references or links
- If this looks like a new issue not in the KB, say so and suggest who to escalate to
- Keep response under 350 words
- Use bullet points for steps
- Use *bold* for key terms (not markdown headers)
""",
    )
    return (resp.output_text or "").strip()


def run_query(query: str, embedder, qclient, show_hits=True):
    print(f"\n{'='*60}")
    print(f"  Query: {query}")
    print("="*60)

    hits = search(query, embedder, qclient)

    if not hits:
        print("  ❌ No similar issues found in knowledge base.")
        return

    if show_hits:
        icons = {"Fixed": "✅", "Partial": "⚠️", "Unresolved": "❌", "Rejected": "🚫"}
        print(f"\n  Similar past issues:\n")
        for i, h in enumerate(hits[:3], 1):
            icon  = icons.get(h["resolution_status"], "❓")
            bar   = "█" * int(h["score"] * 10) + "░" * (10 - int(h["score"] * 10))
            print(f"  {i}. {icon} {h['summary'][:65]}")
            print(f"     {bar} {int(h['score']*100)}% match | {h['resolution_status']} | {h['bug_category']}")
            if h["references"] not in ("None", "nan", ""):
                print(f"     📎 {h['references'][:90]}")
            print()

    print("  🤖 AI Answer:\n")
    answer = generate_answer(query, hits)
    print(answer)
    print()


def interactive(embedder, qclient):
    print()
    print("=" * 60)
    print("  🐛 IRT Knowledge Base — Interactive Query")
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
            run_query(q, embedder, qclient)
        except KeyboardInterrupt:
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY missing in .env")
        return

    print("  Loading embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("  Connecting to Qdrant …")
    qclient = QdrantClient(path=STORAGE_DIR)

    existing = [c.name for c in qclient.get_collections().collections]
    if COLLECTION not in existing:
        print(f"❌ Collection '{COLLECTION}' not found!")
        print("   Run step2_build_knowledge_base.py first.")
        return

    count = qclient.count(collection_name=COLLECTION).count
    print(f"  ✅ Knowledge base ready: {count} documents\n")

    if args.query:
        run_query(args.query, embedder, qclient)
    else:
        interactive(embedder, qclient)


if __name__ == "__main__":
    main()
