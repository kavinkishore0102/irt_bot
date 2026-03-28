"""
STEP 1: Enrich bugs_with_comments.xlsx
Adds: Solution, Resolution Status, References columns
Uses: Anthropic Claude

Run: python step1_enrich_excel.py
"""

import os, re, time, json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE  = "bugs_with_comments.xlsx"
OUTPUT_FILE = "bugs_enriched.xlsx"

client = OpenAI()
REQUEST_SLEEP_SECONDS = float(os.environ.get("REQUEST_SLEEP_SECONDS", "0.2"))

# ── helpers ───────────────────────────────────────────────────────────────────

def clean(text) -> str:
    if not isinstance(text, str) or text in ("nan", "None", ""):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def extract_solution(summary, comments) -> dict:
    prompt = f"""Analyze this IRT support ticket and extract key information.

Summary : {summary}
Comments: {clean(comments)[:1200]}

Extract:
1. SOLUTION: What was the actual fix or workaround applied? Be specific and technical.
   If no solution found, write "No solution documented."
2. RESOLUTION_STATUS: Exactly one of — Fixed / Partial / Unresolved / Rejected
3. REFERENCES: Any URLs, ticket IDs, or technical references (comma separated or "None")

Respond ONLY in this exact JSON format, nothing else:
{{"solution": "...", "resolution_status": "Fixed|Partial|Unresolved|Rejected", "references": "..."}}"""

    backoff = 1.5
    for attempt in range(3):
        try:
            resp = client.responses.create(
                model=os.environ.get("OPENAI_MODEL_EXTRACT", "gpt-4.1-mini"),
                input=prompt,
                max_output_tokens=500,
            )
            text = (resp.output_text or "").strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            if attempt == 2:
                print(f"    [Error] {e}")
            time.sleep((2 * (backoff ** attempt)))

    return {
        "solution"          : "Unable to extract solution.",
        "resolution_status" : "Unresolved",
        "references"        : "None"
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY missing in .env")
        return

    print()
    print("=" * 60)
    print("  STEP 1 — Enrich Excel with Solution + Resolution Status")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found in current directory")
        return

    df = pd.read_excel(INPUT_FILE)
    print(f"  Loaded: {len(df)} rows from {INPUT_FILE}")

    # Fix corrupted Status values
    valid_statuses = {
        "Done", "Rejected", "Triaged", "New", "Move to Asana",
        "Under RCA", "Blocked", "In progress", "Backlog", "Testing"
    }
    df["Status"] = df["Status"].apply(
        lambda x: x if x in valid_statuses else "New"
    )

    # Add new columns if not already present
    if "Solution" not in df.columns:
        df["Solution"] = ""
    # Requirement calls it "status" (fixed or not). We store both for compatibility.
    if "Resolution Status" not in df.columns:
        df["Resolution Status"] = ""
    if "status" not in df.columns:
        df["status"] = ""
    if "References" not in df.columns:
        df["References"] = ""

    total     = len(df)
    processed = 0

    for idx, row in df.iterrows():

        # Skip already processed rows
        if pd.notna(row.get("Solution")) and str(row.get("Solution","")).strip() not in ("", "nan"):
            continue

        summary  = str(row.get("Summary",  "")).strip()
        comments = str(row.get("Comments", "")).strip()

        # Skip rows with no useful data
        if not summary or (not comments):
            df.at[idx, "Solution"]          = "No solution documented."
            df.at[idx, "Resolution Status"] = "Unresolved"
            df.at[idx, "status"]            = "Unresolved"
            df.at[idx, "References"]        = "None"
            continue

        result = extract_solution(summary, comments)

        df.at[idx, "Solution"]          = result.get("solution", "")
        df.at[idx, "Resolution Status"] = result.get("resolution_status", "Unresolved")
        df.at[idx, "status"]            = df.at[idx, "Resolution Status"]
        df.at[idx, "References"]        = result.get("references", "None")

        processed += 1

        if processed % 10 == 0 or processed == 1:
            print(f"  [{processed}/{total}] {summary[:55]}…")
            print(f"           → {result.get('resolution_status')} | {result.get('solution','')[:70]}")

        # Checkpoint every 50 rows so you don't lose progress
        if processed % 50 == 0:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"  💾 Checkpoint saved at row {idx}")

        time.sleep(REQUEST_SLEEP_SECONDS)

    # Final save
    df.to_excel(OUTPUT_FILE, index=False)

    fixed      = (df["Resolution Status"] == "Fixed").sum()
    partial    = (df["Resolution Status"] == "Partial").sum()
    unresolved = (df["Resolution Status"] == "Unresolved").sum()
    rejected   = (df["Resolution Status"] == "Rejected").sum()

    print()
    print("=" * 60)
    print(f"  ✅ Done! Saved → {OUTPUT_FILE}")
    print("=" * 60)
    print(f"  Fixed      : {fixed}")
    print(f"  Partial    : {partial}")
    print(f"  Unresolved : {unresolved}")
    print(f"  Rejected   : {rejected}")
    print(f"  Total      : {total}")
    print()
    print("  ➡️  Next: run  python step2_build_knowledge_base.py")


if __name__ == "__main__":
    main()
