import os
import sys
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

# Add app to path so we can import our own modules
sys.path.insert(0, os.path.dirname(__file__))

import google.generativeai as genai
from app.ingestion.loader import load_pdf

# ── Config ────────────────────────────────────────────────────────────────────
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

PROMPT_TEMPLATE = """You are building an evaluation dataset for a RAG (Retrieval-Augmented Generation) system.

Given the text below from page {page_num} of a document, generate exactly 3 question-answer pairs.

Text:
{text}

Rules:
- Questions must be specific and answerable using ONLY the text above
- Avoid trivial or overly obvious questions
- Answers must be concise, factual, and taken directly from the text
- Do not use Markdown formatting in answers — plain text only
- Return ONLY a valid JSON array with no extra text, no code fences, no explanation:
[
    {{"question": "...", "answer": "...", "page_num": {page_num}}},
    {{"question": "...", "answer": "...", "page_num": {page_num}}},
    {{"question": "...", "answer": "...", "page_num": {page_num}}}
]"""


def generate_qa_for_page(text: str, page_num: int) -> list:
    prompt = PROMPT_TEMPLATE.format(text=text.strip(), page_num=page_num)

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Strip code fences if Gemini adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        pairs = json.loads(raw)
        return pairs

    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse failed for page {page_num}: {e}")
        print(f"  Raw response: {response.text[:200]}")
        return []
    except Exception as e:
        print(f"  [!] Error on page {page_num}: {e}")
        return []


def generate_eval_set(pdf_path: str, output_path: str, max_pages: int = None):
    """
    Main function — loads PDF, generates Q&A pairs page by page, saves to JSON.

    Args:
        pdf_path: path to the PDF file
        output_path: where to save eval_set.json
        max_pages: optional limit on pages to process (useful for large PDFs)
    """
    print(f"\n Loading PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    print(f" Found {len(pages)} non-blank pages")

    if max_pages:
        pages = pages[:max_pages]
        print(f" Processing first {max_pages} pages only")

    all_pairs = []
    failed_pages = []

    for i, page in enumerate(pages):
        page_num = page["page_num"]
        text = page["text"]

        # Skip very short pages — not enough content for good questions
        if len(text.split()) < 50:
            print(f"  [skip] Page {page_num} — too short ({len(text.split())} words)")
            continue

        print(f"  Generating Q&A for page {page_num} ({i+1}/{len(pages)})...")
        pairs = generate_qa_for_page(text, page_num)

        if pairs:
            all_pairs.extend(pairs)
            print(f"  Got {len(pairs)} pairs — total so far: {len(all_pairs)}")
        else:
            failed_pages.append(page_num)

        # Rate limiting — Gemini free tier allows ~15 requests/minute
        time.sleep(4)

    # Add unique IDs to every pair
    eval_set = [
        {
            "id": f"q{i+1}",
            "question": pair["question"],
            "expected_answer": pair["answer"],
            "page_num": pair.get("page_num"),
            "reviewed": False,   # flip to True after manual review
        }
        for i, pair in enumerate(all_pairs)
    ]

    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2, ensure_ascii=False)

    print(f"\n Done!")
    print(f" Generated {len(eval_set)} Q&A pairs")
    print(f" Saved to: {output_path}")

    if failed_pages:
        print(f" Failed pages: {failed_pages}")

    print(f"\n Next step: open {output_path} and manually review.")
    print(" Delete bad questions, fix wrong answers, set 'reviewed': true on good ones.")
    print(" Aim to keep the best 25-30 pairs.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval set from a PDF")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--output", default="../eval/eval_set.json", help="Output path for eval_set.json")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to process (optional)")
    args = parser.parse_args()

    generate_eval_set(
        pdf_path=args.pdf,
        output_path=args.output,
        max_pages=args.max_pages,
    )
