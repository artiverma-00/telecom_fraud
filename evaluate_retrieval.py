"""Evaluate top-3 ticket retrieval against ten hand-crafted questions.

Run from this folder after running the ticket ingestion script:
    uv run python evaluate_retrieval.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from retriever import build_ticket_store


EVAL_CASES = [
    ("I switched from 3G to 4G and now mobile internet does not work.", "TK-001"),
    ("My signal keeps dropping to zero in my suburb and calls and data disconnect.", "TK-002"),
    ("I bought a 5 GB add-on, but the data balance in my app has not changed.", "TK-003"),
    ("I returned from Spain and got a large roaming bill even though I had a bundle.", "TK-004"),
    ("After updating Android, my phone says SIM not provisioned.", "TK-005"),
    ("My bank statement has two identical monthly plan charges on the same day.", "TK-006"),
    ("All incoming calls go straight to voicemail while my phone has signal.", "TK-007"),
    ("My 4G download speed is below 1 Mbps in the city centre.", "TK-008"),
    ("The QR code scan fails when I try to activate eSIM on my iPhone.", "TK-009"),
    ("The PDF bill download button spins and I cannot view the itemised bill.", "TK-010"),
]


def evaluate_top3_recall() -> float:
    store = build_ticket_store()
    stored_count = store._collection.count()
    if stored_count == 0:
        raise SystemExit(
            "The tickets collection is empty. From project_2, run "
            "`uv run python ingest_ticket.py` and then rerun this evaluation."
        )

    hits = 0
    for question, expected_ticket_id in EVAL_CASES:
        results = store.similarity_search_with_score(question, k=3)
        retrieved_ids = [str(doc.metadata.get("ticket_id", "")) for doc, _ in results]
        hit = expected_ticket_id in retrieved_ids
        hits += int(hit)
        mark = "HIT " if hit else "MISS"
        print(f"{mark} expected={expected_ticket_id} top3={retrieved_ids} | {question}")

    recall = hits / len(EVAL_CASES)
    print(f"Top-3 recall: {hits}/{len(EVAL_CASES)} = {recall:.1%}")
    return recall


if __name__ == "__main__":
    evaluate_top3_recall()
