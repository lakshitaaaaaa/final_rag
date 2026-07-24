"""
Run the same eval set through several pipeline configurations and print a
side-by-side comparison table — this is the artifact that belongs in your
project report to justify design choices (e.g. chunk size, k).

Usage:
    python -m eval.compare

Edit CONFIGS below to try whatever variations you want to justify.
"""
import json
from pathlib import Path

from eval.run_eval import run

CONFIGS = [
    {"label": "chunk500_overlap100_k4", "chunk_size": 500, "chunk_overlap": 100, "k": 4},
    {"label": "chunk1000_overlap200_k4", "chunk_size": 1000, "chunk_overlap": 200, "k": 4},
    {"label": "chunk1000_overlap200_k2", "chunk_size": 1000, "chunk_overlap": 200, "k": 2},
]

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
MAX_DISTANCE = 1.0


def main():
    all_outputs = []
    for cfg in CONFIGS:
        output = run(
            eval_set_path=EVAL_SET_PATH,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            k=cfg["k"],
            max_distance=MAX_DISTANCE,
            label=cfg["label"],
        )
        all_outputs.append(output)

    print("\n\n=== Comparison across configs ===\n")
    headers = [
        "label", "chunk_size", "k", "avg_correctness", "avg_faithfulness",
        "retrieval_localization_hit_rate", "out_of_scope_correct_refusal_rate",
        "avg_latency_seconds",
    ]
    rows = []
    for o in all_outputs:
        rows.append([
            o["label"],
            o["config"]["chunk_size"],
            o["config"]["k"],
            o["summary"]["avg_correctness"],
            o["summary"]["avg_faithfulness"],
            o["summary"]["retrieval_localization_hit_rate"],
            o["summary"]["out_of_scope_correct_refusal_rate"],
            o["summary"]["avg_latency_seconds"],
        ])

    # Markdown table — paste straight into a report
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(v) for v in row) + " |")

    combined_path = Path(__file__).parent / "results" / "comparison.json"
    combined_path.write_text(
        json.dumps({"headers": headers, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved comparison table data to {combined_path}")


if __name__ == "__main__":
    main()
