"""
Run the eval set through the RAG pipeline and score retrieval + answer
quality. Produces a JSON results file and prints a summary.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --eval-set eval/eval_set.json --chunk-size 500 --k 4 --label "chunk500_k4"

Each run indexes the eval set's videos into an isolated index
(eval/results/<label>/indexes/) so it never touches your live app data in
data/faiss_index/.
"""
import argparse
import json
import time
from pathlib import Path

from rag import ingest, vectorstore, qa_chain, config as rag_config
from eval import metrics


def load_eval_set(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def run(
    eval_set_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    max_distance: float,
    label: str,
) -> dict:
    questions = load_eval_set(eval_set_path)
    run_dir = Path(__file__).parent / "results" / label
    index_root = run_dir / "indexes"
    index_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Run '{label}' (chunk_size={chunk_size}, overlap={chunk_overlap}, k={k}) ===")

    # 1. Index every unique video referenced in the eval set, into an
    #    isolated index for this config.
    video_urls = sorted({q["video_url"] for q in questions})
    for url in video_urls:
        print(f"Indexing {url} ...")
        ingest.ingest_video(
            url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            index_root=index_root,
            update_manifest=False,
        )

    store = vectorstore.load_store(video_id=None, index_root=index_root)

    # 2. Run every question through the pipeline and score it.
    results = []
    for q in questions:
        t0 = time.time()
        answer_result = qa_chain.answer_question(store, q["question"], k=k, max_distance=max_distance)
        latency = time.time() - t0

        row = {
            "id": q["id"],
            "question": q["question"],
            "out_of_scope": q.get("out_of_scope", False),
            "grounded": answer_result.grounded,
            "answer": answer_result.answer,
            "num_sources": len(answer_result.sources),
            "best_distance": min((s.distance for s in answer_result.sources), default=None),
            "latency_seconds": round(latency, 2),
        }

        if q.get("out_of_scope"):
            # For out-of-scope questions, correctness = did the system correctly refuse?
            row["correctly_refused"] = not answer_result.grounded
        else:
            loc_hit = metrics.retrieval_localization_hit(
                answer_result.sources, q.get("expected_timestamp_seconds")
            )
            row["retrieval_localization_hit"] = loc_hit

            if answer_result.grounded:
                faith = metrics.score_faithfulness(answer_result.context, answer_result.answer)
                correct = metrics.score_correctness(
                    q["question"], q.get("expected_answer", ""), answer_result.answer
                )
                row["faithfulness_score"] = faith.score
                row["faithfulness_reason"] = faith.reason
                row["correctness_score"] = correct.score
                row["correctness_reason"] = correct.reason
            else:
                row["faithfulness_score"] = None
                row["correctness_score"] = None

        results.append(row)
        print(f"  [{q['id']}] grounded={row['grounded']} "
              f"correctness={row.get('correctness_score')} "
              f"faithfulness={row.get('faithfulness_score')}")

    summary = summarize(results)
    output = {
        "label": label,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k": k,
            "max_distance": max_distance,
        },
        "summary": summary,
        "results": results,
    }

    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path}")
    print_summary(summary)
    return output


def summarize(results: list) -> dict:
    in_scope = [r for r in results if not r["out_of_scope"]]
    out_scope = [r for r in results if r["out_of_scope"]]

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    loc_checked = [r for r in in_scope if r.get("retrieval_localization_hit") is not None]

    return {
        "num_questions": len(results),
        "num_in_scope": len(in_scope),
        "num_out_of_scope": len(out_scope),
        "avg_faithfulness": avg([r.get("faithfulness_score") for r in in_scope]),
        "avg_correctness": avg([r.get("correctness_score") for r in in_scope]),
        "retrieval_localization_hit_rate": (
            round(sum(r["retrieval_localization_hit"] for r in loc_checked) / len(loc_checked), 2)
            if loc_checked else None
        ),
        "out_of_scope_correct_refusal_rate": (
            round(sum(r["correctly_refused"] for r in out_scope) / len(out_scope), 2)
            if out_scope else None
        ),
        "avg_latency_seconds": avg([r["latency_seconds"] for r in results]),
    }


def print_summary(summary: dict) -> None:
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default=str(Path(__file__).parent / "eval_set.json"))
    parser.add_argument("--chunk-size", type=int, default=rag_config.CHUNK_SIZE_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=rag_config.CHUNK_OVERLAP_CHARS)
    parser.add_argument("--k", type=int, default=rag_config.TOP_K)
    parser.add_argument("--max-distance", type=float, default=rag_config.MAX_DISTANCE_FOR_GROUNDED_ANSWER)
    parser.add_argument("--label", default="default_run")
    args = parser.parse_args()

    run(
        eval_set_path=Path(args.eval_set),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k,
        max_distance=args.max_distance,
        label=args.label,
    )


if __name__ == "__main__":
    main()
