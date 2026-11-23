"""Evals for the gateway parts. Run from the project folder:   python3 evals/gateway_evals.py

 1. Routing        offline. Does each question go to the right model tier?
 2. Cache safety   needs GEMINI_API_KEY. Paraphrases must hit, near-misses must NOT (target: 0 wrong hits).
 3. Retrieval      keyword-only offline; with a Gemini key it compares the weighted blend against rank fusion
                   (recall@3: is the right policy among the top 3 clauses?).

Writes evals/gateway_results.md. Exit code 1 if routing is wrong anywhere or the cache gives a wrong hit.
"""
import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import gateway  # noqa: E402
import rag  # noqa: E402


def read(name):
    return list(csv.DictReader(open(HERE / name, newline="")))


def routing(index):
    out = []
    for row in read("routing.csv"):
        rows, _ = index.search(row["question"], method="weighted")
        got = gateway.route(row["question"], rows)
        out.append((row["question"], row["expected"], got["tier"], ", ".join(got["why"])))
    return out


def cache_safety(index):
    """Store the answer to question a, then ask question b. A hit is only right when should_hit is yes."""
    if not index.vectors:
        return None
    out = []
    old = gateway.DB_PATH
    with tempfile.TemporaryDirectory() as tmp:
        gateway.DB_PATH = Path(tmp) / "eval.db"
        try:
            corpus = gateway.corpus_hash(index)
            for i, row in enumerate(read("cache_pairs.csv")):
                va, vb = rag.embed_query(row["a"]), rag.embed_query(row["b"])
                ra, _ = index.search(row["a"], qvec=va)
                rb, _ = index.search(row["b"], qvec=vb)
                gateway.cache_store(row["a"], va, ra[0]["chunk"].id, corpus, {"answer": "x", "citations": []}, terms=gateway.key_terms(row["a"], ra[0]["chunk"]))
                hit, sim, why = gateway.cache_lookup(row["b"], vb, rb[0]["chunk"].id, corpus, terms=gateway.key_terms(row["b"], rb[0]["chunk"]))
                out.append((row["a"], row["b"], row["should_hit"], "hit" if hit else "miss", sim, why))
                with gateway.db() as con:
                    con.execute("DELETE FROM cache")
        finally:
            gateway.DB_PATH = old
    return out


def retrieval(index):
    rows = [r for r in read("questions.csv") if r["expected"] == "answer"]
    methods = ["weighted", "rrf"] if index.vectors else ["weighted"]
    scores = {m: 0 for m in methods}
    for row in rows:
        qvec = rag.embed_query(row["question"]) if index.vectors else None
        for m in methods:
            top, _ = index.search(row["question"], k=3, qvec=qvec, method=m)
            scores[m] += row["expected_doc"] in [r["chunk"].doc_id for r in top]
    return len(rows), scores


def main():
    index = rag.get_index()
    lines = ["# Gateway evaluation results", ""]
    failed = False

    r = routing(index)
    ok = sum(1 for _, want, got, _ in r if want == got)
    failed |= ok != len(r)
    lines += ["## Model routing", "", "Right tier chosen: **%d / %d**" % (ok, len(r)), "", "| Question | Expected | Chosen | Why |", "|---|---|---|---|"]
    lines += ["| %s | %s | %s | %s |" % (q[:90], want, got, why) for q, want, got, why in r]

    try:
        c = cache_safety(index)
        cache_note = "Skipped: needs GEMINI_API_KEY for question embeddings."
    except rag.RagError as e:
        c, cache_note = None, "Skipped: could not reach Gemini (%s)." % e
        index.vectors = None  # keyword-only for the retrieval check below
    lines += ["", "## Cache safety", ""]
    if c is None:
        lines.append(cache_note)
        print("Cache safety:", cache_note)
    else:
        paraphrases = [x for x in c if x[2] == "yes"]
        near_miss = [x for x in c if x[2] == "no"]
        hits = sum(1 for x in paraphrases if x[3] == "hit")
        wrong = sum(1 for x in near_miss if x[3] == "hit")
        failed |= wrong > 0
        lines += ["Paraphrases that reused the saved answer: **%d / %d**" % (hits, len(paraphrases)),
                  "Near-miss questions wrongly given a saved answer (target 0): **%d / %d**" % (wrong, len(near_miss)), "",
                  "| Saved question | New question | Should reuse | Result | Similarity | Reason |", "|---|---|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s | %.2f | %s |" % (a[:60], b[:60], s, got, sim, why) for a, b, s, got, sim, why in c]

    total, scores = retrieval(index)
    lines += ["", "## Retrieval (right policy in the top 3 clauses)", ""]
    for m, n in scores.items():
        lines.append("- %s: **%d / %d**" % (m, n, total))
    if len(scores) == 1:
        lines.append("- rrf: skipped (rank fusion needs meaning-based search, which needs GEMINI_API_KEY)")

    (HERE / "gateway_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
