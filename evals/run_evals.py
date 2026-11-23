"""Score the assistant against questions.csv and write results.md.

Run from the project folder:
    python3 evals/run_evals.py              all questions
    python3 evals/run_evals.py --limit 10   only the first 10 (saves free-tier quota)
    python3 evals/run_evals.py --max-wrong 1

A provider error (rate limit, no quota, network) is reported as "inconclusive". It is never counted as
a wrong answer and never counted as a pass. The exit code is:
    1  more wrong answers than --max-wrong (default 0)
    2  no wrong answers, but some questions were inconclusive, so the run proves nothing yet
    0  everything was decided and within the limit
"""
import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import rag  # noqa: E402


def contains(answer, rule):
    """rule: groups separated by ';' must all match; inside a group any '|' option matches."""
    if not rule:
        return True
    text = answer.lower().replace(",", "")
    return all(any(opt.strip().lower() in text for opt in group.split("|")) for group in rule.split(";"))


def judge(row, r):
    """Return (outcome, note). outcome is pass, fail or inconclusive."""
    if r["status"] == "error":
        return "inconclusive", "provider error: " + r["answer"][:110]
    cited = [c["doc_id"] for c in r["citations"]]
    if row["expected"] == "answer":
        ok_status = r["status"] == "answered"
        ok_cite = row["expected_doc"] in cited
        ok_text = contains(r["answer"], row["must_contain"])
        if ok_status and ok_cite and ok_text:
            return "pass", ""
        note = ", ".join(
            x for x, bad in (("refused", not ok_status), ("wrong or missing citation", ok_status and not ok_cite), ("answer missing key fact", ok_status and not ok_text)) if bad
        )
        return "fail", note
    return ("pass", "") if r["status"] == "refused" else ("fail", "answered a question it should refuse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    ap.add_argument("--max-wrong", type=int, default=0, help="most wrong answers allowed before the run fails")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(HERE / "questions.csv", newline="")))
    if args.limit:
        rows = rows[: args.limit]
    ai = rag.ai_enabled()
    mode = "AI (OpenRouter)" if ai else "search-only (no API key)"
    print("Running %d questions in %s. AI mode takes about 4 minutes; progress shows below." % (len(rows), mode), flush=True)
    results, streak = [], 0
    for row in rows:
        try:
            r = rag.ask(row["question"])
        except rag.RagError as e:
            r = {"status": "error", "answer": str(e), "citations": [], "reason": "error"}
        outcome, note = judge(row, r)
        results.append((row, r, outcome, note))
        print("  %2d/%d  %-12s %s" % (len(results), len(rows), outcome, row["question"][:60]), flush=True)
        streak = streak + 1 if outcome == "inconclusive" else 0
        if streak >= 3:
            print("\nStopping early: 3 provider errors in a row. Last error: %s" % r["answer"][:200], flush=True)
            print("On OpenRouter free models the daily limit is 50 requests (1,000 after $10 of credits). Try again tomorrow.", flush=True)
            break
        time.sleep(3.5 if ai else 0)  # stay under free-tier rate limits

    skipped = len(rows) - len(results)
    ans = [x for x in results if x[0]["expected"] == "answer"]
    ref = [x for x in results if x[0]["expected"] == "refuse"]
    count = lambda group, outcome: sum(1 for x in group if x[2] == outcome)
    wrong_answers = count(ref, "fail")
    inconclusive = count(results, "inconclusive") + skipped
    cite_ok = sum(1 for x in ans if x[1]["status"] == "answered" and x[0]["expected_doc"] in [c["doc_id"] for c in x[1]["citations"]])

    lines = [
        "# Evaluation results",
        "",
        "Mode: %s" % mode,
        "",
        "| Measure | Result |",
        "|---|---|",
        "| Answerable questions answered correctly | %d / %d |" % (count(ans, "pass"), len(ans)),
        "| Answers that cite the right policy | %d / %d |" % (cite_ok, len(ans)),
        "| Out-of-scope questions correctly refused | %d / %d |" % (count(ref, "pass"), len(ref)),
        "| **Wrong answers given instead of refusing (target: 0)** | **%d** |" % wrong_answers,
        "| Inconclusive (provider errors or not run, not counted above) | %d |" % inconclusive,
        "",
        "## Every question",
        "",
        "| # | Question | Expected | Result | Note |",
        "|---|---|---|---|---|",
    ]
    for row, r, outcome, note in results:
        lines.append("| %s | %s | %s | %s | %s |" % (row["id"], row["question"].replace("|", "/"), row["expected"], outcome.upper() if outcome != "pass" else "pass", note.replace("|", "/")))
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:11]))
    bad = [x for x in results if x[2] != "pass"]
    if bad:
        print("\nNot passed:")
        for row, r, outcome, note in bad:
            print("  #%s [%s] %s -> %s" % (row["id"], outcome, row["question"], note))
    if wrong_answers > args.max_wrong:
        sys.exit(1)
    sys.exit(2 if inconclusive else 0)


if __name__ == "__main__":
    main()
