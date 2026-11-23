"""gateway.py - the AI gateway that sits in front of the Policy Desk.

Every question goes through here. The gateway adds, around the guardrails in rag.py:

  * a semantic cache      (same meaning + same numbers + same policy clause = reuse the answer)
  * a token budget        (a daily allowance per team and a size cap per request)
  * model routing         (plain rules pick a small or a large model, with a fallback)
  * async batch jobs      (submit many questions, a background worker answers them)
  * tracing and metrics   (every request gets a trace id, step timings and a log row)

Nothing here uses a framework. State lives in one SQLite file (.cache/gateway.db).
"""
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import rag

DB_PATH = rag.CACHE_DIR / "gateway.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norm TEXT NOT NULL,
    question TEXT NOT NULL,
    vec TEXT,
    top_clause TEXT NOT NULL,
    numbers TEXT NOT NULL,
    terms TEXT NOT NULL DEFAULT '[]',
    corpus TEXT NOT NULL,
    payload TEXT NOT NULL,
    created REAL NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    team TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, team)
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace TEXT NOT NULL,
    ts REAL NOT NULL,
    team TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    cache TEXT NOT NULL,
    tier TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    ms INTEGER NOT NULL,
    spans TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    idem TEXT,
    team TEXT NOT NULL,
    status TEXT NOT NULL,
    items TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_idem ON jobs (team, idem) WHERE idem IS NOT NULL;
"""

_ready = set()
_ready_lock = threading.Lock()


@contextmanager
def db():
    """Open the database, save on success, always close. Use as:  with db() as con: ..."""
    path = Path(DB_PATH)
    path.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(str(path), timeout=10)
    con.row_factory = sqlite3.Row
    try:
        if str(path) not in _ready:
            with _ready_lock:
                con.executescript(SCHEMA)
                _ready.add(str(path))
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def clean_team(name):
    name = re.sub(r"[^a-z0-9_-]", "", (name or "").lower())[:32]
    return name or "default"


# ------------------------------------------------------------------ tokens and money
def estimate_tokens(text):
    """Rough size: about 4 characters per token. Real counts from the provider replace this when available."""
    return max(1, len(text) // 4)


def messages_tokens(messages):
    return sum(estimate_tokens(m["content"]) for m in messages)


def limits():
    return int(rag.cfg("DAILY_TOKEN_BUDGET", "20000")), int(rag.cfg("REQUEST_TOKEN_CAP", "3000"))


def price_per_1k(tier):
    """Notional rupees per 1,000 tokens. Free models cost nothing; this shows how cost would be tracked."""
    return float(rag.cfg("PRICE_LARGE_INR_PER_1K" if tier == "large" else "PRICE_SMALL_INR_PER_1K", "0.30" if tier == "large" else "0.05"))


def used_today(team):
    with db() as con:
        row = con.execute("SELECT tokens FROM usage WHERE day=? AND team=?", (today(), team)).fetchone()
    return row["tokens"] if row else 0


def record_usage(team, tokens):
    with db() as con:
        con.execute(
            "INSERT INTO usage (day, team, tokens, requests) VALUES (?, ?, ?, 1) "
            "ON CONFLICT(day, team) DO UPDATE SET tokens = tokens + excluded.tokens, requests = requests + 1",
            (today(), team, int(tokens)),
        )


def fit_to_cap(question, rows, cap):
    """Drop the lowest-ranked clauses until the prompt fits the per-request cap. Always keeps at least one."""
    kept = list(rows)
    while len(kept) > 1 and messages_tokens(rag.build_messages(question, kept)) > cap:
        kept.pop()
    return kept, len(rows) - len(kept)


# ------------------------------------------------------------------ model routing
MULTI_PART = re.compile(r"\b(compare|difference|both|unless|exception|in addition|as well as|combined|together with)\b", re.I)


def route(question, rows):
    """Plain rules, no AI. Two or more points means the large model. Returns the tier and why."""
    points, why = 0, []
    if len(question) > 120:
        points += 1
        why.append("long question")
    if question.count("?") >= 2 or len(re.findall(r"\band\b", question, re.I)) >= 2:
        points += 1
        why.append("several questions in one")
    close = [r for r in rows[:3] if r["score"] >= 0.7 * rows[0]["score"]]
    if len({r["chunk"].doc_id for r in close}) >= 3:
        points += 1
        why.append("3 different policies match about equally")
    if MULTI_PART.search(question):
        points += 1
        why.append("asks for a comparison or an exception")
    tier = "large" if points >= 2 else "small"
    small = rag.cfg("MODEL_SMALL", rag.cfg("OPENROUTER_MODEL", "openrouter/free"))
    large = rag.cfg("MODEL_LARGE", small)
    return {"tier": tier, "model": large if tier == "large" else small, "points": points, "why": why or ["simple lookup"]}


# ------------------------------------------------------------------ semantic cache
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def numbers_in(text):
    return sorted({n.replace(",", "") for n in NUMBER.findall(text)})


def key_terms(question, chunk):
    """Words in the question that the best clause itself uses (for example 'savings', 'fraud', 'video').
    Two questions that differ in one of these ('savings' vs 'current') must not share an answer."""
    return sorted(set(rag.tokens(question)) & set(rag.tokens(chunk.title + " " + chunk.text)))


def corpus_hash(index):
    """Changes whenever any policy text changes, which invalidates every saved answer."""
    h = hashlib.sha1()
    for c in sorted(index.chunks, key=lambda c: c.id):
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()


def cache_settings():
    return float(rag.cfg("CACHE_SIMILARITY", "0.92")), float(rag.cfg("CACHE_TTL_HOURS", "72"))


def cache_lookup(question, qvec, top_clause, corpus, now=None, terms=()):
    """Return (payload or None, similarity, why). A hit needs ALL of: same policy text version,
    same numbers, same best clause, same key words, and the same wording or a close enough meaning."""
    now = time.time() if now is None else now
    threshold, ttl_hours = cache_settings()
    norm, nums, terms = normalize(question), json.dumps(numbers_in(question)), json.dumps(sorted(terms))
    with db() as con:
        con.execute("DELETE FROM cache WHERE corpus != ? OR created < ?", (corpus, now - ttl_hours * 3600))
        rows = con.execute("SELECT * FROM cache").fetchall()
        hit, hit_sim = None, 0.0
        near, why = None, "nothing saved yet" if not rows else "no similar question saved"
        for row in rows:
            if row["norm"] == norm:
                hit, hit_sim = row, 1.0
                break
            if qvec is None or not row["vec"]:
                continue
            sim = rag.cosine(qvec, json.loads(row["vec"]))
            if sim < threshold:
                if near is None or sim > near:
                    near, why = sim, "meaning not close enough (%.2f)" % sim
                continue
            if row["numbers"] != nums:
                why = "the numbers differ"
            elif row["top_clause"] != top_clause:
                why = "a different policy clause matches"
            elif row["terms"] != terms:
                why = "a key word differs"
            elif sim > hit_sim:
                hit, hit_sim = row, sim
        if hit is None:
            return None, near or 0.0, why
        con.execute("UPDATE cache SET hits = hits + 1 WHERE id=?", (hit["id"],))
        return json.loads(hit["payload"]), hit_sim, "hit"


def cache_store(question, qvec, top_clause, corpus, payload, now=None, terms=()):
    now = time.time() if now is None else now
    with db() as con:
        con.execute(
            "INSERT INTO cache (norm, question, vec, top_clause, numbers, terms, corpus, payload, created) VALUES (?,?,?,?,?,?,?,?,?)",
            (normalize(question), question, json.dumps(qvec) if qvec else None, top_clause,
             json.dumps(numbers_in(question)), json.dumps(sorted(terms)), corpus, json.dumps(payload), now),
        )
        con.execute("DELETE FROM cache WHERE id NOT IN (SELECT id FROM cache ORDER BY id DESC LIMIT 500)")


# ------------------------------------------------------------------ tracing
class Gateway(rag.NoGateway):
    """One of these exists per request. rag.ask calls its hooks at each step."""

    def __init__(self, team="default"):
        self.team = clean_team(team)
        self.trace = uuid.uuid4().hex[:12]
        self.t0 = time.perf_counter()
        self.spans = []
        self.info = {"trace": self.trace, "team": self.team, "cache": "miss", "tier": "none", "model": "", "tokens": 0, "cost_inr": 0.0}

    fallback = True  # the web app shows the closest clause when the AI is unavailable; batch jobs and evals do not

    def degraded(self, err):
        self.info["error"] = str(err)[:200]
        self.info["degraded"] = True

    @contextmanager
    def span(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append({"name": name, "ms": round((time.perf_counter() - t) * 1000, 1)})

    # -- hooks used by rag.ask
    def query_vector(self, index, question):
        if not index.vectors:
            return None
        try:
            return rag.embed_query(question)
        except rag.RagError:
            return None

    def cached(self, question, rows, qvec):
        index = rag.get_index()
        top = rows[0]["chunk"]
        entry, sim, why = cache_lookup(question, qvec, top.id, corpus_hash(index), terms=key_terms(question, top))
        self.info["cache_note"] = "similarity %.2f" % sim if entry else why
        if not entry:
            return None
        self.info["cache"] = "hit"
        self.info["tier"] = "cache"
        return entry

    def plan(self, question, rows):
        daily, cap = limits()
        used = used_today(self.team)
        self.info["budget"] = {"daily": daily, "used": used, "request_cap": cap}
        if used >= daily:
            self.info["tier"] = "blocked"
            return {"refuse": "budget_exceeded", "rows": rows, "model": ""}
        decision = route(question, rows)
        kept, dropped = fit_to_cap(question, rows, cap)
        estimate = messages_tokens(rag.build_messages(question, kept))
        if used + estimate > daily:
            self.info["tier"] = "blocked"
            return {"refuse": "budget_exceeded", "rows": rows, "model": ""}
        self.info.update({"tier": decision["tier"], "model": decision["model"], "route_why": decision["why"],
                          "prompt_estimate": estimate, "clauses_dropped": dropped})
        return {"refuse": "", "rows": kept, "model": decision["model"]}

    def answer(self, question, plan):
        models = [plan["model"]]
        fallback = rag.cfg("MODEL_FALLBACK", "")
        if fallback and fallback != plan["model"]:
            models.append(fallback)
        last = None
        for model in models:
            try:
                reply = rag.generate(question, plan["rows"], model)
            except rag.RagError as e:
                last = e
                self.info["fallback_from"] = model
                continue
            if model != plan["model"]:
                self.info["model"], self.info["tier"] = model, "fallback"
            usage = reply.get("usage") or {}
            tokens = int(usage.get("total_tokens") or 0)
            if not tokens:
                tokens = messages_tokens(rag.build_messages(question, plan["rows"])) + estimate_tokens(reply.get("answer", ""))
                self.info["tokens_estimated"] = True
            self.info["tokens"] = tokens
            tier = "large" if self.info["tier"] == "large" else "small"
            self.info["cost_inr"] = round(tokens / 1000 * price_per_1k(tier), 4)
            record_usage(self.team, tokens)
            return reply
        raise last

    def stored(self, question, rows, qvec, result):
        index = rag.get_index()
        top = rows[0]["chunk"]
        cache_store(question, qvec, top.id, corpus_hash(index),
                    {"answer": result["answer"], "citations": result["citations"]}, terms=key_terms(question, top))

    def finish(self, result):
        self.info["spans"] = self.spans
        self.info["ms"] = round((time.perf_counter() - self.t0) * 1000, 1)
        result["gateway"] = self.info
        log_request(self, "degraded" if result.get("degraded") else result["status"], result["reason"])


def log_request(gw, status, reason):
    try:
        with db() as con:
            con.execute(
                "INSERT INTO requests (trace, ts, team, status, reason, cache, tier, model, tokens, cost, ms, spans) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (gw.trace, time.time(), gw.team, status, reason, gw.info.get("cache", "miss"), gw.info.get("tier", "none"),
                 gw.info.get("model", ""), int(gw.info.get("tokens", 0)), float(gw.info.get("cost_inr", 0.0)),
                 int((time.perf_counter() - gw.t0) * 1000), json.dumps(gw.spans)),
            )
    except sqlite3.Error:
        pass


def ask(question, team="default", fallback=True):
    """The one entry point the web server and the batch worker use."""
    gw = Gateway(team)
    gw.fallback = fallback
    try:
        result = rag.ask(question, gw)
    except rag.RagError as e:
        gw.info["error"] = str(e)[:200]
        log_request(gw, "error", "provider_error")
        raise
    result["trace"] = gw.trace
    return result


# ------------------------------------------------------------------ metrics
def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def metrics(hours=24):
    since = time.time() - hours * 3600
    with db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM requests WHERE ts >= ? ORDER BY id DESC", (since,))]
        usage = [dict(r) for r in con.execute("SELECT team, tokens, requests FROM usage WHERE day=?", (today(),))]
    total = len(rows)
    by = lambda key: {k: sum(1 for r in rows if r[key] == k) for k in sorted({r[key] for r in rows})}
    answered = [r for r in rows if r["status"] == "answered"]
    eligible = [r for r in rows if r["tier"] != "none"]  # requests that reached the cache step
    ms = [r["ms"] for r in rows]
    daily, cap = limits()
    return {
        "hours": hours,
        "requests": total,
        "status": by("status"),
        "reasons": {k: v for k, v in by("reason").items() if k},
        "cache_hits": sum(1 for r in rows if r["cache"] == "hit"),
        "cache_hit_rate": round(sum(1 for r in rows if r["cache"] == "hit") / len(eligible), 3) if eligible else 0,
        "answered_rate": round(len(answered) / total, 3) if total else 0,
        "latency_ms": {"p50": percentile(ms, 50), "p95": percentile(ms, 95)},
        "tokens": sum(r["tokens"] for r in rows),
        "cost_inr": round(sum(r["cost"] for r in rows), 4),
        "routes": by("tier"),
        "errors": sum(1 for r in rows if r["status"] in ("error", "degraded")),
        "budget": {"daily_per_team": daily, "request_cap": cap, "today": usage},
        "recent": [dict(r, spans=json.loads(r["spans"])) for r in rows[:20]],
    }


# ------------------------------------------------------------------ async batch jobs
MAX_JOB_QUESTIONS = 200
_wake = threading.Event()
_workers_started = False


def create_job(questions, team="default", idem=None):
    """Queue a batch. Sending the same idempotency key again returns the first job instead of a duplicate."""
    team = clean_team(team)
    qs = [str(q).strip() for q in questions if str(q).strip()]
    if not qs:
        raise ValueError("Add at least one question.")
    if len(qs) > MAX_JOB_QUESTIONS:
        raise ValueError("A batch can hold at most %d questions." % MAX_JOB_QUESTIONS)
    if any(len(q) > 500 for q in qs):
        raise ValueError("Keep every question under 500 characters.")
    idem = (idem or "").strip()[:64] or None
    job_id = uuid.uuid4().hex[:10]
    items = [{"question": q, "status": "queued"} for q in qs]
    now = time.time()
    with db() as con:
        if idem:
            row = con.execute("SELECT id FROM jobs WHERE team=? AND idem=?", (team, idem)).fetchone()
            if row:
                return row["id"], False
        con.execute("INSERT INTO jobs (id, idem, team, status, items, created, updated) VALUES (?,?,?,?,?,?,?)",
                    (job_id, idem, team, "queued", json.dumps(items), now, now))
    _wake.set()
    return job_id, True


def job_view(row):
    items = json.loads(row["items"])
    done = sum(1 for i in items if i["status"] in ("done", "failed"))
    return {
        "id": row["id"], "team": row["team"], "status": row["status"], "error": row["error"],
        "total": len(items), "finished": done, "failed": sum(1 for i in items if i["status"] == "failed"),
        "items": items, "created": row["created"],
    }


def get_job(job_id):
    with db() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return job_view(row) if row else None


def list_jobs(limit=10):
    with db() as con:
        rows = con.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for row in rows:
        v = job_view(row)
        v["items"] = []
        out.append(v)
    return out


def _save(job_id, items, status, error=""):
    with db() as con:
        con.execute("UPDATE jobs SET items=?, status=?, error=?, updated=? WHERE id=?",
                    (json.dumps(items), status, error, time.time(), job_id))


def _short(result):
    return {
        "status": result["status"], "reason": result["reason"], "answer": result["answer"],
        "cited": [c["chunk_id"] for c in result["citations"]],
        "cache": result.get("gateway", {}).get("cache", "miss"),
    }


def run_job(job_id, ask_fn=None, sleep=time.sleep):
    """Work through one job. Each question gets 3 tries with a growing wait. Three questions in a row
    failing means the provider is down or out of quota, so the job stops instead of burning more calls."""
    ask_fn = ask_fn or (lambda q, t: ask(q, t, fallback=False))  # a batch job must retry, not accept a stand-in answer
    with db() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return
    items, team = json.loads(row["items"]), row["team"]
    _save(job_id, items, "running")
    pace = float(rag.cfg("JOB_PACE_SECONDS", "3.5")) if rag.ai_enabled() else 0.0
    retry_wait = float(rag.cfg("JOB_RETRY_SECONDS", "5"))
    streak = 0
    for item in items:
        if item["status"] in ("done", "failed"):
            continue
        for attempt in range(1, 4):
            try:
                result = ask_fn(item["question"], team)
                item.update(status="done", result=_short(result))
                streak = 0
                break
            except rag.RagError as e:
                item["error"] = str(e)[:200]
                if attempt < 3:
                    sleep(retry_wait * attempt)
        else:
            item["status"] = "failed"
            streak += 1
        _save(job_id, items, "running")
        if streak >= 3:
            _save(job_id, items, "failed", "Stopped early: the AI provider failed 3 questions in a row. Check your key and daily quota, then submit again.")
            return
        if item["status"] == "done" and item["result"]["cache"] != "hit" and pace:
            sleep(pace)  # stay under the provider's per-minute limit
    ok = sum(1 for i in items if i["status"] == "done")
    _save(job_id, items, "done" if ok else "failed", "" if ok else "No question could be answered.")


def _claim():
    with db() as con:
        row = con.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        if not row:
            return None
        cur = con.execute("UPDATE jobs SET status='running' WHERE id=? AND status='queued'", (row["id"],))
        return row["id"] if cur.rowcount else None


def _worker():
    while True:
        job_id = _claim()
        if job_id:
            try:
                run_job(job_id)
            except Exception as e:  # a worker must never die
                job = get_job(job_id)
                _save(job_id, job["items"] if job else [], "failed", "Worker error: %s" % e)
        else:
            _wake.wait(timeout=2)
            _wake.clear()


def start_workers(n=None):
    """Start background workers once. Jobs that were running when the app stopped are queued again."""
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    with db() as con:
        con.execute("UPDATE jobs SET status='queued' WHERE status='running'")
    for _ in range(n or int(rag.cfg("JOB_WORKERS", "1"))):
        threading.Thread(target=_worker, daemon=True).start()
