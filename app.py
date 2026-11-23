"""app.py - tiny local web server for the Policy Desk. Run:  python3 app.py"""
import json
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import gateway
import rag

UI_FILE = Path(__file__).parent / "ui" / "index.html"
OPS_FILE = Path(__file__).parent / "ui" / "ops.html"
PORT = int(rag.cfg("PORT", "8000"))


class RateLimiter:
    """Allow at most `limit` calls per `window` seconds for each client address."""

    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self.calls = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, who, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.calls[who]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


ASK_LIMIT = RateLimiter(int(rag.cfg("RATE_LIMIT_PER_MIN", "10")), 60)
JOB_LIMIT = RateLimiter(int(rag.cfg("JOB_LIMIT_PER_MIN", "5")), 60)


def doc_summary(doc):
    return {
        "code": doc["code"],
        "title": doc["title"],
        "category": doc.get("category", ""),
        "version": doc.get("version", ""),
        "effective": doc.get("effective", ""),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json", headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        index = rag.get_index()
        if path in ("/", "/index.html"):
            return self._send(200, UI_FILE.read_bytes(), "text/html")
        if path in ("/ops", "/ops.html"):
            return self._send(200, OPS_FILE.read_bytes(), "text/html")
        if path == "/api/metrics":
            return self._send(200, gateway.metrics())
        if path == "/api/jobs":
            return self._send(200, gateway.list_jobs())
        if path.startswith("/api/jobs/"):
            job = gateway.get_job(path[len("/api/jobs/"):])
            return self._send(200, job) if job else self._send(404, {"error": "Job not found."})
        if path == "/api/health":
            return self._send(200, {
                "mode": "ai" if rag.ai_enabled() else "search-only",
                "model": rag.cfg("OPENROUTER_MODEL", "openrouter/free"),
                "semantic": index.vectors is not None,
                "docs": len(index.docs),
                "warning": index.warning,
                "cache": True,
                "retrieval": rag.cfg("RETRIEVAL", "weighted"),
                "daily_token_budget": gateway.limits()[0],
            })
        if path == "/api/docs":
            return self._send(200, [doc_summary(d) for d in index.docs.values()])
        if path.startswith("/api/docs/"):
            doc = index.docs.get(path[len("/api/docs/"):])
            if not doc:
                return self._send(404, {"error": "Policy not found."})
            body = doc_summary(doc)
            body["sections"] = [{"section": s, "text": t} for s, t in doc["sections"]]
            return self._send(200, body)
        self._send(404, {"error": "Not found."})

    def _json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 100_000:
            raise ValueError("Request too large.")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object.")
        return data

    def do_POST(self):
        if self.path == "/api/jobs":
            if not JOB_LIMIT.allow(self.client_address[0]):
                return self._send(429, {"error": "Too many batches. Wait a minute and try again."})
            try:
                data = self._json_body()
                questions = data.get("questions")
                if not isinstance(questions, list):
                    raise ValueError("Send {\"questions\": [\"...\", \"...\"]}.")
                team = self.headers.get("X-Team", "default")
                key = self.headers.get("Idempotency-Key") or data.get("key")
                job_id, created = gateway.create_job(questions, team, key)
            except (ValueError, TypeError) as e:
                return self._send(400, {"error": str(e)})
            return self._send(202 if created else 200, {"job_id": job_id, "created": created})
        if self.path == "/api/escalate":
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                question = str(data.get("question", "")).strip()
            except (ValueError, TypeError, AttributeError):
                return self._send(400, {"error": "Send JSON like {\"question\": \"...\"}."})
            if not question:
                return self._send(400, {"error": "There is no question to escalate."})
            return self._send(200, {"ticket": rag.escalate(question, data.get("reason", ""))})
        if self.path != "/api/ask":
            return self._send(404, {"error": "Not found."})
        if not ASK_LIMIT.allow(self.client_address[0]):
            return self._send(429, {"error": "Too many questions. Wait a minute and try again."})
        try:
            length = int(self.headers.get("Content-Length", 0))
            question = json.loads(self.rfile.read(length)).get("question", "")
        except (ValueError, TypeError):
            return self._send(400, {"error": "Send JSON like {\"question\": \"...\"}."})
        question = str(question).strip()
        if not question:
            return self._send(400, {"error": "Type a question first."})
        if len(question) > 500:
            return self._send(400, {"error": "Keep the question under 500 characters."})
        try:
            result = gateway.ask(question, self.headers.get("X-Team", "default"))
            self._send(200, result, headers={"X-Trace-Id": result["trace"]})
        except rag.QuotaError:
            self._send(503, {"error": "The AI service has reached its daily limit. Please try again tomorrow."})
        except rag.RagError:
            self._send(502, {"error": "The AI service could not be reached. Please try again in a moment."})

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet; questions are logged (masked) in logs/queries.jsonl


def main():
    print("Loading policies...")
    index = rag.get_index()
    mode = "AI answers via OpenRouter (%s)" % rag.cfg("OPENROUTER_MODEL", "openrouter/free") if rag.ai_enabled() else "search-only (no OPENROUTER_API_KEY found)"
    print("%d policies, %d clauses. Mode: %s" % (len(index.docs), len(index.chunks), mode))
    if index.warning:
        print("Warning:", index.warning)
    gateway.start_workers()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("Open http://localhost:%d in your browser (operations dashboard: /ops). Press Ctrl+C to stop." % PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
