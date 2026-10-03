"""OpenAI gateway: stable conversation routing to independent Strata workers.

Run: python -m serve.kv_gateway --config kv-pool.json
Authentication uses STRATA_API_KEY. X-Strata-Session identifies a conversation;
otherwise its initial messages identify the cache branch. KV stays on its worker.
"""
import argparse
import hashlib
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import requests

class Pool:
    def __init__(self, workers, key):
        if not workers or not key:
            raise ValueError("workers and STRATA_API_KEY are required")
        self.workers = [w.rstrip("/") for w in workers]
        self.key = key
        self.locks = [threading.Lock() for _ in workers]
        self.mutex = threading.Lock()
        self.next = 0
        self.stats = {"requests": 0, "errors": 0}
        self.requests_by_worker = [0 for _ in workers]
        self.admission = threading.BoundedSemaphore(64)
    def headers(self):
        return {"Authorization": "Bearer " + self.key}
    def choose(self, session=None):
        with self.mutex:
            if session:
                index = int(session[:16], 16) % len(self.workers)
            else:
                index = self.next % len(self.workers)
                self.next += 1
            self.stats["requests"] += 1
            self.requests_by_worker[index] += 1
            return index

def make_handler(pool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def authorized(self):
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if hmac.compare_digest(given.encode(), pool.key.encode()):
                return True
            self.json(401, {"error": {"message": "missing or wrong API key"}})
            return False
        def do_GET(self):
            if not self.authorized():
                return
            if self.path in ("/health", "/v1/status"):
                with pool.mutex:
                    state = {**pool.stats, "routing": "sticky", "kv_transfer": False,
                             "workers": pool.workers,
                             "requests_by_worker": dict(zip(pool.workers, pool.requests_by_worker))}
                self.json(200, state)
            elif self.path == "/v1/models":
                try:
                    with requests.get(pool.workers[0] + self.path, headers=pool.headers(), timeout=(5, 10)) as r:
                        self.json(r.status_code, r.json())
                except requests.RequestException:
                    self.json(503, {"error": {"message": "worker unavailable"}})
            else:
                self.json(404, {"error": {"message": "not found"}})
        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/chat/completions":
                self.json(404, {"error": {"message": "not found"}}); return
            if not pool.admission.acquire(blocking=False):
                self.json(429, {"error": {"message": "pool queue is full"}}); return
            worker = None
            acquired = False
            sent = False
            try:
                self.connection.settimeout(120)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("invalid request size")
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("truncated request")
                req = json.loads(body)
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                raw_session = self.headers.get("X-Strata-Session", "")
                if len(raw_session) > 256:
                    raise ValueError("session header too long")
                if raw_session:
                    session_material = "session:" + raw_session
                else:
                    initial = []
                    for message in req.get("messages", []):
                        if not isinstance(message, dict) or message.get("role") == "assistant":
                            break
                        initial.append(message)
                        if message.get("role") == "user":
                            break
                    session_material = json.dumps({"model": req.get("model"), "initial": initial,
                                                   "tools": req.get("tools")}, sort_keys=True,
                                                  ensure_ascii=False) if initial else ""
                session = hashlib.sha256(session_material.encode()).hexdigest() if session_material else None
                worker = pool.choose(session)
                acquired = pool.locks[worker].acquire(timeout=120)
                if not acquired:
                    self.json(503, {"error": {"message": "worker queue timeout"}}); return
                with requests.post(pool.workers[worker] + self.path, data=body,
                        headers={**pool.headers(), "Content-Type": "application/json"},
                        stream=True, timeout=(10, 120)) as r:
                    self.send_response(r.status_code)
                    self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                    self.send_header("Connection", "close")
                    self.send_header("X-Strata-Worker", pool.workers[worker])
                    self.send_header("X-Strata-Routing", "sticky")
                    self.end_headers(); sent = True
                    self.close_connection = True
                    if req.get("stream"):
                        for line in r.iter_lines(chunk_size=1):
                            self.wfile.write(line + b"\n"); self.wfile.flush()
                    else:
                        for block in r.iter_content(65536):
                            if block:
                                self.wfile.write(block); self.wfile.flush()
            except (ValueError, OSError, requests.RequestException) as e:
                with pool.mutex:
                    pool.stats["errors"] += 1
                if not sent:
                    self.json(502, {"error": {"message": str(e)}})
                self.close_connection = True
            finally:
                if acquired:
                    pool.locks[worker].release()
                pool.admission.release()
    return Handler

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    pool = Pool(cfg["workers"], os.environ.get("STRATA_API_KEY", ""))
    httpd = ThreadingHTTPServer((cfg.get("host", "127.0.0.1"), cfg.get("port", 18081)), make_handler(pool))
    httpd.daemon_threads = True
    print(f"Sticky gateway ready on {httpd.server_address}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
