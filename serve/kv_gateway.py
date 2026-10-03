"""OpenAI gateway: round-robin workers with real, bounded cross-host snapshots.

Run: python -m serve.kv_gateway --config kv-pool.json
Authentication uses STRATA_API_KEY. X-Strata-Session identifies a conversation;
otherwise its initial messages identify the cache branch. Snapshots are ephemeral
private files and are removed on eviction or shutdown.
"""
import argparse
import collections
import hashlib
import hmac
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import requests

MAX_SNAPSHOT = 1024 * 1024 * 1024

class Pool:
    def __init__(self, workers, key, budget=4096 * 1024 * 1024, slots=16):
        if not workers or not key:
            raise ValueError("workers and STRATA_API_KEY are required")
        self.workers = [w.rstrip("/") for w in workers]
        self.key, self.budget, self.slots = key, budget, slots
        self.locks = [threading.Lock() for _ in workers]
        self.mutex = threading.Lock()
        self.next = 0
        self.cache = collections.OrderedDict()
        self.directory = tempfile.TemporaryDirectory(prefix="strata-kv-pool-")
        self.stats = {"imports": 0, "exports": 0, "import_bytes": 0, "export_bytes": 0, "cache_errors": 0}
        self.admission = threading.BoundedSemaphore(64)
    def headers(self):
        return {"Authorization": "Bearer " + self.key}
    def choose(self):
        with self.mutex:
            index = self.next % len(self.workers)
            self.next += 1
            return index
    def import_session(self, session, worker):
        if not session:
            return False
        with self.mutex:
            item = self.cache.get(session)
            if not item or item["worker"] == worker:
                return False
            self.cache.move_to_end(session)
            f = open(item["path"], "rb")  # Open under lock; eviction cannot invalidate this fd.
        try:
            with f, requests.post(self.workers[worker] + "/v1/cache/import", data=f,
                headers={**self.headers(), "Content-Type": "application/octet-stream",
                         "X-Strata-KV-SHA256": item["sha"], "X-Strata-KV-Model": item["identity"]},
                timeout=(10, 120)) as r:
                r.raise_for_status()
            with self.mutex:
                self.stats["imports"] += 1
                self.stats["import_bytes"] += item["size"]
            return True
        except (OSError, requests.RequestException):
            with self.mutex:
                self.stats["cache_errors"] += 1
            return False  # Cache misses safely compute the full prompt.
    def export_session(self, session, worker):
        if not session:
            return
        path = None
        try:
            with requests.post(self.workers[worker] + "/v1/cache/export", data=b"",
                    headers=self.headers(), stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length", "0"))
                identity, digest = r.headers.get("X-Strata-KV-Model", ""), r.headers.get("X-Strata-KV-SHA256", "")
                if not 0 < size <= min(MAX_SNAPSHOT, self.budget) or not identity or len(digest) != 64:
                    raise ValueError("invalid snapshot response")
                sha = hashlib.sha256()
                total = 0
                with tempfile.NamedTemporaryFile(dir=self.directory.name, delete=False) as f:
                    path = f.name
                    for block in r.iter_content(1024 * 1024):
                        total += len(block)
                        if total > size:
                            raise ValueError("snapshot exceeds announced size")
                        sha.update(block); f.write(block)
                if total != size or not hmac.compare_digest(sha.hexdigest(), digest):
                    raise ValueError("snapshot length or SHA256 mismatch")
            with self.mutex:
                previous = self.cache.pop(session, None)
                if previous:
                    os.unlink(previous["path"])
                while self.cache and (len(self.cache) >= self.slots or
                        sum(x["size"] for x in self.cache.values()) + size > self.budget):
                    _, old = self.cache.popitem(last=False)
                    os.unlink(old["path"])
                self.cache[session] = {"path": path, "size": size, "identity": identity,
                                       "sha": digest, "worker": worker}
                path = None
                self.stats["exports"] += 1; self.stats["export_bytes"] += size
        except (OSError, ValueError, requests.RequestException):
            with self.mutex:
                self.stats["cache_errors"] += 1
        finally:
            if path:
                os.unlink(path)

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
                    state = {**pool.stats, "sessions": len(pool.cache),
                             "cache_bytes": sum(x["size"] for x in pool.cache.values()), "workers": pool.workers}
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
            worker = pool.choose()
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
                acquired = pool.locks[worker].acquire(timeout=120)
                if not acquired:
                    self.json(503, {"error": {"message": "worker queue timeout"}}); return
                imported = pool.import_session(session, worker)
                with requests.post(pool.workers[worker] + self.path, data=body,
                        headers={**pool.headers(), "Content-Type": "application/json"},
                        stream=True, timeout=(10, 120)) as r:
                    self.send_response(r.status_code)
                    self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                    self.send_header("Connection", "close")
                    self.send_header("X-Strata-Worker", pool.workers[worker])
                    self.send_header("X-Strata-KV-Imported", str(imported).lower())
                    self.end_headers(); sent = True
                    self.close_connection = True
                    if req.get("stream"):
                        for line in r.iter_lines(chunk_size=1):
                            self.wfile.write(line + b"\n"); self.wfile.flush()
                    else:
                        for block in r.iter_content(65536):
                            if block:
                                self.wfile.write(block); self.wfile.flush()
                    if r.status_code == 200:
                        pool.export_session(session, worker)
            except (ValueError, OSError, requests.RequestException) as e:
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
    pool = Pool(cfg["workers"], os.environ.get("STRATA_API_KEY", ""),
                cfg.get("cache_mib", 4096) * 1024 * 1024, cfg.get("cache_slots", 16))
    if pool.budget <= 0 or pool.slots <= 0:
        raise ValueError("cache budget and slots must be positive")
    httpd = ThreadingHTTPServer((cfg.get("host", "127.0.0.1"), cfg.get("port", 18081)), make_handler(pool))
    httpd.daemon_threads = True
    print(f"KV pool ready on {httpd.server_address}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close(); pool.directory.cleanup()

if __name__ == "__main__":
    main()
