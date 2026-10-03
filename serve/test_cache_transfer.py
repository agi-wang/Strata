"""Binary endpoint admission and authentication without a GPU."""
import hashlib
import os
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

class TransferEngine(MockEngine):
    payload = b"snapshot-state\x00\xff"
    def transfer_cache(self, operation, path):
        if operation == "CLEAR":
            self.cleared = True
        elif operation == "EXPORT":
            Path(path).write_bytes(self.payload)
        else:
            self.imported = Path(path).read_bytes()

class CacheTransfer(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"STRATA_KV_MODEL_ID": "model-A"})
        self.env.start()
        tok = ByteTokenizer()
        self.engine = TransferEngine(tok, "ok", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc.api_key = "test-secret"
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
    def tearDown(self):
        self.httpd.shutdown(); self.httpd.server_close(); self.env.stop()
    def request(self, operation, body=b"", **headers):
        req = urllib.request.Request(self.base + "/v1/cache/" + operation, data=body,
            headers={"Authorization": "Bearer test-secret", **headers})
        return urllib.request.urlopen(req, timeout=5)
    def test_binary_roundtrip(self):
        with self.request("export") as r:
            data = r.read()
            sha, identity = r.headers["X-Strata-KV-SHA256"], r.headers["X-Strata-KV-Model"]
            self.assertGreaterEqual(float(r.headers["X-Strata-KV-Engine-Ms"]), 0)
            self.assertGreaterEqual(float(r.headers["X-Strata-KV-Hash-Ms"]), 0)
        self.assertEqual(sha, hashlib.sha256(data).hexdigest())
        with self.request("import", data, **{"X-Strata-KV-Model": identity, "X-Strata-KV-SHA256": sha}) as r:
            self.assertEqual(r.status, 200)
            import json
            timings = json.load(r)["timings"]
            self.assertGreaterEqual(timings["receive_ms"], 0)
            self.assertGreaterEqual(timings["engine_ms"], 0)
        self.assertEqual(self.engine.imported, data)
        with self.request("clear") as r:
            self.assertEqual(r.status, 200)
        self.assertTrue(self.engine.cleared)
    def test_response_releases_engine_before_file_cleanup(self):
        original_unlink = os.unlink
        cleaning, allow_cleanup, finished = threading.Event(), threading.Event(), threading.Event()
        def delayed_unlink(path, *args, **kwargs):
            if Path(path).name.startswith("strata-kv-"):
                cleaning.set()
                allow_cleanup.wait(5)
                try:
                    return original_unlink(path, *args, **kwargs)
                finally:
                    finished.set()
            return original_unlink(path, *args, **kwargs)
        with patch("serve.server.os.unlink", side_effect=delayed_unlink):
            try:
                with self.request("export") as r:
                    self.assertEqual(r.read(), self.engine.payload)
                self.assertTrue(cleaning.wait(1))
                # The previous handler is still cleaning its private file, but
                # this subsequent engine command must already be admitted.
                with self.request("clear") as r:
                    self.assertEqual(r.status, 200)
                self.assertTrue(self.engine.cleared)
            finally:
                allow_cleanup.set()
                self.assertTrue(finished.wait(1))

    def test_rejects_bad_identity_and_hash(self):
        for identity, digest in [("model-B", hashlib.sha256(b"x").hexdigest()), ("model-A", "0" * 64)]:
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.request("import", b"x", **{"X-Strata-KV-Model": identity, "X-Strata-KV-SHA256": digest})
            self.assertEqual(e.exception.code, 400)
        self.assertFalse(hasattr(self.engine, "imported"))
    def test_auth_and_busy(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request("export", Authorization="Bearer wrong")
        self.assertEqual(e.exception.code, 401)
        self.svc.fifo.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.request("export")
            self.assertEqual(e.exception.code, 409)
        finally:
            self.svc.fifo.release()
    def test_requires_opt_in_and_auth(self):
        self.svc.api_key = ""
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.request("export")
        self.assertEqual(e.exception.code, 404)

if __name__ == "__main__":
    unittest.main()
