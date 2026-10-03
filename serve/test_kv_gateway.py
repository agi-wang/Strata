"""Real HTTP transport and bounded storage, with mock inference workers."""
import os
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import requests
from serve.frontend import ChatTemplate
from serve.kv_gateway import Pool, make_handler
from serve.server import ByteTokenizer, Service, serve
from serve.test_cache_transfer import TransferEngine

class Gateway(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'STRATA_KV_MODEL_ID': 'model-A'})
        self.env.start()
        self.engines, self.workers = [], []
        for _ in range(2):
            tok = ByteTokenizer()
            engine = TransferEngine(tok, '</think>\n\nok', max_context=4096)
            svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            svc.api_key = 'secret'
            self.engines.append(engine); self.workers.append(serve(svc, port=0))
        urls = [f'http://127.0.0.1:{s.server_address[1]}' for s in self.workers]
        self.pool = Pool(urls, 'secret', budget=1024, slots=1)
        self.gateway = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.pool))
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.gateway.server_address[1]}'
    def tearDown(self):
        self.gateway.shutdown(); self.gateway.server_close()
        for s in self.workers:
            s.shutdown(); s.server_close()
        self.pool.directory.cleanup(); self.env.stop()
    def turn(self, session, stream=False):
        return requests.post(self.url + '/v1/chat/completions', timeout=10,
            headers={'Authorization': 'Bearer secret', 'X-Strata-Session': session},
            json={'model': 'test', 'max_tokens': 8, 'stream': stream,
                  'messages': [{'role': 'user', 'content': 'hello'}]})
    def test_round_robin_migrates_binary_state(self):
        with self.turn('same') as a:
            self.assertEqual(a.status_code, 200)
            self.assertEqual(a.headers['X-Strata-KV-Imported'], 'false')
            first = a.headers['X-Strata-Worker']
        with self.turn('same', stream=True) as b:
            self.assertEqual(b.status_code, 200)
            self.assertEqual(b.headers['X-Strata-KV-Imported'], 'true')
            self.assertNotEqual(first, b.headers['X-Strata-Worker'])
            self.assertIn('data: [DONE]', b.text)
        self.assertEqual(self.engines[1].imported, TransferEngine.payload)
        self.assertEqual(self.pool.stats['imports'], 1)
        self.assertEqual(self.pool.stats['exports'], 2)
    def test_ordinary_openai_requests_migrate_without_session_header(self):
        with self.turn('') as r:
            self.assertEqual(r.status_code, 200)
        with self.turn('') as r:
            self.assertEqual(r.headers['X-Strata-KV-Imported'], 'true')
        self.assertEqual(self.engines[1].imported, TransferEngine.payload)

    def test_eviction_and_auth(self):
        with self.turn('one') as r:
            self.assertEqual(r.status_code, 200)
        with self.turn('two') as r:
            self.assertEqual(r.status_code, 200)
        self.assertEqual(len(self.pool.cache), 1)
        self.assertEqual(len(list(Path(self.pool.directory.name).iterdir())), 1)
        with requests.get(self.url + '/v1/status', timeout=5) as r:
            self.assertEqual(r.status_code, 401)

if __name__ == '__main__':
    unittest.main()
