"""Sticky routing and HTTP/SSE forwarding with independent mock workers."""
import hashlib
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
        self.engines, self.workers = [], []
        for _ in range(3):
            tok = ByteTokenizer()
            engine = TransferEngine(tok, '</think>\n\nok', max_context=4096)
            svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            svc.api_key = 'secret'
            self.engines.append(engine); self.workers.append(serve(svc, port=0))
        urls = [f'http://127.0.0.1:{s.server_address[1]}' for s in self.workers]
        self.pool = Pool(urls, 'secret')
        self.gateway = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.pool))
        threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.gateway.server_address[1]}'
    def tearDown(self):
        self.gateway.shutdown(); self.gateway.server_close()
        for s in self.workers:
            s.shutdown(); s.server_close()
    def turn(self, session, stream=False, messages=None):
        return requests.post(self.url + '/v1/chat/completions', timeout=10,
            headers={'Authorization': 'Bearer secret', 'X-Strata-Session': session},
            json={'model': 'test', 'max_tokens': 8, 'stream': stream,
                  'messages': messages or [{'role': 'user', 'content': 'hello'}]})
    def test_same_session_stays_on_worker_without_transfer(self):
        with patch.object(TransferEngine, 'transfer_cache', side_effect=AssertionError('KV must stay local')):
            with self.turn('same') as a:
                self.assertEqual(a.status_code, 200)
                first = a.headers['X-Strata-Worker']
                self.assertEqual(a.headers['X-Strata-Routing'], 'sticky')
                self.assertNotIn('X-Strata-KV-Imported', a.headers)
            with self.turn('same', stream=True) as b:
                self.assertEqual(b.status_code, 200)
                self.assertEqual(first, b.headers['X-Strata-Worker'])
                self.assertIn('data: [DONE]', b.text)
        self.assertEqual(self.pool.stats['errors'], 0)
    def test_ordinary_openai_requests_keep_affinity_across_turns(self):
        messages = [{'role': 'system', 'content': 'Be helpful.'}, {'role': 'user', 'content': 'hello'}]
        with self.turn('', messages=messages) as r:
            self.assertEqual(r.status_code, 200)
            first = r.headers['X-Strata-Worker']
        messages += [{'role': 'assistant', 'content': 'ok'}, {'role': 'user', 'content': 'continue'}]
        with self.turn('', messages=messages) as r:
            self.assertEqual(r.status_code, 200)
            self.assertEqual(first, r.headers['X-Strata-Worker'])
    def test_sessions_cover_workers_and_survive_gateway_restart(self):
        restarted = Pool(self.pool.workers, 'secret')
        selected = set()
        for i in range(30):
            session = hashlib.sha256(('session:' + str(i)).encode()).hexdigest()
            worker = self.pool.choose(session)
            self.assertEqual(worker, restarted.choose(session))
            selected.add(worker)
        self.assertEqual(selected, {0, 1, 2})
    def test_unidentified_requests_round_robin(self):
        self.assertEqual([self.pool.choose() for _ in range(6)], [0, 1, 2, 0, 1, 2])
    def test_auth_and_status(self):
        with requests.get(self.url + '/v1/status', timeout=5) as r:
            self.assertEqual(r.status_code, 401)
        with self.turn('one') as r:
            self.assertEqual(r.status_code, 200)
            worker = r.headers['X-Strata-Worker']
        with requests.get(self.url + '/v1/status', timeout=5,
                headers={'Authorization': 'Bearer secret'}) as r:
            status = r.json()
        self.assertEqual(status['routing'], 'sticky')
        self.assertFalse(status['kv_transfer'])
        self.assertEqual(status['requests'], 1)
        self.assertEqual(status['requests_by_worker'][worker], 1)
        self.assertNotIn('cache_bytes', status)

if __name__ == '__main__':
    unittest.main()
