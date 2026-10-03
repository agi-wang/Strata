"""Bounded real-GPU check of sticky forwarding and local prefix reuse.

Uses three short requests per worker and two ordinary OpenAI requests.
STRATA_API_KEY must be set; no keys or private chat logs are recorded.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import requests

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gateway', required=True)
    ap.add_argument('--model', default='qwen3.8-flash-next')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    key = os.environ.get('STRATA_API_KEY', '')
    if not key:
        raise ValueError('STRATA_API_KEY is required')
    base = args.gateway.rstrip('/')
    headers = {'Authorization': 'Bearer ' + key}
    with requests.get(base + '/v1/status', headers=headers, timeout=10) as r:
        r.raise_for_status(); before = r.json()
    assert before['routing'] == 'sticky' and before['kv_transfer'] is False, before
    workers = before['workers']
    run = uuid.uuid4().hex
    def request(messages, session=None):
        start = time.monotonic()
        with requests.post(base + '/v1/chat/completions', headers={**headers,
                **({'X-Strata-Session': session} if session else {})},
                json={'model': args.model, 'messages': messages, 'max_tokens': 32,
                      'temperature': 0, 'reasoning_effort': 'none'}, timeout=(10, 120)) as r:
            r.raise_for_status(); result = r.json()
            assert r.headers['X-Strata-Routing'] == 'sticky'
            return {'worker': r.headers['X-Strata-Worker'],
                    'wall_s': round(time.monotonic() - start, 3),
                    'answer': result['choices'][0]['message']['content'],
                    'timings': result.get('timings')}
    def messages(label):
        return [{'role': 'system', 'content': 'Verification ' + run + label + '\n'
                 + 'Always reply with only the number.\n'
                 + 'The city is quiet and the sky is blue.\n' * 80},
                {'role': 'user', 'content': 'What is 17 times 23 plus 358?'}]
    checks = []
    for index, worker in enumerate(workers):
        sessions = []
        for n in range(1000):
            session = run + '-' + str(n)
            digest = hashlib.sha256(('session:' + session).encode()).hexdigest()
            if int(digest[:16], 16) % len(workers) == index:
                sessions.append(session)
                if len(sessions) == 2:
                    break
        assert len(sessions) == 2
        history = messages('-a-' + str(index))
        cold = request(history, sessions[0])
        interleaved = request(messages('-b-' + str(index)), sessions[1])
        history += [{'role': 'assistant', 'content': '749'},
                    {'role': 'user', 'content': 'Now subtract 49. Only reply with the number.'}]
        warm = request(history, sessions[0])
        checks.append({'cold': cold, 'interleaved': interleaved, 'warm': warm})
        Path(args.out).write_text(json.dumps({'gateway': base, 'workers': checks, 'passed': False}, indent=2) + '\n')
        print(json.dumps(checks[-1]), flush=True)
        assert cold['answer'].strip() == interleaved['answer'].strip() == '749', {'cold': cold, 'interleaved': interleaved, 'warm': warm}
        assert warm['answer'].strip() == '700' and warm['timings']['cache_n'] > 900, warm
        assert cold['worker'] == interleaved['worker'] == warm['worker'] == worker
    history = messages('-automatic')
    first = request(history)
    history += [{'role': 'assistant', 'content': '749'},
                {'role': 'user', 'content': 'Now add 8. Only reply with the number.'}]
    second = request(history)
    assert first['answer'].strip() == '749' and second['answer'].strip() == '757'
    assert first['worker'] == second['worker'] and second['timings']['cache_n'] > 900
    with requests.get(base + '/v1/status', headers=headers, timeout=10) as r:
        r.raise_for_status(); after = r.json()
    assert after['kv_transfer'] is False and after['errors'] == before['errors'], after
    evidence = {'gateway': base, 'workers': checks, 'automatic': [first, second],
                'before': before, 'after': after, 'passed': True}
    Path(args.out).write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))

if __name__ == '__main__':
    main()
