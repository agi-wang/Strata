"""Bounded real-GPU check: identical cold/warm output and imported prefix reuse.

STRATA_API_KEY=... python tools/test_cross_host_kv.py --source http://worker-a:18080
    --target http://worker-b:18080 --out evidence.json
Two cold prompts and one imported prompt; no continuous load or chat logs.
"""
import argparse
import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
import requests

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--model', default='qwen3.8-flash-next')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    key = os.environ.get('STRATA_API_KEY', '')
    if not key:
        raise ValueError('STRATA_API_KEY is required')
    headers = {'Authorization': 'Bearer ' + key}
    req = {'model': args.model, 'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': 32,
           'messages': [{'role': 'system', 'content': 'Verification ' + uuid.uuid4().hex + '\n'
                         + 'Always reply with only the number.\n'
                         + 'The city is quiet and the sky is blue.\n' * 80},
                        {'role': 'user', 'content': 'What is 17 times 23 plus 358?'}]}
    def generate(base):
        start = time.monotonic()
        with requests.post(base.rstrip('/') + '/v1/chat/completions', json=req, headers=headers,
                           timeout=(10, 120)) as r:
            r.raise_for_status(); result = r.json()
        return {'wall_s': round(time.monotonic() - start, 3), 'timings': result.get('timings'),
                'answer': result['choices'][0]['message'].get('content')}
    cold = generate(args.target)
    source = generate(args.source)
    with tempfile.TemporaryFile() as snapshot:
        start = time.monotonic()
        with requests.post(args.source.rstrip('/') + '/v1/cache/export', data=b'', headers=headers,
                           stream=True, timeout=(10, 120)) as r:
            r.raise_for_status()
            header_s = time.monotonic() - start
            download_start = time.monotonic()
            source_engine_ms = float(r.headers.get('X-Strata-KV-Engine-Ms', 0))
            source_hash_ms = float(r.headers.get('X-Strata-KV-Hash-Ms', 0))
            sha = hashlib.sha256(); size = 0
            identity, digest = r.headers['X-Strata-KV-Model'], r.headers['X-Strata-KV-SHA256']
            for block in r.iter_content(1024 * 1024):
                size += len(block)
                if size > 1024 * 1024 * 1024:
                    raise ValueError('snapshot exceeds 1 GiB')
                sha.update(block); snapshot.write(block)
            assert sha.hexdigest() == digest
            download_s = time.monotonic() - download_start
        export_s = time.monotonic() - start
        # Remove ALL target state, including parked prefixes from its cold run.
        # The imported snapshot must be the sole source of the subsequent cache hit.
        with requests.post(args.target.rstrip('/') + '/v1/cache/clear', data=b'', headers=headers,
                           timeout=(10, 120)) as r:
            r.raise_for_status()
        snapshot.seek(0)
        start = time.monotonic()
        with requests.post(args.target.rstrip('/') + '/v1/cache/import', data=snapshot,
                           headers={**headers, 'X-Strata-KV-Model': identity, 'X-Strata-KV-SHA256': digest},
                           timeout=(10, 120)) as r:
            r.raise_for_status()
            target_timings = r.json().get('timings', {})
        import_s = time.monotonic() - start
    warm = generate(args.target)
    evidence = {'source': args.source, 'target': args.target, 'snapshot_bytes': size,
                'model_identity': identity, 'export_s': round(export_s, 3), 'import_s': round(import_s, 3),
                'transfer': {'export_engine_ms': source_engine_ms, 'export_hash_ms': source_hash_ms,
                             'export_header_s': round(header_s, 3), 'download_s': round(download_s, 3),
                             'download_MB_s': round(size / max(download_s, 1e-9) / 1e6, 1),
                             'receive_ms': target_timings.get('receive_ms'),
                             'import_engine_ms': target_timings.get('engine_ms'),
                             'export_plus_import_s': round(export_s + import_s, 3)},
                'cold': cold, 'source_result': source, 'warm': warm}
    assert cold['answer'].strip() == source['answer'].strip() == warm['answer'].strip() == '749', evidence
    assert warm['timings']['cache_n'] > 512 and warm['timings']['prompt_n'] < cold['timings']['prompt_n'], evidence
    evidence['passed'] = True
    Path(args.out).write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))

if __name__ == '__main__':
    main()
