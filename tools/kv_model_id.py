"""Compute a common KV identity from exact artifacts, independent of host paths.

python tools/kv_model_id.py --config strata-service.json --revision <fork-commit>
Hashes all GGUF shards, packed dense/tokenizer files, and the draft pack. Run once
per deployment; all peers must use the same revision and steering settings.
"""
import argparse
import hashlib
import json
from pathlib import Path

def fingerprint(config, revision):
    args = config['args']
    cwd = Path(config.get('cwd', '.'))
    def value(flag):
        return args[args.index(flag) + 1] if flag in args else None
    def path(name):
        p = Path(name)
        return p if p.is_absolute() else cwd / p
    artifacts = {}
    def add(role, p):
        if not p.is_file():
            raise ValueError(f'missing artifact: {p}')
        h = hashlib.sha256()
        with p.open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                h.update(block)
        artifacts[role] = h.hexdigest()
    native = value('--native')
    if native:
        first = path(native)
        import re
        match = re.fullmatch(r'(.*)-\d{5}-of-(\d{5})\.gguf', first.name)
        if match:
            count = int(match[2])
            for n in range(1, count + 1):
                add(f'gguf/{n}', first.with_name(f'{match[1]}-{n:05d}-of-{count:05d}.gguf'))
        else:
            add('gguf/1', first)
    for flag in ('--pack', '--mtp'):
        if value(flag):
            root = path(value(flag))
            for p in sorted(root.rglob('*')):
                if (not p.is_file() or p.name.endswith('.src.json') or
                        (flag == '--pack' and p.parent == root and p.suffix == '.json')):
                    continue
                # Native experts.bin is a verified rearrangement of the GGUF.
                if flag == '--pack' and native and p.name == 'experts.bin':
                    continue
                add(flag + '/' + p.relative_to(root).as_posix(), p)
    if value('--ple-gguf'):
        add('ple', path(value('--ple-gguf')))
    steering = {}
    for flag in ('--control-vector', '--control-vector-scaled', '--cvec-mode', '--cvec-dir'):
        if value(flag):
            steering[flag] = value(flag)
    if '--control-vector-layer-range' in args:
        at = args.index('--control-vector-layer-range')
        steering['layers'] = args[at + 1:at + 3]
    for flag in ('--control-vector', '--control-vector-scaled'):
        if value(flag):
            files = value(flag).split(',') if flag.endswith('scaled') else [value(flag)]
            for i, item in enumerate(files):
                name, scale = item.rsplit(':', 1) if flag.endswith('scaled') else (item, '1')
                add(f'steering/{flag}/{i}/{scale}', path(name))
                # Hash content and scale, never a machine-specific path.
            steering.pop(flag)
    data = {'wire': 1, 'revision': revision, 'artifacts': artifacts, 'steering': steering}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--revision', required=True)
    args = ap.parse_args()
    print(fingerprint(json.loads(Path(args.config).read_text()), args.revision))

if __name__ == '__main__':
    main()
