import json
import shutil
import tempfile
import unittest
from pathlib import Path
from tools.kv_model_id import fingerprint

class ModelIdentity(unittest.TestCase):
    def test_paths_do_not_matter_but_weights_and_revision_do(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = root / 'a'; a.mkdir()
            (a / 'model-00001-of-00002.gguf').write_bytes(b'weights-one')
            (a / 'model-00002-of-00002.gguf').write_bytes(b'weights-two')
            (a / 'pack/tokenizer').mkdir(parents=True)
            (a / 'pack/dense.bin').write_bytes(b'dense')
            (a / 'pack/tokenizer/tokenizer.json').write_text('{"token":"A"}')
            (a / 'pack/conversions.json').write_text(json.dumps({'host': str(a)}))
            (a / 'mtp').mkdir(); (a / 'mtp/experts.bin').write_bytes(b'draft')
            b = root / 'b'; shutil.copytree(a, b)
            (b / 'pack/conversions.json').write_text(json.dumps({'host': str(b)}))
            (b / 'pack/experts.bin').write_bytes(b'derived native layout')
            def cfg(p):
                return {'cwd': str(p), 'args': ['--native', 'model-00001-of-00002.gguf',
                        '--ple-gguf', 'model-00002-of-00002.gguf', '--pack', 'pack', '--mtp', 'mtp']}
            identity = fingerprint(cfg(a), 'commit-A')
            self.assertEqual(identity, fingerprint(cfg(b), 'commit-A'))
            self.assertNotEqual(identity, fingerprint(cfg(a), 'commit-B'))
            (b / 'mtp/experts.bin').write_bytes(b'other draft')
            self.assertNotEqual(identity, fingerprint(cfg(b), 'commit-A'))
            (a / 'model-00002-of-00002.gguf').unlink()
            with self.assertRaises(ValueError):
                fingerprint(cfg(a), 'commit-A')

if __name__ == '__main__':
    unittest.main()
