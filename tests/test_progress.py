import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('voiceover_server', Path(__file__).resolve().parents[1] / 'app/server.py')
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ProgressTests(unittest.TestCase):
    def test_separation_completion_does_not_claim_ocr_is_complete(self):
        def process(item, cmd, on_line, name):
            for line in ('[00:00:01] ocr 1/2 (50.0%)', '[00:00:02] separate 1/1 (100.0%)',
                         '[00:00:03] ocr 2/2 (100.0%)', '[00:00:04] tts 1/2 (50.0%)'):
                on_line(line)
            return 1, ['intentional test exit']
        with TemporaryDirectory() as d, patch.object(server, 'P', SimpleNamespace(data=d, work=d, models=d, py='python')), \
             patch.dict(server.STATE, settings=dict(output_dir=d, duck_db=-9)), \
             patch.dict(server.CURRENT, cancel=False), patch.object(server, 'run_proc', side_effect=process), \
             patch.object(server, 'upd') as update:
            with self.assertRaises(RuntimeError):
                server.voiceover(dict(id='test', kind='file', output=d+'/output.mp4'), 'video.mkv', 0)
            values = [c.kwargs['pct'] for c in update.call_args_list if 'pct' in c.kwargs]
            self.assertEqual(values, [0, 20, 38, 58, 73])
