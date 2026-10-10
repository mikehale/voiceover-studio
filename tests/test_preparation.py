import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from lotgh_vo import preparation, clone_worker, cli


class PreparationTests(unittest.TestCase):
    def test_defaults_and_escape_hatches(self):
        a = cli.parse_args(['video.mkv'])
        self.assertTrue(a.sep_reuse_model)
        self.assertTrue(a.overlap_preparation)
        self.assertLessEqual(a.jobs, 4)
        a = cli.parse_args(['video.mkv', '--no-sep-reuse-model', '--no-overlap-preparation'])
        self.assertFalse(a.sep_reuse_model)
        self.assertFalse(a.overlap_preparation)

    def test_overlap_skips_cpu_srt_partial_ocr_and_low_headroom(self):
        a = cli.parse_args(['video.mkv'])
        torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
        with patch.dict(sys.modules, {'torch': torch}), patch.object(preparation.subprocess, 'check_output', return_value='System-wide memory free percentage: 80%') as pressure:
            self.assertTrue(preparation.can_overlap(a))
            pressure.return_value = 'System-wide memory free percentage: 59%'
            self.assertFalse(preparation.can_overlap(a))
            for name, value in [('device', 'cpu'), ('srt', 'provided.srt'), ('until', 'ocr')]:
                old = getattr(a, name); setattr(a, name, value)
                pressure.return_value = 'System-wide memory free percentage: 80%'
                self.assertFalse(preparation.can_overlap(a))
                setattr(a, name, old)

    def test_cancellation_reaps_worker_and_its_tool(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = Path(d) / 'child.pid'
            script = """import sys,subprocess
from lotgh_vo import preparation,separate
def fake(*args):
    subprocess.run([sys.executable,'-c',%r],check=True)
separate.run_separation=fake
preparation.worker(sys.argv[1])
""" % (f"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)")
            real = subprocess.Popen
            def launch(cmd, **kwargs):
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'pipeline'))
                return real([sys.executable, '-c', script, cmd[-1]], env=env, **kwargs)
            a = cli.parse_args(['video.mkv'])
            with self.assertRaisesRegex(RuntimeError, 'OCR failed'):
                with preparation.SeparationTask() as task, patch.object(preparation.subprocess, 'Popen', side_effect=launch):
                    task.start(a, 'video.mkv', {}, d)
                    deadline = time.monotonic() + 10
                    while not pidfile.exists() and time.monotonic() < deadline: time.sleep(.02)
                    self.assertTrue(pidfile.exists())
                    raise RuntimeError('OCR failed')
            self.assertIsNotNone(task.proc.poll())
            with self.assertRaises(ProcessLookupError): os.kill(int(pidfile.read_text()), 0)
            self.assertFalse(Path(task.temp.name).exists())

    def test_cfm_cache_changes_without_changing_seed(self):
        job = dict(version=2, seed=1234, exaggeration=.5, cfg=.5)
        import hashlib
        original = hashlib.sha1('2|Hello|ref|1234|0.5|0.5|german_v3'.encode()).hexdigest()
        seed = clone_worker.gen_seed(job, 'Hello', 'ref', 'german_v3')
        self.assertEqual(seed, (1234 + int(original[:8], 16)) % 2**31)
        six = clone_worker.gen_key(job, 'Hello', 'ref', 'german_v3')
        self.assertNotEqual(six, original[:16])
        with patch.object(clone_worker, 'CFM_STEPS', 10):
            self.assertNotEqual(six, clone_worker.gen_key(job, 'Hello', 'ref', 'german_v3'))
            self.assertEqual(seed, clone_worker.gen_seed(job, 'Hello', 'ref', 'german_v3'))
