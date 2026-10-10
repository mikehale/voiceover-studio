"""Benchmark integrity and process cleanup tests; no models or downloads."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from voiceover_studio import benchmark as b


class BenchmarkTests(unittest.TestCase):
    def test_pipeline_metrics_survive_failure_and_partial_success(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
        from lotgh_vo import cli
        def successful(args, metrics):
            metrics['stages_s']['ocr'] = 1.25
        def failed(args, metrics):
            successful(args, metrics)
            raise RuntimeError('test separation failure')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'pipeline.json'
            args = ['unused.mkv', '--metrics-json', str(path)]
            with patch.object(cli, '_main', side_effect=successful):
                cli.main(args)
            report = json.loads(path.read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['stages_s'], {'ocr': 1.25})
            with patch.object(cli, '_main', side_effect=failed):
                with self.assertRaisesRegex(RuntimeError, 'separation failure'):
                    cli.main(args)
            report = json.loads(path.read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['stages_s'], {'ocr': 1.25})

    def test_swap_units(self):
        self.assertEqual(b.swap_bytes('total = 4G used = 3072.25M free = 1G'), int(3072.25 * 1024**2))
        with self.assertRaises(ValueError):
            b.swap_bytes('permission denied')

    def test_guard_and_missing_telemetry(self):
        limits = SimpleNamespace(max_swap_gib=8, max_rss_gib=24, min_free_percent=15)
        sample = dict(swap_bytes=0, tree_rss_bytes=10 * b.GIB, free_percent=30)
        self.assertIsNone(b.guard_reason(sample, limits))
        sample['free_percent'] = 14
        self.assertIn('free-memory', b.guard_reason(sample, limits))
        sample.update(free_percent=30, swap_bytes=9 * b.GIB)
        self.assertIn('swap', b.guard_reason(sample, limits))

    def test_comparison_rejects_changed_samples_and_aborted_runs(self):
        with tempfile.TemporaryDirectory() as d:
            a, c = Path(d) / 'a', Path(d) / 'c'
            a.mkdir(); c.mkdir()
            for p, key in ((a, 'one'), (c, 'two')):
                b.write_json(p / 'manifest.json', {'workload_sha256': key})
                b.write_json(p / 'summary.json', {'status': 'completed', 'cached_lines': 0})
            with self.assertRaisesRegex(ValueError, 'Workloads differ'):
                b.compare(a, c)
            b.write_json(c / 'manifest.json', {'workload_sha256': 'one'})
            b.write_json(c / 'summary.json', {'status': 'aborted'})
            with self.assertRaisesRegex(ValueError, 'must complete'):
                b.compare(a, c)

    def test_existing_directory_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                b.main(['clone', '--job', 'unused', '--run-dir', d])

    def test_worker_restarts_do_not_count_as_warm_cache(self):
        events = [dict(ev='gen_start', cached=0), dict(ev='synth', seconds=2, audio_s=1),
                  dict(ev='line', line=1), dict(ev='recycle'), dict(ev='gen_start', cached=1),
                  dict(ev='synth', seconds=3, audio_s=2), dict(ev='line', line=2),
                  dict(ev='result', results={'1': {'path': 'a', 'cached': True}, '2': {'path': 'b'}})]
        script = '\n'.join('print(' + repr('BENCH ' + json.dumps(e)) + ', flush=True)' for e in events)
        limits = SimpleNamespace(max_swap_gib=8, max_rss_gib=24, min_free_percent=15,
                                 timeout_minutes=1, interval=.01)
        sample = dict(swap_bytes=0, tree_rss_bytes=0, free_percent=50, processes=[])
        with tempfile.TemporaryDirectory() as d, patch.object(b, 'memory_sample', return_value=sample.copy()):
            summary = b.supervise([sys.executable, '-c', script], dict(b.os.environ), Path(d), limits)
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['generated_lines'], 2)
            self.assertEqual(summary['cached_lines'], 0)
            self.assertEqual(summary['worker_restarts'], 1)
            self.assertAlmostEqual(summary['synth_rtf_including_retries'], 5 / 3)

    def test_guard_terminates_process_group_and_writes_summary(self):
        limits = SimpleNamespace(max_swap_gib=8, max_rss_gib=24, min_free_percent=15,
                                 timeout_minutes=1, interval=.01)
        samples = [dict(swap_bytes=0, tree_rss_bytes=0, free_percent=50, processes=[]),
                   dict(swap_bytes=9*b.GIB, tree_rss_bytes=0, free_percent=50, processes=[])]
        with tempfile.TemporaryDirectory() as d, patch.object(b, 'memory_sample', side_effect=samples):
            summary = b.supervise([sys.executable, '-c', 'import time; time.sleep(60)'],
                                  dict(b.os.environ), Path(d), limits)
            self.assertEqual(summary['status'], 'aborted')
            self.assertIn('swap', summary['reason'])
            self.assertTrue((Path(d) / 'summary.json').exists())
            self.assertIsNotNone(summary['exit_code'])


if __name__ == '__main__':
    unittest.main()
