"""Public CLI contracts and bundle operation without Git or a source checkout."""
import contextlib
import io
import json
import os
from pathlib import Path
import plistlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from voiceover_studio import benchmark, cli, runtime


class CLITests(unittest.TestCase):
    def test_bundle_identity_without_git(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'Moved App.app/Contents/Resources/app'
            (root / 'pipeline').mkdir(parents=True)
            (root / 'pipeline/example.py').write_text('print(1)')
            (root / 'build-info.json').write_text(json.dumps({'commit': 'abcd', 'dirty': False}))
            with patch.object(benchmark, 'command_output', side_effect=AssertionError('Git invoked')):
                identity = benchmark.source_id(root)
            self.assertEqual(identity['commit'], 'abcd')
            self.assertEqual(identity['status'], 'clean build')
            self.assertIn('pipeline/example.py', identity['files'])

    def test_moved_bundle_uses_own_tools_and_cleans_environment(self):
        root = Path('/tmp/An App With Spaces.app/Contents/Resources/app')
        data = Path('/tmp/Another Data Directory')
        with patch.dict(os.environ, {'PYTHONHOME': '/bad/python', 'PYTHONPATH': '/bad/modules',
                                     'VIRTUAL_ENV': '/wrong/venv', 'LOTGH_FFMPEG': '/wrong/ffmpeg'}):
            env = runtime.environment(root, data)
        self.assertEqual(env['LOTGH_FFMPEG'], str(root.parent / 'bin/ffmpeg'))
        self.assertEqual(env['PYTHONPATH'], str(root / 'pipeline'))
        self.assertEqual(env['PYTHONPYCACHEPREFIX'], str(data / 'pycache'))
        self.assertNotIn('PYTHONHOME', env)
        self.assertNotIn('VIRTUAL_ENV', env)

    def test_processing_defaults_allow_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / 'voices.json').write_text('{}')
            cmd = cli.pipeline_command(['--models=/other/models', 'a video.mkv', '-o', 'some output.mp4'], data)
            self.assertIn('a video.mkv', cmd)
            self.assertIn('some output.mp4', cmd)
            self.assertNotIn('--models', cmd)
            self.assertIn('--models=/other/models', cmd)
            self.assertEqual(cmd[0], str(data / 'venv/bin/python'))
            self.assertEqual(cmd[cmd.index('--voices') + 1], str(data / 'voices.json'))

    def test_benchmark_dispatch_passes_data_and_arguments(self):
        with patch.object(benchmark, 'main', return_value=0) as run:
            self.assertEqual(cli.main(['--data-dir', '/tmp/private data', 'benchmark', 'clone', '--lines', '320']), 0)
        run.assert_called_once_with(['clone', '--lines', '320'], default_data=Path('/tmp/private data').resolve())

    def test_doctor_incomplete_installation_returns_failure_json(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as output:
            result = cli.main(['--data-dir', d, 'doctor', '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertFalse(report['ready'])
        self.assertFalse(report['checks']['python'])

    def test_processing_uses_exec_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            (data / 'venv/bin').mkdir(parents=True)
            (data / 'venv/bin/python').touch()
            with patch.object(os, 'execve') as execute:
                cli.main(['--data-dir', d, 'process', 'a video.mkv', '--start', '2:00'])
            executable, command, env = execute.call_args.args
            self.assertEqual(executable, str(data.resolve() / 'venv/bin/python'))
            self.assertEqual(command[-3:], ['a video.mkv', '--start', '2:00'])
            self.assertEqual(env['VOICEOVER_STUDIO_DATA_DIR'], str(data.resolve()))


if __name__ == '__main__':
    unittest.main()
