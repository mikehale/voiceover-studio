"""Model reuse must preserve stem output and release GPU state on exit/failure."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from lotgh_vo import separate


class SeparationTests(unittest.TestCase):
    def test_session_reuses_model_and_preserves_stem_sum(self):
        api = ModuleType('demucs.api')
        model = Mock(samplerate=44100)
        model.separate_audio_file.side_effect = lambda path: (None, {
            'vocals': np.array([.1, .2]), 'drums': np.array([.2, .3]),
            'bass': np.array([.3, .4]), 'other': np.array([.1, .1])})
        api.Separator = Mock(return_value=model)
        api.save_audio = Mock()
        torch = SimpleNamespace(zeros_like=np.zeros_like, set_num_threads=Mock(),
                                mps=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()))
        with patch.dict(sys.modules, {'demucs.api': api, 'torch': torch}), TemporaryDirectory() as d:
            session = separate.DemucsSession()
            session('one.wav', d, 'mps', 2)
            session('two.wav', d, 'mps', 2)
            self.assertEqual(api.Separator.call_count, 1)
            self.assertEqual(model.separate_audio_file.call_count, 2)
            np.testing.assert_allclose(api.save_audio.call_args_list[1].args[0], [.6, .8])
            self.assertEqual(api.save_audio.call_args.kwargs['bits_per_sample'], 16)
            session.close()
            self.assertIsNone(session.separator)
            torch.mps.synchronize.assert_called_once()
            torch.mps.empty_cache.assert_called_once()

    def test_stage_failure_closes_session(self):
        session = Mock()
        with patch.object(separate, 'DemucsSession', return_value=session), \
             patch.object(separate, '_run_separation', side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                separate.run_separation(SimpleNamespace(sep_reuse_model=True), 'video', {}, 'work')
            session.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
