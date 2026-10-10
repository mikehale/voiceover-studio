"""Overlap CPU OCR with an owned separation process; join before speech synthesis."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def can_overlap(a):
    if not a.overlap_preparation or a.srt or a.until == 'ocr' or a.device == 'cpu' or a.onnx_provider != 'cpu' or a.jobs > 4:
        return False
    try:
        import torch
        if not torch.backends.mps.is_available():
            return False
        result = subprocess.check_output(['/usr/bin/memory_pressure', '-Q'], text=True, timeout=3)
        match = re.search(r'free percentage:\s*(\d+)%', result)
        return bool(match and int(match[1]) >= 60)
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
        return False


class SeparationTask:
    def __init__(self):
        self.proc = None
        self.temp = None

    def __enter__(self):
        return self

    def start(self, a, video, info, wd):
        self.temp = tempfile.TemporaryDirectory(prefix='preparation-', dir=wd)
        self.result = Path(self.temp.name) / 'result.json'
        request = Path(self.temp.name) / 'request.json'
        args = dict(sep_chunk=a.sep_chunk, device=a.device, jobs=a.jobs,
                    force=list(a.force), sep_reuse_model=a.sep_reuse_model)
        request.write_text(json.dumps(dict(args=args, video=video, info=info, wd=wd, result=str(self.result))))
        # Keep the app/benchmark process group, so its group cancellation and memory accounting include us.
        self.proc = subprocess.Popen([sys.executable, '-u', '-m', 'lotgh_vo.preparation', str(request)])

    def finish(self):
        code = self.proc.wait()
        if code:
            raise RuntimeError(f'Background separation failed (exit {code}); see separation log')
        result = json.loads(self.result.read_text())
        return result['chunks'], result['seconds']

    def __exit__(self, *exc):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.temp is not None:
            self.temp.cleanup()


def worker(path):
    # SystemExit lets subprocess.run kill/reap a currently running FFmpeg or Demucs child.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    from .separate import run_separation
    from .util import atomic_json
    request = json.loads(Path(path).read_text())
    started = time.monotonic()
    chunks = run_separation(SimpleNamespace(**request['args']), request['video'], request['info'], request['wd'])
    atomic_json(request['result'], dict(chunks=chunks, seconds=time.monotonic() - started))


if __name__ == '__main__':
    worker(sys.argv[1])
