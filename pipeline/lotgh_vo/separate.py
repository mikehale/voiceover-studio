"""Demucs htdemucs two-stem separation in padded time chunks (cached per chunk; MPS when available)."""
import os, shutil, subprocess, sys
import numpy as np, soundfile as sf
from .util import FFMPEG, log, Progress, run

SR = 44100

def pick_device(pref):
    if pref in ('cpu', 'mps'): return pref
    try:
        import torch
        if torch.backends.mps.is_available(): return 'mps'
    except Exception: pass
    return 'cpu'

def chunk_paths(wd, k, L):
    d = os.path.join(wd, 'stems', f'len{int(L)}_chunk_{k:05d}')       # chunk length in the name: safe cache key
    return d, os.path.join(d, 'vocals.flac'), os.path.join(d, 'no_vocals.flac')   # 24-bit FLAC keeps disk use ~1/3

def _demucs(src, outdir, device, threads):
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1')
    if device == 'cpu' and threads: env['OMP_NUM_THREADS'] = str(threads)
    cmd = [sys.executable, '-m', 'demucs', '-n', 'htdemucs', '--two-stems=vocals', '-d', device, '-o', outdir, src]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0: raise RuntimeError(r.stderr[-1500:])

class DemucsSession:
    """One model per separation stage, with the same settings and WAV encoding as the CLI."""
    def __init__(self):
        self.separator = None
        self.device = None

    def __call__(self, src, outdir, device, threads):
        from pathlib import Path
        import torch
        from demucs.api import Separator, save_audio
        if self.device != device:
            self.close()
        if self.separator is None:
            if device == 'cpu' and threads:
                torch.set_num_threads(threads)
            self.separator = Separator(model='htdemucs', device=device, shifts=1,
                                       overlap=0.25, progress=False)
            self.device = device
        _, stems = self.separator.separate_audio_file(Path(src))
        folder = Path(outdir) / 'htdemucs' / 'src'
        folder.mkdir(parents=True, exist_ok=True)
        vocals = stems.pop('vocals')
        other = torch.zeros_like(next(iter(stems.values())))
        for stem in stems.values():
            other += stem
        for name, audio in (('vocals', vocals), ('no_vocals', other)):
            save_audio(audio, str(folder / (name + '.wav')), samplerate=self.separator.samplerate,
                       clip='rescale', as_float=False, bits_per_sample=16)

    def close(self):
        device = self.device
        self.separator = None
        self.device = None
        if device is not None:
            import gc
            gc.collect()
            if device == 'mps':
                import torch
                torch.mps.synchronize()
                torch.mps.empty_cache()


def run_separation(a, video, info, wd):
    session = DemucsSession() if getattr(a, 'sep_reuse_model', False) else None
    try:
        return _run_separation(a, video, info, wd, session or _demucs)
    finally:
        if session is not None:
            session.close()


def _run_separation(a, video, info, wd, demucs):
    """Returns list of (t0, t1, vocals_path, no_vocals_path)."""
    dur, L, pad = info['duration'], a.sep_chunk, 5.0
    n = int(np.ceil(dur / L)); device = pick_device(a.device)
    todo = [k for k in range(n) if not all(os.path.exists(p) for p in chunk_paths(wd, k, L)[1:]) or 'force_separate' in a.force]
    log('separate', f'Demucs htdemucs on {device.upper()}: {n} chunks of {L}s (+{pad:.0f}s overlap), '
                    f'{n - len(todo)} cached, {len(todo)} to do')
    pr = Progress('separate', len(todo), every=0)
    for k in todo:
        d, vp, bp = chunk_paths(wd, k, L); os.makedirs(d, exist_ok=True)
        t0, t1 = k * L, min(dur, (k + 1) * L)
        a0, a1 = max(0.0, t0 - pad), min(dur, t1 + pad)
        src = os.path.join(d, 'src.wav')
        run([FFMPEG, '-v', 'error', '-y', '-ss', f'{a0:.3f}', '-t', f'{a1 - a0:.3f}', '-i', video, '-vn',
             '-ac', '2', '-ar', str(SR), src])
        tmpd = os.path.join(d, 'demucs')
        retry_cpu = False
        try:
            demucs(src, tmpd, device, a.jobs)
        except RuntimeError as e:
            if device == 'cpu': raise
            log('separate', f'chunk {k}: {device} failed ({str(e).strip().splitlines()[-1][:120]}), retrying on CPU')
            retry_cpu = True
        # Leave the exception scope before retrying so its traceback cannot retain GPU tensors.
        if retry_cpu:
            demucs(src, tmpd, 'cpu', a.jobs)
        sd = os.path.join(tmpd, 'htdemucs', 'src')
        i0, i1 = int(round((t0 - a0) * SR)), int(round((t1 - a0) * SR))
        for name, dst in (('vocals', vp), ('no_vocals', bp)):
            x, _ = sf.read(os.path.join(sd, f'{name}.wav'), dtype='float32')
            x = x[i0:i1]
            if len(x) < i1 - i0: x = np.pad(x, ((0, i1 - i0 - len(x)), (0, 0)))
            sf.write(dst + '.tmp.flac', np.clip(x, -1, 1), SR, subtype='PCM_24'); os.replace(dst + '.tmp.flac', dst)
        shutil.rmtree(tmpd, ignore_errors=True); os.remove(src)
        pr.step(extra=f'(chunk {k + 1}/{n})')
    return [(k * L, min(dur, (k + 1) * L), *chunk_paths(wd, k, L)[1:]) for k in range(n)]
