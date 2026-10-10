"""Instrument the real clone worker without changing its synthesis settings."""
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')
import lotgh_vo.clone_worker as worker
import torch


def telemetry(**event):
    if torch.backends.mps.is_available():
        event.update(mps_live_bytes=torch.mps.current_allocated_memory(),
                     mps_driver_bytes=torch.mps.driver_allocated_memory())
    print('BENCH ' + json.dumps(event), flush=True)


original_synth = worker.synth

def synth(*args, **kwargs):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    start = time.monotonic()
    wave, sr = original_synth(*args, **kwargs)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    telemetry(ev='synth', seconds=time.monotonic() - start, audio_s=len(wave) / sr)
    return wave, sr


worker.synth = synth
original_emit = worker.emit
recycle = False
completed = False


def emit(**event):
    global recycle, completed
    original_emit(**event)
    kind = event.get('ev')
    if kind == 'gen_step':
        telemetry(ev='line', line=event['done'], worker_rtf=event.get('rtf'))
    elif kind == 'gen_start':
        telemetry(**event)
    elif kind == 'recycle':
        recycle = True
        telemetry(**event)
    elif kind == 'done':
        completed = True
        telemetry(ev='result', results=event['results'])


worker.emit = emit
if __name__ == '__main__':
    try:
        worker.run_job(json.load(open(sys.argv[1])))
        # Parent runner can restart clean processes and preserve generation cache.
        if recycle:
            count = int(os.environ.get('VO_BENCH_RESTARTS', '0')) + 1
            if count >= 40:
                raise RuntimeError('Worker restart limit reached')
            os.environ['VO_BENCH_RESTARTS'] = str(count)
            sys.stdout.flush(); sys.stderr.flush()
            os.execv(sys.executable, [sys.executable, '-u', __file__, sys.argv[1]])
        if not completed:
            raise RuntimeError('Worker exited without a completion event')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
