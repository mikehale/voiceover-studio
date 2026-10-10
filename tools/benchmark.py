#!/usr/bin/env python3
"""Repeatable, isolated pipeline and clone benchmarks; Python standard library only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / 'Library/Application Support/Voiceover Studio'
GIB = 2 ** 30


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_id(path):
    p = Path(path).resolve()
    s = p.stat()
    return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def command_output(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()


def source_id(repo):
    files = sorted(p for folder in ('pipeline', 'tools') for p in (repo / folder).rglob('*.py'))
    content = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'commit': command_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
            'status': command_output(['git', '-C', str(repo), 'status', '--porcelain']),
            'source_sha256': digest(content), 'files': content}


def process_table():
    rows = command_output(['ps', '-axo', 'pid=,ppid=,pgid=,rss=,pcpu=,command='])
    result = []
    for row in rows.splitlines():
        fields = row.split(None, 5)
        if len(fields) == 6:
            result.append(dict(pid=int(fields[0]), ppid=int(fields[1]), pgid=int(fields[2]),
                               rss_bytes=int(fields[3]) * 1024, cpu_percent=float(fields[4]), command=fields[5]))
    return result


def swap_bytes(text):
    match = re.search(r'used\s*=\s*([\d.]+)([KMG])', text)
    if not match:
        raise ValueError('Cannot read system swap usage')
    return int(float(match[1]) * {'K': 1024, 'M': 1024**2, 'G': GIB}[match[2]])


def memory_sample(pgid=None):
    if platform.system() != 'Darwin':
        raise RuntimeError('Memory telemetry currently requires macOS')
    swap = swap_bytes(command_output(['sysctl', '-n', 'vm.swapusage']))
    pressure = command_output(['memory_pressure'])
    match = re.search(r'free percentage:\s*(\d+)%', pressure)
    if not match:
        raise ValueError('Cannot read memory pressure')
    processes = [p for p in process_table() if p['pgid'] == pgid] if pgid else []
    return {'swap_bytes': swap, 'free_percent': int(match[1]),
            'tree_rss_bytes': sum(p['rss_bytes'] for p in processes), 'processes': processes}


def competing_jobs():
    # Only refuse other heavy voiceover jobs. The idle app server is harmless.
    return [p for p in process_table() if p['pid'] != os.getpid() and
            re.search(r'python[^ ]* .*?(?:-m lotgh_vo(?:\s|\.clone_worker)|memprobe\.py|_benchmark_worker\.py)', p['command'])]


def guard_reason(sample, limits):
    if sample['swap_bytes'] > limits.max_swap_gib * GIB:
        return 'system swap limit exceeded'
    if sample['free_percent'] < limits.min_free_percent:
        return 'system free-memory limit exceeded'
    if sample['tree_rss_bytes'] > limits.max_rss_gib * GIB:
        return 'process-group RSS limit exceeded'
    return None


def stop_group(proc):
    # Include grandchildren and workers whose parents already exited.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    if not any(p['pgid'] == proc.pid for p in process_table()):
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def supervise(cmd, env, folder, limits):
    before = memory_sample()
    reason = guard_reason(before, limits)
    if reason:
        raise RuntimeError('Benchmark not started: ' + reason)
    started = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True, bufsize=1)
    events = []
    def read_output():
        with (folder / 'run.log').open('w') as log, (folder / 'events.jsonl').open('w') as event_log:
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.startswith('BENCH '):
                    try:
                        ev = json.loads(line[6:])
                        ev['elapsed_s'] = time.monotonic() - started
                        events.append(ev)
                        event_log.write(json.dumps(ev) + '\n'); event_log.flush()
                    except (ValueError, TypeError):
                        pass
                elif line.startswith('[', 0) or line.startswith('CLONE '):
                    # Full verbose worker output remains in run.log.
                    if 'gen_step' not in line and '"results"' not in line and '"lines": {' not in line:
                        print(line.rstrip(), flush=True)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    samples = []
    status, reason = 'completed', None
    try:
        with (folder / 'memory.jsonl').open('w') as output:
            while True:
                sample = memory_sample(proc.pid)
                sample['elapsed_s'] = time.monotonic() - started
                samples.append(sample)
                output.write(json.dumps(sample) + '\n'); output.flush()
                reason = guard_reason(sample, limits)
                if sample['elapsed_s'] > limits.timeout_minutes * 60:
                    reason = 'time limit exceeded'
                if reason:
                    status = 'aborted'; stop_group(proc); break
                if proc.poll() is not None:
                    break
                time.sleep(limits.interval)
    except KeyboardInterrupt:
        status, reason = 'interrupted', 'user interrupted'; stop_group(proc)
    except Exception as exc:
        status, reason = 'aborted', 'telemetry failed: ' + str(exc); stop_group(proc)
    finally:
        if proc.poll() is None:
            stop_group(proc)
        reader.join(timeout=10)
        if reader.is_alive():
            stop_group(proc)
            reader.join(timeout=5)
        if not reader.is_alive():
            proc.stdout.close()
    if status == 'completed' and proc.returncode:
        status = 'failed'
    steps = [e for e in events if e.get('ev') == 'line']
    synth = [e for e in events if e.get('ev') == 'synth']
    results = {}
    for event in events:
        if event.get('ev') == 'result':
            results.update(event.get('results', {}))
    successful = [r for r in results.values() if r.get('path')]
    initial_cached = next((e['cached'] for e in events if e.get('ev') == 'gen_start'), 0)
    compute = sum(e['seconds'] for e in synth)
    audio = sum(e['audio_s'] for e in synth)
    summary = {'status': status, 'reason': reason, 'exit_code': proc.returncode,
               'wall_s': time.monotonic() - started, 'system_before': before,
               'peak_tree_rss_gib': max((s['tree_rss_bytes'] for s in samples), default=0) / GIB,
               'peak_system_swap_gib': max((s['swap_bytes'] for s in samples), default=0) / GIB,
               'min_system_free_percent': min((s['free_percent'] for s in samples), default=None),
               'line_events': len(steps), 'synth_attempts': len(synth),
               'synth_compute_s': compute, 'synth_audio_s_including_retries': audio,
               'synth_rtf_including_retries': compute / audio if audio else None,
               'generated_lines': len(successful) - initial_cached,
               'cached_lines': initial_cached,
               'failed_lines': sum('error' in r for r in results.values()),
               'accent_fallback_lines': sum(bool(r.get('accent_fallback')) for r in results.values()),
               'worker_restarts': sum(e.get('ev') == 'recycle' for e in events),
               'peak_mps_live_gib': max((e.get('mps_live_bytes', 0) for e in events), default=0) / GIB,
               'peak_mps_driver_gib': max((e.get('mps_driver_bytes', 0) for e in events), default=0) / GIB}
    summary['length_retries'] = (folder / 'run.log').read_text().count('suspicious length')
    metrics = folder / 'pipeline.json'
    if metrics.exists():
        summary['pipeline'] = json.loads(metrics.read_text())
    write_json(folder / 'summary.json', summary)
    return summary


def runtime_env(repo, data):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(repo / 'pipeline'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               HF_HOME=str(data / 'hf'), TORCH_HOME=str(data / 'torch'), HF_HUB_OFFLINE='1',
               LOTGH_CLONE_MODEL=str(data / 'models/chatterbox'), PYTORCH_ENABLE_MPS_FALLBACK='1',
               TOKENIZERS_PARALLELISM='false', TQDM_DISABLE='1')
    bundled = Path('/Applications/Voiceover Studio.app/Contents/Resources/bin')
    for name in ('ffmpeg', 'ffprobe'):
        for candidate in (repo / 'cache' / name, bundled / name):
            if candidate.is_file():
                env['LOTGH_' + name.upper()] = str(candidate); break
    return env


def positive(value):
    n = float(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    for mode in ('run', 'clone'):
        p = sub.add_parser(mode, help='full pipeline sample' if mode == 'run' else 'isolated saved clone job sample')
        p.add_argument('--run-dir', type=Path, required=True, help='new directory; existing paths are refused')
        p.add_argument('--repo', type=Path, default=ROOT)
        p.add_argument('--data-dir', type=Path, default=DATA)
        p.add_argument('--label', default='baseline')
        p.add_argument('--interval', type=positive, default=3)
        p.add_argument('--max-swap-gib', type=positive, default=8)
        p.add_argument('--max-rss-gib', type=positive, default=24)
        p.add_argument('--min-free-percent', type=int, choices=range(0, 101), default=15, metavar='0..100')
        p.add_argument('--timeout-minutes', type=positive, default=120)
        p.add_argument('--wait-idle', action='store_true', help='wait for other voiceover GPU jobs to finish')
        if mode == 'run':
            p.add_argument('input', type=Path)
            p.add_argument('--start', type=float, default=0, help='seconds')
            p.add_argument('--duration', type=positive, default=900, help='seconds; default 15 minutes')
            p.add_argument('--clone', action='store_true')
            p.add_argument('--accent', choices=['original', 'german_v3'], default='german_v3')
            p.add_argument('--until', choices=['ocr', 'separate', 'clone', 'tts', 'mix', 'mux'], default='mux')
            p.add_argument('--jobs', type=int, default=4, help='OCR workers (bounded for memory)')
        else:
            p.add_argument('--job', type=Path, required=True)
            p.add_argument('--lines', type=int, default=320)
            p.add_argument('--offset', type=int, default=0, help='chronological line offset')
    p = sub.add_parser('compare', help='compare two successful identical-workload runs')
    p.add_argument('baseline', type=Path)
    p.add_argument('candidate', type=Path)
    return ap


def compare(a, b):
    ma, mb = [json.loads((p / 'manifest.json').read_text()) for p in (a, b)]
    sa, sb = [json.loads((p / 'summary.json').read_text()) for p in (a, b)]
    if ma['workload_sha256'] != mb['workload_sha256']:
        raise ValueError('Workloads differ; refusing to present a speedup')
    if any(s['status'] != 'completed' for s in (sa, sb)):
        raise ValueError('Both runs must complete before comparing speed')
    if any(s['cached_lines'] for s in (sa, sb)):
        raise ValueError('Generated-audio cache hits invalidate a cold comparison')
    for name in ('wall_s', 'peak_tree_rss_gib', 'peak_mps_driver_gib', 'synth_rtf_including_retries',
                 'generated_lines', 'failed_lines', 'accent_fallback_lines', 'length_retries'):
        print(f'{name}: {sa.get(name)} -> {sb.get(name)}')
    print(f"Wall time reduction: {100 * (1 - sb['wall_s'] / sa['wall_s']):.1f}%")
    print('Check generated audio and fallback counts before accepting a performance improvement.')


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == 'compare':
        compare(args.baseline, args.candidate); return 0
    repo, data, folder = args.repo.resolve(), args.data_dir.resolve(), args.run_dir.resolve()
    if folder.exists():
        raise ValueError('Run directory already exists; choose a new one to avoid cache-biased timings')
    if args.mode == 'clone':
        if args.lines < 1 or args.offset < 0:
            raise ValueError('--lines must be positive and --offset nonnegative')
        job = json.loads(args.job.read_text())
        selected = sorted(job['lines'], key=lambda x: (x['s'], x['i']))[args.offset:args.offset + args.lines]
        if len(selected) != args.lines:
            raise ValueError('Not enough lines in the saved job for this sample')
        job['lines'] = selected
        job.pop('dir', None)
        workload = {'mode': 'clone', 'job': job,
                    'stems': [file_id(p) for c in job['chunks'] for p in c[2:]],
                    'custom': [file_id(v['path']) for v in job.get('custom', {}).values()]}
    else:
        if args.start < 0 or args.jobs < 1:
            raise ValueError('--start must be nonnegative and --jobs positive')
        if args.until == 'clone' and not args.clone:
            raise ValueError('--until clone requires --clone')
        workload = {'mode': 'run', 'input': file_id(args.input), 'start': args.start,
                    'duration': args.duration, 'clone': args.clone, 'accent': args.accent,
                    'until': args.until, 'jobs': args.jobs}
    wait_start = time.monotonic()
    while competing_jobs():
        if not args.wait_idle:
            raise RuntimeError('Another voiceover job is running; use --wait-idle or finish it first')
        if time.monotonic() - wait_start > args.timeout_minutes * 60:
            raise RuntimeError('Timed out waiting for other voiceover jobs')
        print('Waiting for the other voiceover job to finish...', flush=True)
        time.sleep(30)
    folder.mkdir(parents=True)
    env = runtime_env(repo, data)
    python = data / ('clone-venv' if args.mode == 'clone' else 'venv') / 'bin/python'
    if args.mode == 'clone':
        job['dir'] = str(folder / 'cache/clone')
        write_json(folder / 'job.json', job)
        cmd = [str(python), '-u', str(ROOT / 'tools/_benchmark_worker.py'), str(folder / 'job.json')]
    else:
        cmd = [str(python), '-u', '-m', 'lotgh_vo', str(args.input.resolve()),
               '-o', str(folder / 'output.mp4'), '--workdir', str(folder / 'cache'),
               '--start', str(args.start), '--end', str(args.start + args.duration),
               '--models', str(data / 'models'), '--video-codec', 'copy', '--jobs', str(args.jobs),
               '--until', args.until, '--metrics-json', str(folder / 'pipeline.json')]
        if args.clone:
            cmd += ['--clone', '--clone-python', str(data / 'clone-venv/bin/python'), '--clone-accent', args.accent]
    manifest = {'label': args.label, 'created_unix': time.time(), 'source': source_id(repo),
                'harness': source_id(ROOT), 'workload': workload, 'workload_sha256': digest(workload),
                'command': cmd, 'platform': platform.platform(), 'machine': platform.machine(),
                'runtime': command_output([str(python), '-c',
                    'import sys,importlib.metadata as m,json; print(json.dumps(dict(python=sys.version,packages={d.metadata["Name"]:d.version for d in m.distributions()})))']),
                'limits': {k: getattr(args, k) for k in ('max_swap_gib', 'max_rss_gib', 'min_free_percent', 'timeout_minutes', 'interval')}}
    write_json(folder / 'manifest.json', manifest)
    try:
        summary = supervise(cmd, env, folder, args)
    except Exception as exc:
        write_json(folder / 'summary.json', {'status': 'not_started', 'reason': str(exc)})
        raise
    final_source = source_id(repo)
    if final_source['source_sha256'] != manifest['source']['source_sha256']:
        summary.update(status='invalid', reason='Python source changed during the benchmark')
        write_json(folder / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    print('Results: ' + str(folder))
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'benchmark: {exc}', file=sys.stderr)
        sys.exit(1)
