"""Kokoro TTS per subtitle line in a worker pool, fitted to its slot, cached per line."""
import os, time
import numpy as np, soundfile as sf
from .util import FFMPEG, log, Progress, atomic_json, h, load_json, ort_session, pick_provider, run

SR = 44100
TTS_VERSION = 2
_W = {}

def _load_kokoro(model, voices, provider, threads):
    from kokoro_onnx import Kokoro
    sess = ort_session(model, provider, threads)
    return Kokoro.from_session(sess, voices)

def _init_worker(model, voices, provider, threads):
    _W['k'] = _load_kokoro(model, voices, provider, threads)

def _trim(a, sr, th=0.01):
    idx = np.where(np.abs(a) > th)[0]
    return a if len(idx) == 0 else a[max(0, idx[0] - int(0.02 * sr)): idx[-1] + int(0.05 * sr)]

def synth_line(job):
    """Generate one line; Kokoro native speed first (cleaner), ffmpeg atempo for the rest; total <= max_tempo."""
    j = job; out = j['out']
    if os.path.exists(out):
        return j['i'], sf.info(out).duration, j.get('meta_cached', {})
    k = _W['k']
    base = float(j.get('base_speed', 1.0))
    a, sr = k.create(j['text'], voice=j['voice'], speed=base, lang=j['lang']); a = _trim(a, sr)
    raw = len(a) / sr; need = raw / j['slot']; sp = base
    if need > 1.0:
        sp = min(base * need, max(1.25, base))
        a, sr = k.create(j['text'], voice=j['voice'], speed=sp, lang=j['lang']); a = _trim(a, sr)
    tempo = min(max((len(a) / sr) / j['slot'], 1.0), j['max_tempo'] / sp)
    tmp_in = out + '.in.wav'; sf.write(tmp_in, a, sr)
    filt = f'aresample={SR}' + (f',atempo={tempo:.4f}' if tempo > 1.001 else '')
    run([FFMPEG, '-v', 'error', '-y', '-i', tmp_in, '-af', filt, '-ac', '1', '-c:a', 'pcm_f32le', out + '.tmp.wav'])
    os.replace(out + '.tmp.wav', out); os.remove(tmp_in)
    return j['i'], sf.info(out).duration, dict(raw=round(raw, 2), speed=round(sp, 2), atempo=round(tempo, 3))

def run_tts(a, lines, dur, wd):
    """lines: [[start,end,text,voice,lang], ...] (already filtered). Returns placements [[start,dur,path,i]]."""
    from multiprocessing import get_context
    tdir = os.path.join(wd, 'tts'); os.makedirs(tdir, exist_ok=True)
    jobs = []
    speeds = dict((load_json(a.voices, {}) or {}).get('speeds', {}))
    for ov in getattr(a, 'voice_speed', []) or []:
        kk, _, vv = ov.partition('='); speeds[kk.strip()] = float(vv)
    for i, (s, e, text, voice, lang) in enumerate(lines):
        base = float(speeds.get(voice, 1.0))
        nxt = lines[i + 1][0] if i + 1 < len(lines) else dur
        slot = max(0.3, nxt - s - 0.08)
        key = h(TTS_VERSION, text, voice, lang, round(slot, 2), a.max_tempo, base)
        jobs.append(dict(i=i, text=text, voice=voice, lang=lang, slot=slot, max_tempo=a.max_tempo, base_speed=base,
                         out=os.path.join(tdir, f'{key}.wav')))
    todo = [j for j in jobs if not os.path.exists(j['out'])]
    log('tts', f'{len(jobs)} lines, {len(jobs) - len(todo)} cached, {len(todo)} to synthesise, {a.tts_jobs} workers')
    meta = {}
    if todo:
        provider = 'cpu'
        tts_pref = getattr(a, 'tts_provider', 'cpu')
        if tts_pref == 'cpu':
            log('tts', 'onnxruntime provider: CPU (CoreML skipped for Kokoro; its dynamic shapes are unsupported)')
        else:
            def bench(prov):
                k = _load_kokoro(a.kokoro_model, a.kokoro_voices, prov, 0)
                k.create('Warm up.', voice='bm_george', lang='en-gb')
                t = time.time(); k.create('The war has already been going on for one hundred and fifty years.',
                                          voice='bm_george', lang='en-gb')
                return time.time() - t
            provider = pick_provider('tts', tts_pref, bench)
        threads = max(1, a.cpu_cores // a.tts_jobs) if provider == 'cpu' else 0
        pr = Progress('tts', len(todo))
        ctx = get_context('spawn')
        with ctx.Pool(a.tts_jobs, initializer=_init_worker,
                      initargs=(a.kokoro_model, a.kokoro_voices, provider, threads)) as pool:
            for i, d, m in pool.imap_unordered(synth_line, todo, chunksize=1):
                meta[i] = m; pr.step()
    # sequential placement: a line that still overruns its slot pushes the next one back (never overlaps)
    place, cursor, report = [], 0.0, []
    for j, (s, e, text, voice, lang) in zip(jobs, lines):
        d = sf.info(j['out']).duration
        start = max(s, cursor + 0.05); cursor = start + d
        place.append([start, d, j['out'], j['i']])
        report.append(dict(i=j['i'] + 1, start=round(start, 2), slot=round(j['slot'], 2), dur=round(d, 2), voice=voice,
                           shift=round(start - s, 2), text=text, **meta.get(j['i'], {})))
    atomic_json(os.path.join(wd, 'tts_report.json'), report)
    shifted = sum(1 for r in report if r['shift'] > 0.01)
    log('tts', f'placed {len(place)} lines; {shifted} pushed later by an overrunning previous line '
               f'(max {max([r["shift"] for r in report] or [0]):.2f}s)')
    return place
