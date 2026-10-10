"""Optional cloned voices (Chatterbox). Only used with --clone; the default Kokoro path never imports this.

Runs lotgh_vo.clone_worker in the separate Chatterbox venv (--clone-python): speakers are grouped from the Demucs
vocals stem by voice fingerprint, each gets a 6-10 s reference clip (the narrator gets its own from cyan lines), and
every line is generated in its speaker's voice. Here each line is then fitted to its subtitle slot exactly like
Kokoro lines (trim, speed-up <= max_tempo, no overlap). Lines that fail fall back to Kokoro (run_tts).
With --clone-accent german_v3 the worker reads the English text with Chatterbox Multilingual V3 (language 'de');
fallback order per line: German clone -> original-accent clone -> Kokoro, each logged."""
import json, os, subprocess, sys
import numpy as np, soundfile as sf
from .util import FFMPEG, log, Progress, atomic_json, h, run
from .tts import _trim, SR

CLONE_VERSION = 2
CLONE_SEED = 1234
EXAGGERATION, CFG = 0.5, 0.5
NARRATOR_CLASSES = ('cyan',)
SKIP_CLASSES = ('white',)        # song lyrics (if voiced at all) stay Kokoro

def _fit(raw, slot, max_tempo, out):
    a, sr = sf.read(raw, dtype='float32')
    if a.ndim > 1: a = a.mean(1)
    a = _trim(a, sr); dur = len(a) / sr
    tempo = min(max(dur / slot, 1.0), max_tempo)
    tmp_in = out + '.in.wav'; sf.write(tmp_in, a, sr)
    filt = f'aresample={SR}' + (f',atempo={tempo:.4f}' if tempo > 1.001 else '')
    run([FFMPEG, '-v', 'error', '-y', '-i', tmp_in, '-af', filt, '-ac', '1', '-c:a', 'pcm_f32le', out + '.tmp.wav'])
    os.replace(out + '.tmp.wav', out); os.remove(tmp_in)
    return dict(raw=round(dur, 2), speed=1.0, atempo=round(tempo, 3))

def run_clone(a, lines, classes, chunks, dur, wd):
    """lines: same list run_tts gets; classes: colour class per line. Returns {i: dict(out=wav, meta=...)}."""
    cdir = os.path.join(wd, 'clone'); fdir = os.path.join(cdir, 'fit'); os.makedirs(fdir, exist_ok=True)
    jl = []
    for i, (s, e, text, voice, lang) in enumerate(lines):
        if classes[i] in SKIP_CLASSES: continue
        nxt = lines[i + 1][0] if i + 1 < len(lines) else dur
        jl.append(dict(i=i, s=s, e=e, text=text, slot=max(0.3, nxt - s - 0.08),
                       role='narrator' if classes[i] in NARRATOR_CLASSES else 'dialogue'))
    if not jl: return {}
    job = dict(version=CLONE_VERSION, seed=CLONE_SEED, exaggeration=EXAGGERATION, cfg=CFG, device=a.clone_device,
               dir=cdir, lines=jl, chunks=[list(c) for c in chunks], accent=getattr(a, 'clone_accent', 'original'),
               custom=getattr(a, 'clone_custom', None) or {})
    jp = os.path.join(cdir, 'job.json'); atomic_json(jp, job)
    acc = job['accent'] if job['accent'] != 'original' else ''
    for role, v in job['custom'].items():
        log('clone', f"custom {role} voice: {v['name']} ({v['accent']} accent, {v['path']})")
    log('clone', f"cloned voices (Chatterbox{', ' + acc + ' accent' if acc else ''}) for {len(jl)} lines; "
                 + (f'the original-accent clone, then Kokoro, is used for any line that fails' if acc else 'Kokoro is used for any line that fails'))
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', TQDM_DISABLE='1')
    res, pr, info = None, None, {}
    try:
        p = subprocess.Popen([a.clone_python, '-u', '-m', 'lotgh_vo.clone_worker', jp], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            if not line.startswith('CLONE '):
                tail = (tail + [line])[-20:]; continue
            ev = json.loads(line[6:])
            k = ev.get('ev')
            if k == 'log': log('clone', ev['msg'])
            elif k == 'phase': log('clone', {'load': 'loading Chatterbox', 'speakers': 'grouping speakers'}.get(ev['name'], ev['name']))
            elif k == 'speakers': info['speakers'] = ev['speakers']; info['lines'] = ev['lines']
            elif k == 'gen_start':
                log('clone', f"{ev['total'] + ev['cached']} lines, {ev['cached']} cached, {ev['total']} to generate")
                pr = Progress('clone', ev['total'])
            elif k == 'gen_step' and pr:
                pr.step(extra=f"({ev['rtf']:.1f}s of compute per second of speech)" if ev.get('rtf') else '')
            elif k == 'done': res = ev
            elif k == 'fatal': log('clone', 'Chatterbox worker failed: ' + ev['error'].strip().splitlines()[-1][:300])
        p.wait()
        if p.returncode != 0 and res is None and tail:
            log('clone', 'worker output: ' + ' | '.join(tail[-5:])[-600:])
    except Exception as e:
        log('clone', f'could not run Chatterbox ({e}); all lines use Kokoro')
    if not res:
        log('clone', 'cloned voices unavailable for this run; falling back to Kokoro for every line'); return {}
    out, failed = {}, []
    for L in jl:
        r = res['results'].get(str(L['i'])) or res['results'].get(L['i'])
        if not r or 'path' not in r:
            failed.append((L['i'], (r or {}).get('error', 'no output'))); continue
        key = h('clonefit', CLONE_VERSION, os.path.basename(r['path']), round(L['slot'], 2), a.max_tempo)
        f = os.path.join(fdir, f'{key}.wav')
        try:
            meta = _fit(r['path'], L['slot'], a.max_tempo, f) if not os.path.exists(f) else {}
            out[L['i']] = dict(out=f, meta=dict(meta, clone=r['spk'], ref=r['ref'], accent=r.get('accent', 'original'),
                                                **({'custom': r['custom']} if r.get('custom') else {})))
        except Exception as e:
            failed.append((L['i'], f'fit failed: {e}'))
    for i, why in failed:
        log('clone', f'line {i + 1}: using Kokoro {lines[i][3]} instead ({str(why)[:160]})')
    atomic_json(os.path.join(cdir, 'clone_report.json'), dict(device=res.get('device'), rtf=res.get('rtf'),
                gen_s=res.get('gen_s'), audio_s=res.get('audio_s'), total_s=res.get('total_s'),
                speakers=info.get('speakers'), lines=info.get('lines'), fallback=[i for i, _ in failed],
                accent=job['accent'], accent_lines=sorted(i for i, v in out.items() if v['meta']['accent'] != 'original'),
                accent_fallback=sorted(int(i) for i, r in res['results'].items() if r.get('accent_fallback'))))
    nc = sum(1 for v in out.values() if v['meta'].get('custom'))
    if job['custom']: log('clone', f'{nc} lines in custom voices')
    if acc or any(v['accent'] != 'original' for v in job['custom'].values()):
        acc = acc or '/'.join(sorted({v['accent'] for v in job['custom'].values() if v['accent'] != 'original'}))
        n_acc = sum(1 for v in out.values() if v['meta']['accent'] != 'original')
        log('clone', f'{n_acc} lines with {acc} accent, {len(out) - n_acc} with the original accent')
    log('clone', f"{len(out)} lines cloned, {len(failed)} on Kokoro; Chatterbox on {str(res.get('device')).upper()}, "
                 f"{res.get('gen_s')}s to generate {res.get('audio_s')}s of speech ({res.get('rtf') or 0:.2f}s per second of speech)")
    return out
