"""Cloned-voice worker. Runs ONLY in the optional Chatterbox venv (never imported by the base pipeline).

python -m lotgh_vo.clone_worker JOB.json      -> speaker grouping, reference clips, one Chatterbox wav per line
python -m lotgh_vo.clone_worker --selftest    -> load the model, synthesise one sentence, print device + speed

Talks to the parent (lotgh_vo.clone) through stdout lines that start with 'CLONE ' followed by JSON.
"""
import hashlib, json, os, re, shutil, sys, time, traceback
import numpy as np

REPO_ID = 'ResembleAI/chatterbox'
REVISION = '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'   # pinned model snapshot (~3.2 GB)
MODEL_FILES = ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']
SR_IN = 44100
MIN_SPEECH_EMB = 0.6     # s of speech needed to fingerprint a line
TIE_MARGIN = 0.02        # cosine margin below which the top-2 speakers count as a tie
REF_MIN, REF_TARGET, REF_MAX = 6.0, 8.0, 10.0
REF_ABS_MIN = 3.0
CLEAN_SNR = 15.0         # dB of voice over the background stem for a line to be used in a reference        # below this a speaker gets no reference (its lines fall back to Kokoro)
MAX_SPEAKERS = 8

def emit(**kw): print('CLONE ' + json.dumps(kw), flush=True)
def say(msg): emit(ev='log', msg=msg)

def pick_device(pref):
    import torch
    if pref in ('auto', 'mps') and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def model_dir():
    """The app keeps the model as plain files in LOTGH_CLONE_MODEL (easy to measure and remove);
    without it (command-line use) the normal Hugging Face cache is used."""
    d = os.environ.get('LOTGH_CLONE_MODEL')
    if d and all(os.path.exists(os.path.join(d, f)) for f in MODEL_FILES): return d
    from huggingface_hub import hf_hub_download
    p = None
    for f in MODEL_FILES: p = hf_hub_download(REPO_ID, f, revision=REVISION)
    return os.path.dirname(p)

def download(dest):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, revision=REVISION, allow_patterns=MODEL_FILES, local_dir=dest)
    shutil.rmtree(os.path.join(dest, '.cache'), ignore_errors=True)        # download bookkeeping only
    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(dest, f))]
    if missing: raise RuntimeError(f'model files missing after download: {missing}')
    return dest

def load_model(pref):
    import torch
    from chatterbox.tts import ChatterboxTTS
    dev = pick_device(pref); d = model_dir()
    try:
        m = ChatterboxTTS.from_local(d, dev)
    except Exception as e:
        if dev == 'cpu': raise
        say(f'loading on {dev.upper()} failed ({str(e)[:160]}); using CPU'); dev = 'cpu'; m = ChatterboxTTS.from_local(d, dev)
    if dev == 'cpu': torch.set_num_threads(max(1, os.cpu_count() or 4))
    return m, dev

# ------------------------------------------------------------------ audio helpers
class Vocals:
    """Reads [t0, t1) of a Demucs stem (vocals, or background with stem=3) across its time chunks (mono, 44.1 kHz)."""
    def __init__(self, chunks, stem=2): self.chunks = chunks; self.stem = stem
    def read(self, t0, t1):
        import soundfile as sf
        out = []
        for c in self.chunks:
            c0, c1, vp = c[0], c[1], c[self.stem]
            a, b = max(t0, c0), min(t1, c1)
            if b <= a: continue
            x, sr = sf.read(vp, start=int(round((a - c0) * SR_IN)), stop=int(round((b - c0) * SR_IN)), dtype='float32')
            out.append(x.mean(1) if x.ndim > 1 else x)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

def speech_parts(x, top_db=30):
    import librosa
    if len(x) < 2048 or np.abs(x).max() < 1e-4: return []
    iv = librosa.effects.split(x, top_db=top_db, frame_length=2048, hop_length=512)
    return [(int(a), int(b)) for a, b in iv if b - a > int(0.08 * SR_IN)]

def trimmed(x, parts):
    if not parts: return np.zeros(0, np.float32)
    return x[max(0, parts[0][0] - int(.03 * SR_IN)): parts[-1][1] + int(.05 * SR_IN)]

def unit(v): return v / (np.linalg.norm(v) + 1e-9)

# ------------------------------------------------------------------ speakers
def fingerprint(model, lines, voc, bg=None):
    import librosa
    for L in lines:
        x = voc.read(L['s'], L['e']); parts = speech_parts(x)
        L['speech'] = round(sum(b - a for a, b in parts) / SR_IN, 2); L['emb'] = None; L['_clip'] = trimmed(x, parts)
        L['snr'] = 0.0
        if parts and bg is not None:         # how loud the voice is over music/effects: bleed makes poor references
            y = bg.read(L['s'], L['e'])
            v = np.concatenate([x[a:b] for a, b in parts]); u = np.concatenate([y[a:min(b, len(y))] for a, b in parts])
            L['snr'] = round(float(20 * np.log10((np.sqrt(np.mean(v ** 2)) + 1e-9) / (np.sqrt(np.mean(u ** 2)) + 1e-9))), 1)
        if L['speech'] >= MIN_SPEECH_EMB:
            w = librosa.resample(np.concatenate([x[a:b] for a, b in parts]), orig_sr=SR_IN, target_sr=16000)
            L['emb'] = model.ve.embeds_from_wavs([w], sample_rate=16000, trim_top_db=None)[0].astype(np.float64)

def cluster(dlg, thresh):
    """Average-linkage cosine clustering of dialogue lines; tiny groups merged into their nearest neighbour."""
    from scipy.cluster.hierarchy import linkage, fcluster
    E = [L for L in dlg if L['emb'] is not None]
    if not E: return {}
    if len(E) == 1: lab = [1]
    else:
        X = np.array([L['emb'] for L in E])
        lab = list(fcluster(linkage(X, 'average', metric='cosine'), t=1 - thresh, criterion='distance'))
    groups = {}
    for L, g in zip(E, lab): groups.setdefault(int(g), []).append(L)
    def cent(g): return unit(np.mean([L['emb'] for L in groups[g]], 0))
    def speech(g): return sum(L['speech'] for L in groups[g])
    while len(groups) > 1:
        small = [g for g in groups if speech(g) < REF_MIN]
        if not small and len(groups) <= MAX_SPEAKERS: break
        g = min(small, key=speech) if small else min(groups, key=speech)
        c = cent(g); tgt = max((h for h in groups if h != g), key=lambda h: float(c @ cent(h)))
        groups[tgt] += groups.pop(g)
    order = sorted(groups, key=lambda g: min(L['s'] for L in groups[g]))    # speaker numbers in order of appearance
    return {f'speaker{k + 1}': cent(g) for k, g in enumerate(order)}

def assign(dlg, cents):
    """Nearest centroid; ties and lines too short to fingerprint use conversation context."""
    names = list(cents)
    for L in dlg:
        L['spk'] = None; L['why'] = ''
        if L['emb'] is None: L['cand'] = names; L['why'] = 'short'; continue
        sims = sorted(((float(L['emb'] @ cents[n]), n) for n in names), reverse=True)
        L['sims'] = {n: round(s, 3) for s, n in sims}
        if len(sims) == 1 or sims[0][0] - sims[1][0] >= TIE_MARGIN:
            L['spk'] = sims[0][1]; L['why'] = f'fingerprint {sims[0][0]:.2f}'
        else:
            L['cand'] = [sims[0][1], sims[1][1]]; L['why'] = 'tie'
    for k, L in enumerate(dlg):
        if L['spk']: continue
        prev = dlg[k - 1] if k and L['s'] - dlg[k - 1]['e'] < 3.0 else None
        cont = L['text'].lstrip().startswith(('...', '…')) or (prev is not None and prev['text'].rstrip().endswith(('...', '…'))
                                                               and not L['text'].lstrip()[:1].isupper())
        if prev is not None and cont and prev['spk'] in L['cand']:
            L['spk'] = prev['spk']; L['why'] += ' -> continues previous line'; continue
        if prev is not None and len(names) > 1:
            # turn-taking: the nearest-in-time other speaker within 30 s, before or after
            near = sorted((abs(M['s'] - L['s']), M['spk']) for M in dlg
                          if M is not L and M['spk'] and M['spk'] != prev['spk'] and M['spk'] in L['cand'] and abs(M['s'] - L['s']) < 30)
            if near: L['spk'] = near[0][1]; L['why'] += ' -> turn order'; continue
        if L['why'] == 'tie': L['spk'] = L['cand'][0]; L['why'] += ' -> best fingerprint'; continue
        L['spk'] = prev['spk'] if prev is not None and prev['spk'] else names[0] if names else None
        L['why'] += ' -> previous speaker' if prev is not None else ' -> main speaker'

def build_ref(lines, cent, path):
    """Concatenate the most typical clean lines of one speaker into a 6-10 s reference clip."""
    import soundfile as sf
    pool = [L for L in lines if L['emb'] is not None and len(L['_clip'])]
    clean = [L for L in pool if L['snr'] >= CLEAN_SNR]
    if sum(L['speech'] for L in clean) >= REF_MIN: pool = clean          # skip lines with loud music under them
    else: pool.sort(key=lambda L: -L['snr']); pool = pool[:max(3, len(clean))]
    pool.sort(key=lambda L: -float(L['emb'] @ cent) if cent is not None else -L['speech'])
    parts, used, tot = [], [], 0.0
    gap = np.zeros(int(.25 * SR_IN), np.float32)
    for L in pool:
        x = L['_clip']
        if tot + len(x) / SR_IN > REF_MAX:
            if tot >= REF_MIN: break
            x = x[:int((REF_MAX - tot) * SR_IN)]
        parts += [x, gap]; used.append(L['i']); tot += len(x) / SR_IN + .25
        if tot >= REF_TARGET: break
    if not parts: return None
    y = np.concatenate(parts[:-1]); dur = len(y) / SR_IN
    snr = round(float(np.mean([L['snr'] for L in pool if L['i'] in used])), 1)
    if dur < REF_ABS_MIN: return dict(lines=used, dur=round(dur, 2), hash=None, snr=snr)
    y = (y / (np.abs(y).max() + 1e-9) * 10 ** (-3 / 20)).astype(np.float32)
    hsh = hashlib.sha1(y.tobytes()).hexdigest()[:16]
    out = os.path.join(path, f'ref_{hsh}.wav')
    if not os.path.exists(out): sf.write(out, y, SR_IN, subtype='PCM_16')
    return dict(lines=used, dur=round(dur, 2), hash=hsh, path=out, snr=snr)

# ------------------------------------------------------------------ generation
def expected_len(text): return 0.5 + 0.065 * len(text)

def speakable(text):
    """Subtitle text -> model text. Chatterbox turns '...' into ', ', so a continuation line like
    '...have been seeking' starts with a comma and lower-case word and often comes out garbled."""
    t = re.sub(r'^[\s.,;:\u2026\-\u2013\u2014]+', '', text.strip())
    t = re.sub(r'(\.\.\.|\u2026)\s*$', ',', t).strip()
    if not t: return text
    return t[0].upper() + t[1:]

def synth(model, text, seed, exaggeration, cfg):
    import torch
    torch.manual_seed(seed)
    w = model.generate(speakable(text), exaggeration=exaggeration, cfg_weight=cfg)
    return w.squeeze(0).detach().cpu().numpy().astype(np.float32), model.sr

def run_job(job):
    import soundfile as sf
    t_all = time.time()
    voc = Vocals(job['chunks']); refdir = os.path.join(job['dir'], 'refs'); gdir = os.path.join(job['dir'], 'gen')
    os.makedirs(refdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)
    emit(ev='phase', name='load'); t = time.time()
    model, dev = load_model(job.get('device', 'auto'))
    say(f'Chatterbox loaded on {dev.upper()} in {time.time() - t:.0f}s')
    lines = job['lines']
    emit(ev='phase', name='speakers'); t = time.time()
    fingerprint(model, lines, voc, Vocals(job['chunks'], stem=3))
    nar = [L for L in lines if L['role'] == 'narrator']; dlg = [L for L in lines if L['role'] == 'dialogue']
    cents = cluster(dlg, job.get('cluster_sim', 0.80)); assign(dlg, cents)
    for L in nar: L['spk'] = 'narrator'; L['why'] = 'cyan subtitle'
    refs = {}
    if nar:
        e = [L['emb'] for L in nar if L['emb'] is not None]
        refs['narrator'] = build_ref(nar, unit(np.mean(e, 0)) if e else None, refdir)
    for n, c in cents.items(): refs[n] = build_ref([L for L in dlg if L['spk'] == n], c, refdir)
    spk_summary = {n: dict(lines=sum(1 for L in lines if L['spk'] == n), ref=r) for n, r in refs.items()}
    for n, s in spk_summary.items():
        r = s['ref'] or {}
        say(f"{n}: {s['lines']} lines, reference {r.get('dur', 0):.1f}s from lines {[i + 1 for i in r.get('lines', [])]}"
            f" (voice {r.get('snr', 0):.0f} dB over background)"
            + ('' if r.get('hash') else ' -> too little clean speech, these lines use Kokoro'))
    say(f'speaker grouping took {time.time() - t:.0f}s ({len(cents)} dialogue speakers + narrator)')
    emit(ev='speakers', speakers={n: dict(lines=s['lines'], ref_dur=(s['ref'] or {}).get('dur', 0),
                                          ref_lines=(s['ref'] or {}).get('lines', []), ref_hash=(s['ref'] or {}).get('hash'))
                                  for n, s in spk_summary.items()},
         lines={L['i']: dict(spk=L['spk'], why=L['why'], speech=L['speech']) for L in lines})
    # generation, grouped by speaker so each reference is prepared once
    res = {}; todo = []
    for L in lines:
        r = refs.get(L['spk']) or {}
        if not r.get('hash'): res[L['i']] = dict(error=f"no reference for {L['spk']}"); continue
        key = hashlib.sha1('|'.join(map(str, (job['version'], L['text'], r['hash'], job['seed'], job['exaggeration'],
                                              job['cfg']))).encode()).hexdigest()[:16]
        L['key'] = key; L['out'] = os.path.join(gdir, f'{key}.wav')
        if os.path.exists(L['out']): res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'], cached=True)
        else: todo.append(L)
    emit(ev='gen_start', total=len(todo), cached=len(res) - sum(1 for v in res.values() if 'error' in v))
    cur = None; gen_t = 0.0; gen_a = 0.0
    for L in sorted(todo, key=lambda L: (L['spk'], L['s'])):
        r = refs[L['spk']]
        try:
            if cur != r['hash']: model.prepare_conditionals(r['path'], exaggeration=job['exaggeration']); cur = r['hash']
            seed = (job['seed'] + int(L['key'][:8], 16)) % 2 ** 31
            t = time.time(); w, sr = synth(model, L['text'], seed, job['exaggeration'], job['cfg'])
            d = len(w) / sr
            if d < 0.25 or d > 2.5 * expected_len(L['text']):          # silence or run-on babble: one retry
                say(f"line {L['i'] + 1}: suspicious length {d:.1f}s, retrying with another seed")
                w, sr = synth(model, L['text'], seed + 7919, job['exaggeration'], job['cfg']); d = len(w) / sr
                if d < 0.25 or d > 2.5 * expected_len(L['text']): raise RuntimeError(f'bad output length {d:.1f}s')
            gen_t += time.time() - t; gen_a += d
            sf.write(L['out'] + '.tmp.wav', w, sr, subtype='FLOAT'); os.replace(L['out'] + '.tmp.wav', L['out'])
            res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'])
        except Exception as e:
            res[L['i']] = dict(error=str(e)[:300]); say(f"line {L['i'] + 1}: Chatterbox failed: {str(e)[:200]}")
        emit(ev='gen_step', done=L['i'], rtf=round(gen_t / gen_a, 2) if gen_a else None)
    emit(ev='done', results=res, device=dev, rtf=round(gen_t / gen_a, 2) if gen_a else None,
         gen_s=round(gen_t, 1), audio_s=round(gen_a, 1), total_s=round(time.time() - t_all, 1))

def selftest(pref):
    t = time.time(); model, dev = load_model(pref); tl = time.time() - t
    w, sr = synth(model, 'Testing cloned voices.', 1, 0.5, 0.5)            # built-in voice; warms up
    t = time.time(); w, sr = synth(model, 'The war has already been going on for one hundred and fifty years.', 1, 0.5, 0.5)
    g = time.time() - t
    print(f'chatterbox ok device={dev} load={tl:.1f}s rtf={g / (len(w) / sr):.2f}', flush=True)

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if sys.argv[1] == '--selftest': selftest(sys.argv[2] if len(sys.argv) > 2 else 'auto'); sys.exit(0)
    if sys.argv[1] == '--download':
        os.environ.pop('HF_HUB_OFFLINE', None); print(download(sys.argv[2]), flush=True); sys.exit(0)
    try:
        run_job(json.load(open(sys.argv[1])))
    except Exception:
        emit(ev='fatal', error=traceback.format_exc()[-1500:]); sys.exit(1)
