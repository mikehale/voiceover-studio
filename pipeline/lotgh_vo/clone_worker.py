"""Cloned-voice worker. Runs ONLY in the optional Chatterbox venv (never imported by the base pipeline).

python -m lotgh_vo.clone_worker JOB.json      -> speaker grouping, reference clips, one Chatterbox wav per line
python -m lotgh_vo.clone_worker --selftest [auto|mps|cpu] [de]  -> load the model, synthesise, print device + speed
python -m lotgh_vo.clone_worker --download DEST     -> English model files (~3.2 GB) into DEST
python -m lotgh_vo.clone_worker --download-de DEST  -> extra German-accent files (~2.1 GB) into DEST

Talks to the parent (lotgh_vo.clone) through stdout lines that start with 'CLONE ' followed by JSON.
"""
import hashlib, json, os, re, shutil, sys, time, traceback
import numpy as np

REPO_ID = 'ResembleAI/chatterbox'
REVISION = '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'   # pinned model snapshot (~3.2 GB)
MODEL_FILES = ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']
# German accent = Chatterbox Multilingual with language_id='de' reading the English text. Only its text-to-speech-token
# model and tokenizer are extra: the voice encoder and S3Gen weights are identical to the English files above
# (checked tensor by tensor), so they are shared instead of downloading ve.pt / s3gen.pt again.
DE_FILES = {'t3_mtl23ls_v2.safetensors': 2143989752, 'grapheme_mtl_merged_expanded_v1.json': 69989}
ACCENTS = {'german': 'de'}
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

def download_de(dest):
    from huggingface_hub import hf_hub_download
    for f in DE_FILES: hf_hub_download(REPO_ID, f, revision=REVISION, local_dir=dest)
    shutil.rmtree(os.path.join(dest, '.cache'), ignore_errors=True)
    bad = [f for f, n in DE_FILES.items() if not os.path.exists(os.path.join(dest, f)) or os.path.getsize(os.path.join(dest, f)) != n]
    if bad: raise RuntimeError(f'German-accent files missing or incomplete after download: {bad}')
    return dest

def de_dir():
    d = os.environ.get('LOTGH_CLONE_MODEL')
    if d and all(os.path.exists(os.path.join(d, f)) for f in DE_FILES): return d
    from huggingface_hub import hf_hub_download
    p = None
    for f in DE_FILES: p = hf_hub_download(REPO_ID, f, revision=REVISION)
    return os.path.dirname(p)

def load_accent(model, dev):
    """Multilingual T3 + tokenizer on top of the already-loaded English model's voice encoder and S3Gen."""
    from safetensors.torch import load_file
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import tokenizer as tkm
    # Chinese helpers are not needed for German; without this the tokenizer tries to fetch a Cangjie table into the
    # model folder and to import a Chinese word segmenter.
    tkm.ChineseCangjieConverter._load_cangjie_mapping = lambda self, model_dir=None: None
    tkm.ChineseCangjieConverter._init_segmenter = lambda self: None
    d = de_dir()
    t3 = T3(T3Config.multilingual()); st = load_file(os.path.join(d, 't3_mtl23ls_v2.safetensors'))
    if 'model' in st.keys(): st = st['model'][0]
    t3.load_state_dict(st); t3.to(dev).eval(); del st
    tok = tkm.MTLTokenizer(os.path.join(d, 'grapheme_mtl_merged_expanded_v1.json'))
    return ChatterboxMultilingualTTS(t3, model.s3gen, model.ve, tok, dev)

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
def gen_key(job, text, ref_hash, accent):
    """Cache key of one generated line. The original accent keeps the v0.2.0 key so existing caches stay valid."""
    parts = [job['version'], text, ref_hash, job['seed'], job['exaggeration'], job['cfg']] + ([accent] if accent else [])
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:16]

def expected_len(text): return 0.5 + 0.065 * len(text)

def speakable(text):
    """Subtitle text -> model text. Chatterbox turns '...' into ', ', so a continuation line like
    '...have been seeking' starts with a comma and lower-case word and often comes out garbled."""
    t = re.sub(r'^[\s.,;:\u2026\-\u2013\u2014]+', '', text.strip())
    t = re.sub(r'(\.\.\.|\u2026)\s*$', ',', t).strip()
    if not t: return text
    return t[0].upper() + t[1:]

def synth(model, text, seed, exaggeration, cfg, lang=None):
    import torch
    torch.manual_seed(seed)
    if lang: w = model.generate(speakable(text), language_id=lang, exaggeration=exaggeration, cfg_weight=cfg)
    else: w = model.generate(speakable(text), exaggeration=exaggeration, cfg_weight=cfg)
    return w.squeeze(0).detach().cpu().numpy().astype(np.float32), model.sr

def bad_len(d, text): return d < 0.25 or d > 2.5 * expected_len(text)

def synth_checked(model, text, seed, exaggeration, cfg, lang=None, label=''):
    """One retry with another seed when the length is implausible (silence or run-on babble)."""
    w, sr = synth(model, text, seed, exaggeration, cfg, lang); d = len(w) / sr
    if bad_len(d, text):
        say(f'{label}: suspicious length {d:.1f}s, retrying with another seed')
        w, sr = synth(model, text, seed + 7919, exaggeration, cfg, lang); d = len(w) / sr
        if bad_len(d, text): raise RuntimeError(f'bad output length {d:.1f}s')
    return w, sr

def run_job(job):
    import soundfile as sf
    t_all = time.time()
    voc = Vocals(job['chunks']); refdir = os.path.join(job['dir'], 'refs'); gdir = os.path.join(job['dir'], 'gen')
    os.makedirs(refdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)
    emit(ev='phase', name='load'); t = time.time()
    model, dev = load_model(job.get('device', 'auto'))
    say(f'Chatterbox loaded on {dev.upper()} in {time.time() - t:.0f}s')
    accent = job.get('accent') or 'original'; lang = ACCENTS.get(accent); amodel = None
    if lang:
        try:
            t = time.time(); amodel = load_accent(model, dev)
            say(f'{accent.capitalize()} accent model loaded in {time.time() - t:.0f}s')
        except Exception as e:
            say(f'{accent} accent model could not be loaded ({str(e)[:200]}); using the original-accent clones'); lang = None
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
        L['key'] = gen_key(job, L['text'], r['hash'], None); L['out'] = os.path.join(gdir, f"{L['key']}.wav")
        if lang:
            L['akey'] = gen_key(job, L['text'], r['hash'], accent); L['aout'] = os.path.join(gdir, f"{L['akey']}.wav")
            if os.path.exists(L['aout']): res[L['i']] = dict(path=L['aout'], spk=L['spk'], ref=r['hash'], accent=accent, cached=True); continue
        elif os.path.exists(L['out']): res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'], cached=True); continue
        todo.append(L)
    emit(ev='gen_start', total=len(todo), cached=len(res) - sum(1 for v in res.values() if 'error' in v))
    cur = {}; gen_t = 0.0; gen_a = 0.0; n_acc = n_acc_fb = 0
    def prep(m, r):
        if cur.get(id(m)) != r['hash']: m.prepare_conditionals(r['path'], exaggeration=job['exaggeration']); cur[id(m)] = r['hash']
    def save(w, sr, out):
        sf.write(out + '.tmp.wav', w, sr, subtype='FLOAT'); os.replace(out + '.tmp.wav', out)
    for L in sorted(todo, key=lambda L: (L['spk'], L['s'])):
        r = refs[L['spk']]; t = time.time(); d = 0.0
        if lang:          # accent first; on any failure the original-accent clone, then (in the parent) Kokoro
            try:
                prep(amodel, r)
                w, sr = synth_checked(amodel, L['text'], (job['seed'] + int(L['akey'][:8], 16)) % 2 ** 31, job['exaggeration'],
                                      job['cfg'], lang, f"line {L['i'] + 1} ({accent})")
                d = len(w) / sr; save(w, sr, L['aout']); n_acc += 1
                res[L['i']] = dict(path=L['aout'], spk=L['spk'], ref=r['hash'], accent=accent)
            except Exception as e:
                n_acc_fb += 1
                say(f"line {L['i'] + 1}: {accent} accent failed ({type(e).__name__}: {str(e)[:160]}); using the original-accent clone")
        if L['i'] not in res:
            try:
                if os.path.exists(L['out']):
                    res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'], cached=True)
                else:
                    prep(model, r)
                    w, sr = synth_checked(model, L['text'], (job['seed'] + int(L['key'][:8], 16)) % 2 ** 31, job['exaggeration'],
                                          job['cfg'], None, f"line {L['i'] + 1}")
                    d += len(w) / sr; save(w, sr, L['out'])
                    res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'])
                if lang: res[L['i']]['accent_fallback'] = True
            except Exception as e:
                res[L['i']] = dict(error=str(e)[:300]); say(f"line {L['i'] + 1}: Chatterbox failed: {str(e)[:200]}")
        if d: gen_t += time.time() - t; gen_a += d
        emit(ev='gen_step', done=L['i'], rtf=round(gen_t / gen_a, 2) if gen_a else None)
    if lang: say(f"{accent} accent: {n_acc} lines generated, {sum(1 for v in res.values() if v.get('cached') and v.get('accent'))}"
                 f' from cache, {n_acc_fb} fell back to the original accent')
    emit(ev='done', results=res, device=dev, accent=accent if lang else 'original', rtf=round(gen_t / gen_a, 2) if gen_a else None,
         gen_s=round(gen_t, 1), audio_s=round(gen_a, 1), total_s=round(time.time() - t_all, 1))

def selftest(pref, accent=None):
    t = time.time(); model, dev = load_model(pref); tl = time.time() - t
    lang = None
    if accent:
        lang = ACCENTS.get(accent, accent); t = time.time(); m = load_accent(model, dev); tl += time.time() - t
        m.conds = model.conds                         # built-in voice
    else: m = model
    w, sr = synth(m, 'Testing cloned voices.', 1, 0.5, 0.5, lang)            # warms up
    t = time.time(); w, sr = synth(m, 'The war has already been going on for one hundred and fifty years.', 1, 0.5, 0.5, lang)
    g = time.time() - t
    print(f"chatterbox ok device={dev} load={tl:.1f}s rtf={g / (len(w) / sr):.2f}" + (f' accent={accent}' if accent else ''), flush=True)

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if sys.argv[1] == '--selftest':
        selftest(sys.argv[2] if len(sys.argv) > 2 else 'auto', sys.argv[3] if len(sys.argv) > 3 else None); sys.exit(0)
    if sys.argv[1] == '--download-de':
        os.environ.pop('HF_HUB_OFFLINE', None); print(download_de(sys.argv[2]), flush=True); sys.exit(0)
    if sys.argv[1] == '--download':
        os.environ.pop('HF_HUB_OFFLINE', None); print(download(sys.argv[2]), flush=True); sys.exit(0)
    try:
        run_job(json.load(open(sys.argv[1])))
    except Exception:
        emit(ev='fatal', error=traceback.format_exc()[-1500:]); sys.exit(1)
