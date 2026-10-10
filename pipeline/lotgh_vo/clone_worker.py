"""Cloned-voice worker. Runs ONLY in the optional Chatterbox venv (never imported by the base pipeline).

python -m lotgh_vo.clone_worker JOB.json      -> speaker grouping, reference clips, one Chatterbox wav per line
python -m lotgh_vo.clone_worker --selftest [auto|mps|cpu] [german_v3]  -> load the model, synthesise, print device + speed
python -m lotgh_vo.clone_worker --download DEST     -> English model files (~3.2 GB) into DEST
python -m lotgh_vo.clone_worker --download-de DEST  -> extra German-accent files (Multilingual V3, ~2.1 GB) into DEST

Custom voices (job['custom'] = {role: dict(name, path, accent)}, role narrator or dialogue): every line of that role
uses the given reference clip (e.g. an imported recording) instead of one built from the video, optionally with an
accent; fallback per line is the same as for the accent: custom voice with accent -> same voice without -> Kokoro.

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
# The model is Chatterbox Multilingual V3. Upstream chatterbox (commit 3f35dfc8) runs it with repetition_penalty 1.2,
# without the alignment-stream "hallucination" check (which cut lines short) and drops the final speech token's
# ~40 ms of noise; those three changes are applied here (load_accent / synth) so the installed chatterbox-tts 0.1.7
# package stays as it is.
DE_T3 = 't3_mtl23ls_v3.safetensors'
DE_FILES = {DE_T3: 2143989928, 'grapheme_mtl_merged_expanded_v1.json': 69989}
ACCENTS = {'german_v3': 'de'}
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
    """Multilingual V3 T3 + tokenizer on top of the already-loaded English model's voice encoder and S3Gen."""
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
    t3 = T3(T3Config.multilingual()); st = load_file(os.path.join(d, DE_T3))
    if 'model' in st.keys(): st = st['model'][0]
    t3.load_state_dict(st); t3.to(dev).eval(); del st
    # upstream 3f35dfc8 removed the alignment-stream analyzer; T3.inference only builds it when hp.is_multilingual,
    # which is used for nothing else, so this instance's config reports False.
    cfg = t3.hp
    cfg.__class__ = type('T3ConfigNoAnalyzer', (type(cfg),), {'is_multilingual': property(lambda self: False)})
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
    if lang: w = model.generate(speakable(text), language_id=lang, exaggeration=exaggeration, cfg_weight=cfg, repetition_penalty=1.2)
    else: w = model.generate(speakable(text), exaggeration=exaggeration, cfg_weight=cfg)
    w = w.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if lang and len(w) > 2 * 960: w = w[:-960]   # upstream: drop the last speech token (24 kHz / 25 tokens/s = 960 samples)
    return w, model.sr

def bad_len(d, text): return d < 0.25 or d > 2.5 * expected_len(text)

def synth_checked(model, text, seed, exaggeration, cfg, lang=None, label=''):
    """One retry with another seed when the length is implausible (silence or run-on babble)."""
    w, sr = synth(model, text, seed, exaggeration, cfg, lang); d = len(w) / sr
    if bad_len(d, text):
        say(f'{label}: suspicious length {d:.1f}s, retrying with another seed')
        w, sr = synth(model, text, seed + 7919, exaggeration, cfg, lang); d = len(w) / sr
        if bad_len(d, text): raise RuntimeError(f'bad output length {d:.1f}s')
    return w, sr

# ------------------------------------------------------------------ batched generation
BATCH_CHARS = 600        # max total subtitle characters in one batch (bounds the KV cache: longest line x batch size)

def can_batch(model):
    """True when the model is the Llama-based T3 with learned position embeddings that synth_batch mirrors."""
    t3 = getattr(model, 't3', None)
    return (t3 is not None and not getattr(t3, 'is_gpt', True) and getattr(t3.hp, 'input_pos_emb', None) == 'learned'
            and hasattr(t3, 'tfmr') and hasattr(model, 's3gen') and getattr(model, 'conds', None) is not None)

def is_oom(e):
    s = str(e).lower()
    return 'out of memory' in s or 'failed to allocate' in s or 'mps backend' in s and 'memory' in s

def free_gpu():
    import gc, torch
    gc.collect()
    if torch.backends.mps.is_available():
        try: torch.mps.empty_cache()
        except Exception: pass

def synth_batch(model, texts, seeds, exaggeration, cfg, lang=None, temperature=0.8, min_p=0.05, top_p=1.0,
                repetition_penalty=1.2, max_new_tokens=1000):
    """Several lines of one voice (model.conds already prepared) in one T3 pass; S3Gen then decodes each line.

    Mirrors ChatterboxTTS.generate / ChatterboxMultilingualTTS.generate + T3.inference of chatterbox-tts 0.1.7 with the
    settings synth() uses (CFG pair per line: text row + text-zeroed row; repetition penalty 1.2; temperature 0.8;
    min_p 0.05; top_p 1.0; up to 1000 speech tokens; English drops tokens >= 6561; German trims the last token).
    Differences: rows are left-padded with an attention mask and per-row positions, each line samples from its own
    CPU generator seeded with its seed, and S3Gen is reseeded per line, so a line's output depends only on its seed,
    not on the batch it ran in, but is not bit-identical to the per-line path (another random stream). The
    alignment-stream analyzer is not used (it is off for both models here). Returns [(wav float32, sr)] in order."""
    import torch, torch.nn.functional as F
    from transformers.generation.logits_process import MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    if not cfg or cfg <= 0: raise ValueError('synth_batch needs cfg > 0')
    punc_norm = sys.modules[type(model).__module__].punc_norm
    t3 = model.t3; hp = t3.hp; dev = t3.device; c = model.conds.t3
    if float(exaggeration) != float(c.emotion_adv[0, 0, 0]): raise ValueError('reference prepared with another exaggeration')
    B = len(texts)
    with torch.inference_mode():
        toks = []
        for t in texts:
            tt = (model.tokenizer.text_to_tokens(punc_norm(speakable(t)), language_id=lang.lower()) if lang
                  else model.tokenizer.text_to_tokens(punc_norm(speakable(t)))).view(-1).to(dev)
            toks.append(F.pad(F.pad(tt, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token))
        cond = t3.prepare_conditioning(c)[0]                                          # (len_cond, D)
        sos = torch.tensor([[hp.start_speech_token]], device=dev)
        bos = (t3.speech_emb(sos) + t3.speech_pos_emb.get_fixed_embedding(0))[0]     # (1, D)
        rows = []
        for tt in toks:
            pe = t3.text_pos_emb(tt[None])                                          # (L, D)
            te = t3.text_emb(tt[None])[0] + pe
            # T3.inference: prepare_input_embeds appends one start-of-speech embedding, then another BOS is added
            rows += [torch.cat([cond, te, bos, bos]), torch.cat([cond, pe, bos, bos])]   # CFG: text row, text-zeroed row
        L = max(r.size(0) for r in rows); R = len(rows)
        x = torch.zeros(R, L, rows[0].size(1), dtype=rows[0].dtype, device=dev)
        mask = torch.zeros(R, L, dtype=torch.long, device=dev)
        for k, r in enumerate(rows): x[k, L - r.size(0):] = r; mask[k, L - r.size(0):] = 1
        pos = (mask.cumsum(1) - 1).clamp(min=0); valid = mask.bool()
        # custom 4D masks (True = may attend): causal over real tokens; a padding position sees only itself, so no
        # attention row is empty (an empty row gives NaN with SDPA on MPS, which then leaks into the real rows)
        eye = torch.eye(L, dtype=torch.bool, device=dev)
        m4 = (torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev)) & valid[:, None, :]) | eye
        out = t3.tfmr(inputs_embeds=x, attention_mask=m4[:, None], position_ids=pos, use_cache=True, return_dict=True)
        past = out.past_key_values; pos = pos[:, -1:]
        rep_p = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        minp = MinPLogitsWarper(min_p=min_p); topp = TopPLogitsWarper(top_p=top_p)
        gens = [torch.Generator().manual_seed(int(sd)) for sd in seeds]
        active = list(range(B)); pred = [[] for _ in range(B)]
        ids = torch.full((B, 1), hp.start_speech_token, dtype=torch.long, device=dev)
        for i in range(max_new_tokens):
            lg = t3.speech_head(out.last_hidden_state[:, -1, :])                      # (2*active, V)
            cnd, unc = lg[0::2], lg[1::2]
            lg = cnd + cfg * (cnd - unc)
            lg = rep_p(ids, lg)
            if temperature != 1.0: lg = lg / temperature
            lg = topp(ids, minp(ids, lg))
            probs = torch.softmax(lg.float(), dim=-1).cpu()
            nxt = torch.cat([torch.multinomial(probs[k], 1, generator=gens[b]) for k, b in enumerate(active)])  # (A,)
            keep = []
            for k, b in enumerate(active):
                pred[b].append(int(nxt[k]))
                if int(nxt[k]) != hp.stop_speech_token: keep.append(k)
            if not keep: break
            nxt = nxt.to(dev)
            if len(keep) < len(active):                                                 # drop finished lines
                kk = torch.tensor(keep, device=dev); rk = torch.stack([2 * kk, 2 * kk + 1], 1).view(-1)
                past.batch_select_indices(rk); valid = valid[rk]; pos = pos[rk]; ids = ids[kk]; nxt = nxt[kk]
                active = [active[k] for k in keep]
            ids = torch.cat([ids, nxt[:, None]], 1)
            emb = t3.speech_emb(nxt[:, None]) + t3.speech_pos_emb.get_fixed_embedding(i + 1)   # (A, 1, D)
            emb = emb.repeat_interleave(2, 0)
            valid = torch.cat([valid, torch.ones_like(valid[:, :1])], 1); pos = pos + 1
            out = t3.tfmr(inputs_embeds=emb, attention_mask=valid[:, None, None, :], position_ids=pos, past_key_values=past,
                          use_cache=True, return_dict=True)
            past = out.past_key_values
        del past, out
        res = []
        for b in range(B):
            st = drop_invalid_tokens(torch.tensor(pred[b], dtype=torch.long))
            if not lang: st = st[st < 6561]
            torch.manual_seed(int(seeds[b]))
            wav, _ = model.s3gen.inference(speech_tokens=st.to(dev), ref_dict=model.conds.gen)
            w = model.watermarker.apply_watermark(wav.squeeze(0).detach().cpu().numpy(), sample_rate=model.sr)
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if lang and len(w) > 2 * 960: w = w[:-960]
            res.append((w, model.sr))
    return res

def make_batches(group, size, max_chars=BATCH_CHARS):
    """Lines of one voice -> batches of up to `size` lines and `max_chars` characters, similar lengths together."""
    out, cur, n = [], [], 0
    for L in sorted(group, key=lambda L: len(L['text'])):
        if cur and (len(cur) >= size or n + len(L['text']) > max_chars): out.append(cur); cur, n = [], 0
        cur.append(L); n += len(L['text'])
    if cur: out.append(cur)
    return out

def custom_refs(job):
    """{role: ref dict} for the user's custom voices; a missing or unreadable file is logged and skipped."""
    import soundfile as sf
    out = {}
    for role, v in (job.get('custom') or {}).items():
        try:
            info = sf.info(v['path'])
            with open(v['path'], 'rb') as f: hsh = 'cv' + hashlib.sha1(f.read()).hexdigest()[:14]
            acc = v.get('accent') or 'original'
            out[role] = dict(path=v['path'], hash=hsh, dur=round(info.duration, 2), lines=[], snr=0, custom=v.get('name') or role,
                             accent=acc if acc in ACCENTS else 'original')
        except Exception as e:
            say(f"custom {role} voice {v.get('name')!r} could not be read ({str(e)[:160]}); using the voices from the video")
    return out

def run_job(job):
    import soundfile as sf
    t_all = time.time()
    voc = Vocals(job['chunks']); refdir = os.path.join(job['dir'], 'refs'); gdir = os.path.join(job['dir'], 'gen')
    os.makedirs(refdir, exist_ok=True); os.makedirs(gdir, exist_ok=True)
    emit(ev='phase', name='load'); t = time.time()
    model, dev = load_model(job.get('device', 'auto'))
    say(f'Chatterbox loaded on {dev.upper()} in {time.time() - t:.0f}s')
    accent = job.get('accent') or 'original'; lang = ACCENTS.get(accent); amodels = {}
    custom = custom_refs(job)
    for role, r in custom.items():
        say(f"{role} lines use the custom voice {r['custom']!r} ({r['dur']:.1f}s reference"
            + (f", {r['accent']} accent" if r['accent'] != 'original' else '') + ')')
    for acc_name in sorted(({accent} if lang else set()) | {r['accent'] for r in custom.values() if r['accent'] != 'original'}):
        try:
            t = time.time(); amodels[acc_name] = load_accent(model, dev)
            say(f"{acc_name.replace('_', ' ').capitalize()} accent model loaded in {time.time() - t:.0f}s")
        except Exception as e:
            say(f'{acc_name} accent model could not be loaded ({str(e)[:200]}); using the original-accent clones')
            if acc_name == accent: lang = None
    lines = job['lines']
    emit(ev='phase', name='speakers'); t = time.time()
    nar = [L for L in lines if L['role'] == 'narrator']; dlg = [L for L in lines if L['role'] == 'dialogue']
    need_fp = (nar if 'narrator' not in custom else []) + (dlg if 'dialogue' not in custom else [])
    for L in lines: L.setdefault('emb', None); L.setdefault('speech', 0.0)
    if need_fp: fingerprint(model, need_fp, voc, Vocals(job['chunks'], stem=3))
    refs = {}; cents = {}
    if 'dialogue' in custom:
        n = custom['dialogue']['custom']; refs[n] = custom['dialogue']
        for L in dlg: L['spk'] = n; L['why'] = 'custom dialogue voice'
    else:
        cents = cluster(dlg, job.get('cluster_sim', 0.80)); assign(dlg, cents)
    if 'narrator' in custom:
        n = custom['narrator']['custom']; refs[n] = custom['narrator']
        for L in nar: L['spk'] = n; L['why'] = 'custom narrator voice'
    else:
        for L in nar: L['spk'] = 'narrator'; L['why'] = 'cyan subtitle'
        if nar:
            e = [L['emb'] for L in nar if L['emb'] is not None]
            refs['narrator'] = build_ref(nar, unit(np.mean(e, 0)) if e else None, refdir)
    for n, c in cents.items(): refs[n] = build_ref([L for L in dlg if L['spk'] == n], c, refdir)
    spk_summary = {n: dict(lines=sum(1 for L in lines if L['spk'] == n), ref=r) for n, r in refs.items()}
    for n, s in spk_summary.items():
        r = s['ref'] or {}
        if r.get('custom'):
            say(f"{n}: {s['lines']} lines, custom voice ({r.get('dur', 0):.1f}s reference recording)"); continue
        say(f"{n}: {s['lines']} lines, reference {r.get('dur', 0):.1f}s from lines {[i + 1 for i in r.get('lines', [])]}"
            f" (voice {r.get('snr', 0):.0f} dB over background)"
            + ('' if r.get('hash') else ' -> too little clean speech, these lines use Kokoro'))
    say(f'speaker grouping took {time.time() - t:.0f}s ({len(cents)} dialogue speakers + narrator'
        + (', custom: ' + ', '.join(f"{k}={r['custom']}" for k, r in custom.items()) if custom else '') + ')')
    emit(ev='speakers', speakers={n: dict(lines=s['lines'], ref_dur=(s['ref'] or {}).get('dur', 0),
                                          ref_lines=(s['ref'] or {}).get('lines', []), ref_hash=(s['ref'] or {}).get('hash'))
                                  for n, s in spk_summary.items()},
         lines={L['i']: dict(spk=L['spk'], why=L['why'], speech=L['speech']) for L in lines})
    # generation, grouped by speaker so each reference is prepared once
    res = {}; todo = []
    def line_accent(r):
        """Accent of one line: a custom voice's own accent, else the job's; None when original or not loaded."""
        a = r['accent'] if r.get('custom') else accent
        return a if ACCENTS.get(a) and a in amodels else None
    for L in lines:
        r = refs.get(L['spk']) or {}
        if not r.get('hash'): res[L['i']] = dict(error=f"no reference for {L['spk']}"); continue
        L['key'] = gen_key(job, L['text'], r['hash'], None); L['out'] = os.path.join(gdir, f"{L['key']}.wav")
        L['acc'] = line_accent(r)
        if L['acc']:
            L['akey'] = gen_key(job, L['text'], r['hash'], L['acc']); L['aout'] = os.path.join(gdir, f"{L['akey']}.wav")
            if os.path.exists(L['aout']): res[L['i']] = dict(path=L['aout'], spk=L['spk'], ref=r['hash'], accent=L['acc'], cached=True); continue
        elif os.path.exists(L['out']): res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'], cached=True); continue
        todo.append(L)
    emit(ev='gen_start', total=len(todo), cached=len(res) - sum(1 for v in res.values() if 'error' in v))
    cur = {}; gen_t = 0.0; gen_a = 0.0; n_acc = n_acc_fb = 0
    def prep(m, r):
        if cur.get(id(m)) != r['hash']: m.prepare_conditionals(r['path'], exaggeration=job['exaggeration']); cur[id(m)] = r['hash']
    def save(w, sr, out):
        sf.write(out + '.tmp.wav', w, sr, subtype='FLOAT'); os.replace(out + '.tmp.wav', out)
    def seed_of(L, acc): return (job['seed'] + int((L['akey'] if acc else L['key'])[:8], 16)) % 2 ** 31
    def gen_line(L, pre=None):
        """The per-line path. pre = (wav, sr) from a batch for this line's first model (accent if any), already
        length-checked; without it the line is synthesised here (with the usual one retry)."""
        nonlocal gen_t, gen_a, n_acc, n_acc_fb
        r = refs[L['spk']]; t = time.time(); d = 0.0; d_pre = 0.0
        if L['acc']:          # accent first; on any failure the original-accent clone, then (in the parent) Kokoro
            try:
                amodel = amodels[L['acc']]
                if pre: w, sr = pre; d_pre = len(w) / sr
                else:
                    prep(amodel, r)
                    w, sr = synth_checked(amodel, L['text'], seed_of(L, True), job['exaggeration'],
                                          job['cfg'], ACCENTS[L['acc']], f"line {L['i'] + 1} ({L['acc']})")
                d = len(w) / sr; save(w, sr, L['aout']); n_acc += 1
                res[L['i']] = dict(path=L['aout'], spk=L['spk'], ref=r['hash'], accent=L['acc'])
            except Exception as e:
                n_acc_fb += 1
                say(f"line {L['i'] + 1}: {L['acc']} accent failed ({type(e).__name__}: {str(e)[:160]}); using the original-accent clone")
        if L['i'] not in res:
            try:
                if os.path.exists(L['out']):
                    res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'], cached=True)
                elif pre and not L['acc']:
                    w, sr = pre; d_pre = len(w) / sr; save(w, sr, L['out'])
                    res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'])
                else:
                    prep(model, r)
                    w, sr = synth_checked(model, L['text'], seed_of(L, False), job['exaggeration'],
                                          job['cfg'], None, f"line {L['i'] + 1}")
                    d += len(w) / sr; save(w, sr, L['out'])
                    res[L['i']] = dict(path=L['out'], spk=L['spk'], ref=r['hash'])
                if L['acc']: res[L['i']]['accent_fallback'] = True
            except Exception as e:
                res[L['i']] = dict(error=str(e)[:300]); say(f"line {L['i'] + 1}: Chatterbox failed: {str(e)[:200]}")
        if d > d_pre: gen_t += time.time() - t; gen_a += d - d_pre        # batched audio is counted with its batch
        emit(ev='gen_step', done=L['i'], rtf=round(gen_t / gen_a, 2) if gen_a else None)
    # Lines of one voice and model are generated in batches (job['batch'] lines, BATCH_CHARS characters at most);
    # each batched line still gets the length check, and any line that fails it, or a batch that fails, goes through
    # the per-line path above. Out of GPU memory: the batch is split and the batch size halved for the rest.
    size = max(1, int(job.get('batch') or 1)); nb = nb_lines = nb_rejected = 0
    groups = {}
    for L in todo: groups.setdefault((L['spk'], L['acc'] or ''), []).append(L)
    for (spk, acc), group in sorted(groups.items(), key=lambda kv: min(L['s'] for L in kv[1])):
        m = amodels[acc] if acc else model
        if size < 2 or not can_batch(m) or not job.get('cfg'):
            for L in sorted(group, key=lambda L: L['s']): gen_line(L)
            continue
        queue = make_batches(group, size)
        while queue:
            bt = queue.pop(0)
            if len(bt) < 2: gen_line(bt[0]); continue
            t = time.time()
            try:
                prep(m, refs[spk])
                outs = synth_batch(m, [L['text'] for L in bt], [seed_of(L, bool(acc)) for L in bt], job['exaggeration'],
                                   job['cfg'], ACCENTS.get(acc) if acc else None,
                                   # S3 speech tokens are 25/s: past 2.5x the expected length a line fails bad_len anyway,
                                   # so a runaway line is cut there instead of holding the whole batch to 1000 tokens
                                   max_new_tokens=min(1000, int(25 * (2.5 * max(expected_len(L['text']) for L in bt) + 1))))
            except Exception as e:
                free_gpu()
                if is_oom(e) and len(bt) > 1:
                    size = max(1, len(bt) // 2)
                    say(f'batch of {len(bt)} lines ran out of GPU memory; batch size now {size}')
                    rest = bt + [L for q in queue for L in q]
                    queue = make_batches(rest, size) if size > 1 else [[L] for L in rest]
                    continue
                say(f'batch of {len(bt)} lines failed ({type(e).__name__}: {str(e)[:160]}); generating them one by one')
                for L in bt: gen_line(L)
                continue
            nb += 1; ok_a = 0.0; pres = {}
            for L, (w, sr) in zip(bt, outs):
                dd = len(w) / sr
                if bad_len(dd, L['text']):
                    nb_rejected += 1
                    say(f"line {L['i'] + 1}{' (' + acc + ')' if acc else ''}: batched output {dd:.1f}s fails the length check; generating it on its own")
                else: pres[L['i']] = (w, sr); ok_a += dd
            del outs; free_gpu()      # hand the batch's KV-cache blocks back; the MPS cache otherwise grows into swap
            gen_t += time.time() - t; gen_a += ok_a; nb_lines += len(pres)
            for L in bt: gen_line(L, pres.get(L['i']))
    if nb: say(f'batched generation: {nb_lines} lines in {nb} batches (batch size up to {int(job.get("batch") or 1)}), '
               f'{nb_rejected} redone one by one')
    if any(L.get('acc') for L in lines): say(f"{'/'.join(sorted({L['acc'] for L in lines if L.get('acc')}))} accent: {n_acc} lines generated, {sum(1 for v in res.values() if v.get('cached') and v.get('accent'))}"
                 f' from cache, {n_acc_fb} fell back to the original accent')
    for L in lines:
        if L['i'] in res and 'path' in res[L['i']] and L.get('spk') in {r['custom'] for r in custom.values()}: res[L['i']]['custom'] = L['spk']
    emit(ev='done', results=res, device=dev, accent=accent if lang else 'original', rtf=round(gen_t / gen_a, 2) if gen_a else None,
         gen_s=round(gen_t, 1), audio_s=round(gen_a, 1), total_s=round(time.time() - t_all, 1))

def selftest(pref, accent=None):
    t = time.time(); model, dev = load_model(pref); tl = time.time() - t
    lang = None
    if accent:
        lang = ACCENTS.get(accent, 'de'); t = time.time(); m = load_accent(model, dev); tl += time.time() - t
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
