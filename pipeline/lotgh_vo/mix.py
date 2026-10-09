"""Streaming mix: one separation chunk at a time (never the whole file in RAM), then AAC encode + mux."""
import os
import numpy as np, soundfile as sf
from .util import FFMPEG, log, Progress, atomic_json, load_json, h, run

SR = 44100
def db(x): return 10 ** (x / 20)

def _box(x, w):
    if w < 2: return x
    c = np.cumsum(np.concatenate([np.zeros(1, np.float64), x.astype(np.float64)]))
    lo = np.clip(np.arange(len(x)) - w // 2, 0, len(x)); hi = np.clip(lo + w, 0, len(x))
    return ((c[hi] - c[lo]) / w).astype(np.float32)

def _smooth(x, w):
    """Triangular smoothing (two box filters, O(n)) - same shape as the old Hann convolution, ~1000x faster."""
    return np.clip(_box(_box(x, w // 2), w // 2), 0, 1)

SONG_FADE = 0.25        # s, raised-cosine fade at the edges of a song span (inside its 1 s padding)

def song_weight(spans, t0, n):
    """0..1 per sample from t0: 1 inside song spans, smooth fades at the edges, 0 elsewhere (None if no overlap)."""
    w = None; f = int(SONG_FADE * SR)
    for x, y in spans:
        a, b = int(round((x - t0) * SR)), int(round((y - t0) * SR))
        if b <= 0 or a >= n: continue
        if w is None: w = np.zeros(n, np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f, dtype=np.float32))
        seg = np.ones(b - a, np.float32)
        k = min(f, (b - a) // 2); seg[:k] = ramp[:k]; seg[len(seg) - k:] = ramp[:k][::-1]
        lo, hi = max(0, a), min(n, b)
        w[lo:hi] = np.maximum(w[lo:hi], seg[lo - a:hi - a])
    return w

def mix_sig(a, chunks, place):
    songs = getattr(a, 'song_spans', None) or []
    return h('mix3', 'off' if a.jp_db is None else a.jp_db, *(['songs', songs] if songs else []), a.duck_db, a.tts_db, a.duck_hold, a.duck_smooth, len(chunks), a.sep_chunk, a.abitrate,
             *[f'{p[0]:.3f}:{os.path.basename(p[2])}' for p in place])

def final_mix_path(wd): return os.path.join(wd, 'mix', 'english_mix.m4a')

def mix_cached(a, chunks, place, wd):
    m4a = final_mix_path(wd)
    return os.path.exists(m4a) and (load_json(m4a + '.json') or {}).get('sig') == mix_sig(a, chunks, place) \
        and 'force_mix' not in a.force

def run_mix(a, chunks, place, wd):
    """chunks: [(t0,t1,vocals,no_vocals)], place: [[start,dur,wav,i]]. Writes float32 chunk mixes + peak list."""
    mdir = os.path.join(wd, 'mix'); os.makedirs(mdir, exist_ok=True)
    sig = mix_sig(a, chunks, place)
    pr = Progress('mix', len(chunks), every=10)
    outs, peaks = [], []
    ctx = 1.0                                                 # context so ducking is seamless across chunk edges
    for k, (t0, t1, vp, bp) in enumerate(chunks):
        out = os.path.join(mdir, f'chunk_{k:05d}.wav'); meta = out + '.json'
        m = load_json(meta)
        if m and m.get('sig') == sig and os.path.exists(out) and 'force_mix' not in a.force:
            outs.append(out); peaks.append(m['peak']); pr.step(); continue
        jp_off = a.jp_db is None                              # Japanese voices left out: background + English only
        bg, _ = sf.read(bp, dtype='float32')
        sw = song_weight(getattr(a, 'song_spans', None) or [], t0, min(sf.info(vp).frames, len(bg)))
        if jp_off and sw is None: voc = None; n = min(sf.info(vp).frames, len(bg)); bg = bg[:n]
        else:
            voc, _ = sf.read(vp, dtype='float32')
            n = min(len(voc), len(bg)); voc, bg = voc[:n], bg[:n]
        w0 = t0 - ctx; N = n + int(2 * ctx * SR)              # working window incl. context
        eng = np.zeros(N, np.float32); mask = np.zeros(N, np.float32)
        for start, d, wav, _ in place:
            if start + d < w0 or start > t1 + ctx: continue
            x, _ = sf.read(wav, dtype='float32')
            if x.ndim > 1: x = x.mean(1)
            i0 = int(round((start - w0) * SR)); a0 = max(0, -i0); i0 = max(0, i0)
            i1 = min(N, i0 + len(x) - a0)
            if i1 <= i0: continue
            eng[i0:i1] += x[a0:a0 + i1 - i0] * db(a.tts_db); mask[i0:i1] = 1
        c0 = int(ctx * SR)
        if jp_off and sw is None:
            mix = bg + eng[c0:c0 + n, None]
        else:
            if jp_off: jp_gain = np.zeros(n, np.float32)
            else:
                # duck envelope: pre-roll (attack) and smoothing (release) around English speech
                att, rel = int(a.duck_hold * SR), int(a.duck_smooth * SR)
                env = _smooth(np.pad(mask, (att, 0))[:N], rel)
                jp_gain = db(a.jp_db) * db(a.duck_db * env[c0:c0 + n])
            if sw is not None: sw = sw[:n]; jp_gain = sw + (1 - sw) * jp_gain    # songs: vocals at 0 dB, no ducking
            mix = bg + voc * jp_gain[:, None] + eng[c0:c0 + n, None]
        peak = float(np.abs(mix).max()) if n else 0.0
        sf.write(out + '.tmp.wav', mix, SR, subtype='FLOAT'); os.replace(out + '.tmp.wav', out)
        atomic_json(meta, dict(sig=sig, peak=peak))
        if a.keep_stems:
            sf.write(os.path.join(mdir, f'english_{k:05d}.wav'), eng[c0:c0 + n], SR, subtype='PCM_16')
        outs.append(out); peaks.append(peak); pr.step()
    gpk = max(peaks) if peaks else 1.0
    gain_db = min(0.0, 20 * np.log10(db(-1) / gpk)) if gpk > 0 else 0.0
    log('mix', f'{len(outs)} chunks, global peak {20 * np.log10(max(gpk, 1e-9)):.1f} dBFS -> gain {gain_db:+.1f} dB')
    return outs, gain_db

def encode_mix(a, chunks, place, mix_chunks, gain_db, wd):
    """Concat float chunk mixes -> AAC with the global gain. Chunk WAVs are deleted afterwards unless --keep-temp."""
    lst = os.path.join(wd, 'mix', 'concat.txt'); m4a = final_mix_path(wd)
    with open(lst, 'w') as f:
        for p in mix_chunks: f.write(f"file '{os.path.abspath(p)}'\n")
    log('mux', 'encoding English mix (AAC)')
    run([FFMPEG, '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-af', f'volume={gain_db:.2f}dB',
         '-c:a', 'aac', '-b:a', a.abitrate, m4a + '.tmp.m4a'])
    os.replace(m4a + '.tmp.m4a', m4a); atomic_json(m4a + '.json', dict(sig=mix_sig(a, chunks, place), gain_db=gain_db))
    if not a.keep_temp:
        for p in mix_chunks:
            for q in (p, p + '.json'):
                if os.path.exists(q): os.remove(q)
    return m4a

def mux(a, video, info, wd, output):
    m4a = final_mix_path(wd)
    vargs = ['-c:v', 'copy']
    if a.video_codec == 'h264':
        enc = 'h264_videotoolbox' if _has_encoder('h264_videotoolbox') else 'libx264'
        vargs = ['-c:v', enc] + (['-b:v', '4M'] if enc == 'h264_videotoolbox' else ['-crf', '20', '-preset', 'medium'])
        log('mux', f're-encoding video with {enc}')
    cmd = [FFMPEG, '-v', 'error', '-y', '-i', video, '-i', m4a, '-map', '0:v:0', '-map', '1:a:0']
    if info['has_audio']: cmd += ['-map', '0:a:0']
    cmd += vargs + ['-c:a:0', 'copy']
    if info['has_audio']: cmd += ['-c:a:1', 'aac', '-b:a:1', a.abitrate]
    cmd += ['-metadata:s:a:0', 'language=eng', '-metadata:s:a:0', 'title=English voice-over', '-disposition:a:0', 'default']
    if info['has_audio']:
        cmd += ['-metadata:s:a:1', 'language=jpn', '-metadata:s:a:1', 'title=Japanese original', '-disposition:a:1', '0']
    if output.lower().endswith(('.mp4', '.m4v', '.mov')): cmd += ['-movflags', '+faststart']
    tmp = output + '.part' + os.path.splitext(output)[1]
    cmd += ['-shortest', tmp]
    log('mux', f'writing {output}')
    run(cmd); os.replace(tmp, output)

def _has_encoder(name):
    import subprocess
    r = subprocess.run([FFMPEG, '-hide_banner', '-encoders'], capture_output=True, text=True)
    return f' {name} ' in r.stdout
