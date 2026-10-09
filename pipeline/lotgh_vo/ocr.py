"""Hard-subtitle OCR, parallel over time chunks, cached per chunk.

Per chunk: ffmpeg decodes only the subtitle band at --ocr-fps -> colour-independent text mask (white top-hat +
dark-outline test + component size filter) -> change detection -> one RapidOCR read per stable segment
(two more frames if confidence is low). Chunks are merged, corrected and written as SRT.
"""
import os, re, subprocess, time
import numpy as np, cv2
from rapidfuzz import fuzz
from .textfix import (LATIN, clean, spellfix, fix_caps, quality, load_corrections, apply_corrections,
                      dialogue_nits, caption_nits, color_class)
from .util import FFMPEG, log, Progress, atomic_json, load_json, hwdec_args, ort_session, pick_provider

OCR_VERSION = 3
K_TOP = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_TOL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
K_CLEAN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

def text_mask(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, K_TOP)
    m = (top > 45) & (g > 110)
    mn = cv2.erode(g, K_OUT)                                   # subtitles have a dark outline
    m &= (g.astype(np.int16) - mn) > 60
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    hgt, ar = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    keep[(ar >= 10) & (hgt >= 4) & (hgt <= 45)] = True; keep[0] = False
    return keep[lab].astype(np.uint8)

def change(a, b, min_px=150):
    na, nb = int(a.sum()), int(b.sum())
    if na < min_px and nb < min_px: return 0.0
    da, db = cv2.dilate(a, K_TOL), cv2.dilate(b, K_TOL)
    return int((a & (1 - db)).sum() + (b & (1 - da)).sum()) / max(na, nb, min_px)

class OCR:
    """RapidOCR (PaddleOCR PP-OCRv3 det/rec ONNX models) with sessions rebuilt for thread count / provider."""
    def __init__(self, provider='cpu', threads=1):
        import warnings; warnings.filterwarnings('ignore')
        from rapidocr_onnxruntime import RapidOCR
        self.eng = RapidOCR()
        det, rec = self.eng.text_detector.infer, self.eng.text_recognizer.session
        det.session = ort_session(det.session._model_path, provider, threads)
        rec.session = ort_session(rec.session._model_path, provider, threads)

    def _rec(self, crop):
        # recogniser called directly: RapidOCR.__call__ silently re-enables detection on taller crops
        crop = cv2.copyMakeBorder(crop, 4, 4, 8, 8, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        r = self.eng.text_recognizer([crop])[0]
        if not r: return '', 0.0
        return str(r[0][0]).strip(), float(r[0][1])

    def __call__(self, bgr, mask):
        P = 20
        img = cv2.copyMakeBorder(bgr, P, P, P, P, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        res = self.eng.text_detector(img)[0]
        if res is None or len(res) == 0: return []
        boxes = []
        for box in res:
            b = np.asarray(box, dtype=float) - P
            boxes.append([b[:, 1].min(), b[:, 1].max(), b[:, 0].min(), b[:, 0].max()])
        boxes.sort(key=lambda b: (b[0] + b[1]) / 2)
        hmed = float(np.median([b[1] - b[0] for b in boxes]))
        groups = []                                            # word boxes -> text rows (by box centre)
        for b in boxes:
            c = (b[0] + b[1]) / 2
            if groups and abs(c - np.median([(g[0] + g[1]) / 2 for g in groups[-1]])) < 0.4 * hmed: groups[-1].append(b)
            else: groups.append([b])
        rows = [[min(g[0] for g in G), max(g[1] for g in G), min(g[2] for g in G), max(g[3] for g in G)] for G in groups]
        H, W = mask.shape; hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        cl = bgr.copy(); cl[cv2.dilate(mask, K_CLEAN) == 0] = 0   # background-suppressed variant
        out = []
        for y0, y1, x0, x1 in rows:
            ya, yb, xa, xb = int(max(0, y0 - 3)), int(min(H, y1 + 3)), int(max(0, x0 - 6)), int(min(W, x1 + 6))
            if yb - ya < 8 or xb - xa < 8: continue
            txt, sc = self._rec(bgr[ya:yb, xa:xb])
            if sc < 0.93:
                t2, s2 = self._rec(cl[ya:yb, xa:xb])
                if s2 > sc + 0.02: txt, sc = t2, s2
            if sc < 0.6 or not txt: continue
            if len(LATIN.findall(txt)) / len(txt) < 0.8: continue     # Japanese credits / signs
            txt = spellfix(clean(txt))
            if len(re.findall('[A-Za-z]', txt)) < 2: continue
            mm = mask[ya:yb, xa:xb] > 0
            sat = float(np.median(hsv[ya:yb, xa:xb, 1][mm])) if mm.any() else 255
            val = float(np.median(hsv[ya:yb, xa:xb, 2][mm])) if mm.any() else 0
            col = np.median(bgr[ya:yb, xa:xb][mm], axis=0) if mm.any() else np.zeros(3)
            out.append(dict(text=txt, sat=sat, val=val, score=sc,
                            color='#%02x%02x%02x' % (int(col[2]), int(col[1]), int(col[0]))))
        return out

def classify(rows, cap_sat=30, cap_val=235):
    """Dialogue vs on-screen caption cards (pure white, bright serif: sat<30 & V>=235)."""
    dia, cap, cols = [], [], []
    for r in rows:
        if r['sat'] < cap_sat and r['val'] >= cap_val: cap.append(r['text'])
        else: dia.append(r['text']); cols.append(r['color'])
    score = min([r['score'] for r in rows], default=0.0)
    return fix_caps(' '.join(dia)), ' '.join(cap), score, (cols[0] if cols else '')

def consensus(cands):
    """Best dialogue reading and, independently, best caption reading (a caption may be present in only some
    of the frames read)."""
    best = max(cands, key=lambda c: quality(c[0], c[2]) if c[0] else -1)
    caps = [c for c in cands if c[1]]
    cap = max(caps, key=lambda c: quality(c[1], c[2]))[1] if caps else ''
    return best[0], cap, best[2], best[3]

def merge(items, ratio=88, gap=0.6, same_gap=1.5):
    out = []
    for s, e, txt, sc, col in items:
        r = fuzz.ratio(out[-1][2].lower(), txt.lower()) if out else 0
        if out and ((s - out[-1][1] <= gap and r >= ratio) or (s - out[-1][1] <= same_gap and r >= 95)):
            out[-1][1] = max(out[-1][1], e); out[-1][4].append((sc, txt, col))
        else:
            out.append([s, e, txt, sc, [(sc, txt, col)]])
    return [[s, e, *max(reads, key=lambda x: quality(x[1], x[0]))[1:]] for s, e, _, _, reads in out]

# ---------------- per-chunk worker ----------------
_W = {}
def _init_worker(provider, threads):
    cv2.setNumThreads(1)
    _W['ocr'] = OCR(provider, threads)

def ocr_chunk(job):
    """job: dict(video, t0, t1, fps, y0, y1, width, hw, out, thresh, min_px, min_dur, verify_below)."""
    j = job; ocr = _W['ocr']
    w, hgt = j['width'], j['y1'] - j['y0']
    cmd = [FFMPEG, '-v', 'error', *j['hw'], '-ss', f"{j['t0']:.3f}", '-t', f"{j['t1'] - j['t0']:.3f}", '-i', j['video'],
           '-an', '-sn', '-vf', f"fps={j['fps']},crop={w}:{hgt}:0:{j['y0']}", '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * hgt * 3 * 8)
    n = w * hgt * 3; i = 0; segs = []; prev = None; cur = None
    def close(seg, end):
        seg['end'] = end
        if seg['end'] - seg['start'] < j['min_dur'] - 1e-6 or np.median(seg['counts']) < j['min_px']:
            seg['frames'] = []
        segs.append(seg)
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        t = j['t0'] + i / j['fps']; i += 1
        fr = np.frombuffer(b, np.uint8).reshape(hgt, w, 3); m = text_mask(fr)
        if prev is None or change(prev, m, j['min_px']) > j['thresh']:
            if cur: close(cur, t)
            cur = dict(start=t, frames=[], counts=[])
        cur['counts'].append(int(m.sum())); cur['frames'].append((fr, m))
        if len(cur['frames']) > 24: cur['frames'] = cur['frames'][::2]
        prev = m
    p.wait()
    if p.returncode not in (0, None): raise RuntimeError(f'ffmpeg failed on chunk {j["t0"]:.0f}s')
    if cur: close(cur, min(j['t1'], j['t0'] + i / j['fps']))
    items = []; n_ocr = 0
    for s in segs:
        F = s['frames']
        if not F: continue
        cands = [classify(ocr(*F[len(F) // 2]))]; n_ocr += 1
        if cands[0][2] < j['verify_below'] and len(F) >= 3:
            for k in (len(F) // 4, (3 * len(F)) // 4):
                cands.append(classify(ocr(*F[k]))); n_ocr += 1
        d, c, sc, col = consensus(cands)
        if d or c: items.append(dict(start=round(s['start'], 3), end=round(s['end'], 3), dia=d, cap=c, score=sc, color=col))
    atomic_json(j['out'], dict(version=OCR_VERSION, params=j['params'], items=items, frames=i, ocr_calls=n_ocr))
    return j['out'], n_ocr

def fmt(t):
    ms = int(round(t * 1000)); hh, ms = divmod(ms, 3600000); mm, ms = divmod(ms, 60000); ss, ms = divmod(ms, 1000)
    return f'{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}'

def write_srt(items, path, with_color=True):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for i, it in enumerate(items, 1):
            s, e, txt = it[:3]; col = it[3] if len(it) > 3 else ''
            body = f'<font color="{col}">{txt}</font>' if (col and with_color) else txt
            f.write(f'{i}\n{fmt(s)} --> {fmt(e)}\n{body}\n\n')
    os.replace(tmp, path)

def run_ocr(a, video, info, wd):
    """Returns (dialogue_srt_path, captions_srt_path). Cached per chunk under wd/ocr/."""
    from multiprocessing import get_context
    odir = os.path.join(wd, 'ocr'); os.makedirs(odir, exist_ok=True)
    y1 = a.ocr_y1 or info['height']
    params = dict(v=OCR_VERSION, fps=a.ocr_fps, y0=a.ocr_y0, y1=y1, thresh=a.ocr_thresh, min_px=150, min_dur=0.3,
                  verify_below=0.95, chunk=a.ocr_chunk)
    dur = info['duration']
    chunk = max(30.0, min(a.ocr_chunk, np.ceil(dur / a.jobs)))     # short clips: spread over all workers
    params['chunk'] = chunk; nch = int(np.ceil(dur / chunk))
    jobs = []
    for k in range(nch):
        out = os.path.join(odir, f'chunk_{k:05d}.json')
        old = load_json(out)
        if old and old.get('params') == params and 'force_ocr' not in a.force: continue
        jobs.append(dict(video=video, t0=k * chunk, t1=min(dur, (k + 1) * chunk), fps=a.ocr_fps,
                         y0=a.ocr_y0, y1=y1, width=info['width'], out=out, thresh=a.ocr_thresh, min_px=150,
                         min_dur=0.3, verify_below=0.95, params=params))
    log('ocr', f'{nch} chunks of {chunk:.0f}s, {nch - len(jobs)} cached, {len(jobs)} to do, {a.jobs} workers')
    if jobs:
        hw = hwdec_args(a.hwdec, video)
        for j in jobs: j['hw'] = hw
        provider = 'cpu'
        if a.onnx_provider != 'cpu':
            sample = _bench_frame(video, info, a, y1)
            def bench(prov):
                o = OCR(prov, 0); m = text_mask(sample); o(sample, m)
                t = time.time(); [o(sample, m) for _ in range(3)]; return (time.time() - t) / 3
            provider = pick_provider('ocr', a.onnx_provider, bench)
        threads = 1 if provider == 'cpu' else 0
        pr = Progress('ocr', len(jobs))
        ctx = get_context('spawn')
        with ctx.Pool(a.jobs, initializer=_init_worker, initargs=(provider, threads)) as pool:
            for out, n in pool.imap_unordered(ocr_chunk, jobs):
                pr.step(extra=f'(last chunk {n} OCR reads)')
    # assemble
    dia, cap = [], []
    for k in range(nch):
        for it in load_json(os.path.join(odir, f'chunk_{k:05d}.json'))['items']:
            if it['dia']: dia.append([it['start'], it['end'], it['dia'], it['score'], it['color']])
            if it['cap']: cap.append([it['start'], it['end'], it['cap'], it['score'], ''])
    dia, cap = merge(dia), merge(cap, gap=1.0)
    rules = load_corrections(a.corrections)
    dia = [[s, e, apply_corrections(t, rules, 'dialogue'), c] for s, e, t, c in dia]
    dia = dialogue_nits([d for d in dia if d[2]])
    cap = [[s, e, caption_nits(apply_corrections(t, rules, 'captions')), c] for s, e, t, c in cap]
    dsrt, csrt = os.path.join(wd, 'subs.ocr.srt'), os.path.join(wd, 'subs.captions.srt')
    write_srt(dia, dsrt); write_srt([c for c in cap if c[2]], csrt)
    counts = {}
    for d in dia: counts[color_class(d[3])] = counts.get(color_class(d[3]), 0) + 1
    log('ocr', f'{len(dia)} dialogue lines {counts}, {len(cap)} captions -> {dsrt}')
    return dsrt, csrt

def _bench_frame(video, info, a, y1):
    t = min(info['duration'] / 2, 60)
    r = subprocess.run([FFMPEG, '-v', 'error', '-ss', str(t), '-i', video, '-frames:v', '1', '-vf',
                        f"crop={info['width']}:{y1 - a.ocr_y0}:0:{a.ocr_y0}", '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(y1 - a.ocr_y0, info['width'], 3).copy()
