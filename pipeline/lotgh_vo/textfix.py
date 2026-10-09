"""OCR text clean-up, user corrections, subtitle nits and colour classes."""
import colorsys, gzip, json, os, re

_ZIPF = None
def _load_zipf():
    """Bundled frequency-ordered English word list (data/wordlist_en.txt.gz, extracted from wordfreq 3.1.1):
    word -> Zipf frequency, identical to wordfreq.zipf_frequency(word, 'en') for lowercase a-z(+'suffix) words."""
    global _ZIPF
    if _ZIPF is None:
        d = {}
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'wordlist_en.txt.gz')
        if os.path.exists(p):
            with gzip.open(p, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'): continue
                    z, words = line.rstrip('\n').split('\t', 1)
                    z = float(z)
                    for w in words.split(' '): d[w] = z
        _ZIPF = d
    return _ZIPF

def word_zipf(word, lang='en'):
    """Zipf frequency of a lowercase word; 0.0 if unknown (drop-in for wordfreq.zipf_frequency)."""
    return _load_zipf().get(word, 0.0)

LATIN = re.compile(r"[A-Za-z0-9 ,.'!?\-:;\"()&%$/…’‘“”]")

def clean(t):
    t = t.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"').replace('…', '...')
    t = re.sub(r'^[^A-Za-z0-9"\'(.]+', '', t)            # leading junk like "+.+" or "-"
    t = re.sub(r'[·•|]+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\s+([,.!?;:])', r'\1', t)              # "right ?" -> "right?"
    t = re.sub(r'^\.\s+(?=[A-Za-z])', '...', t)       # "... the" read as ". the"
    t = re.sub(r'^\.+\s*[Il|](?=[a-z]{2})', '...', t)   # "...Ithe" / ".Ithe" (ellipsis dot read as I)
    t = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", ' ', t)   # "Our/battles"
    t = re.sub(r'\.{4,}', '...', t)
    t = re.sub(r'(?<!\.)\.\.(?!\.)', '.', t)          # "damage.." -> "damage."
    t = re.sub(r"(?<=[a-z])'$", '', t)                 # trailing stray quote
    return t

_ALPHA = 'abcdefghijklmnopqrstuvwxyz'
OKWORD = re.compile(r"[a-z]+('(s|t|re|ll|ve|d|m))?")
def spellfix(text):
    """Conservative repair of OCR noise caused by scenery lines touching letters. A token is only touched if
    the bundled word list has never seen it (zipf 0), so names like Phezzan/Kircheis are left alone.
      1) single edit (insert/delete/substitute, incl. stray ' / | l) -> common word (zipf>=3): vast'ess->vastness
      2) else split into two common words (space inserted or replacing a junk char): in'my->in my, tolone->to one
    Also lower-cases common words wrongly capitalised mid-sentence ("what We must" -> "what we must")."""
    if not _load_zipf(): return text
    zf = word_zipf
    def fix(m):
        w = m.group(0); lw = w.lower()
        if len(lw.replace("'", '')) < 4: return w
        if "'" in lw:
            if not OKWORD.fullmatch(lw) or zf(lw.split("'")[0], 'en') == 0: z0 = 0.0   # stray apostrophe = junk
            else: return w
        else: z0 = zf(lw, 'en')
        if z0 >= 1.5: return w
        best, bz = None, 0.0
        for i in range(len(lw) + 1):
            cands = [lw[:i] + c + lw[i:] for c in _ALPHA]
            if i < len(lw): cands += [lw[:i] + c + lw[i + 1:] for c in _ALPHA] + [lw[:i] + lw[i + 1:]]
            for c in cands:
                if not OKWORD.fullmatch(c): continue
                z = zf(c, 'en')
                if z > bz: best, bz = c, z
        junk = "'" in lw
        if best and bz >= (2.5 if junk else 3) and bz - z0 >= 2.5:
            return best.capitalize() if (w[0].isupper() and best[0] == lw[0]) else best
        sbest, sz = None, 0.0
        for i in range(1, len(lw)):
            for l, r in ((lw[:i], lw[i:]), (lw[:i], lw[i + 1:])):
                if not l or not r or "'" in l + r: continue
                z = min(zf(l, 'en'), zf(r, 'en'))
                if (len(l) > 1 or l in ('a', 'i')) and (len(r) > 1 or r in ('a', 'i')) and z > sz: sbest, sz = (l, r), z
        if sbest and sz >= (5.0 if w[0].isupper() else 4.5):
            l, r = sbest
            if l == 'i': l = 'I'
            return (l.capitalize() if w[0].isupper() else l) + ' ' + r
        return w
    return re.sub(r"[A-Za-z']+", fix, text)

def fix_caps(text):
    zf = word_zipf
    # mid-sentence capitalised common word (not followed by another capitalised word, e.g. "Your Excellency")
    toks = text.split(' ')
    for i in range(1, len(toks)):
        t = toks[i]; core = re.sub(r'[^A-Za-z]', '', t)
        if not core or not core[0].isupper() or core == 'I' or core.isupper(): continue
        if toks[i - 1].endswith(('.', '?', '!', '"', ':')): continue
        nxt = re.sub(r'[^A-Za-z]', '', toks[i + 1]) if i + 1 < len(toks) else ''
        if nxt[:1].isupper(): continue
        if zf(core.lower(), 'en') >= 5.0:
            toks[i] = t.replace(core, core.lower(), 1)
    return ' '.join(toks)


def quality(txt, score):
    """OCR confidence + share of dictionary words - junk; used to choose between alternative readings."""
    if not _load_zipf(): return score
    zf = word_zipf
    toks = re.findall(r"[A-Za-z']+", txt)
    if not toks: return score
    known = sum(1 for t in toks if OKWORD.fullmatch(t.lower()) and zf(t.lower(), 'en') >= 2.5) / len(toks)
    junk = len(re.findall(r"[a-z]'[a-z](?<!'s)|[|/]|\b[A-Za-z]{18,}\b", txt))
    return score + 0.6 * known - 0.15 * junk


# ---------------- user corrections ----------------
def load_corrections(path):
    """JSON: {"replacements": [{"find": "...", "replace": "...", "regex": false, "applies_to": "all|dialogue|captions"}]}"""
    if not path: return []
    with open(path) as f: data = json.load(f)
    out = []
    for r in data.get('replacements', []):
        if not r.get('find'): continue
        pat = re.compile(r['find'] if r.get('regex') else re.escape(r['find']), 0 if r.get('case_sensitive', True) else re.I)
        out.append((pat, r.get('replace', ''), r.get('applies_to', 'all')))
    return out

def apply_corrections(text, rules, kind):
    for pat, rep, where in rules:
        if where in ('all', kind): text = pat.sub(rep, text)
    return re.sub(r'\s+', ' ', text).strip()

def dialogue_nits(lines):
    """lines: [[start,end,text,color]]. A line ending in ',' followed by a line starting with a capital letter
    (a new sentence) gets a '.' instead (OCR often reads the final '.' as ',')."""
    for i in range(len(lines) - 1):
        t, n = lines[i][2], lines[i + 1][2]
        if t.endswith(',') and n[:1].isupper() and not n.startswith('I '):
            lines[i][2] = t[:-1] + '.'
    return lines

def caption_nits(text):
    """Name/place cards never end in a full stop: drop a single trailing '.' (keep '...')."""
    return text[:-1] if text.endswith('.') and not text.endswith('..') else text

# ---------------- colour classes ----------------
def color_class(hexcol):
    """'#rrggbb' -> white | yellow | cyan | other (hue-based, robust to the slight tints seen in the encode)."""
    if not hexcol: return 'other'
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hh, s, v = colorsys.rgb_to_hsv(r, g, b); hue = hh * 360
    if s < 0.25: return 'white'
    if 40 <= hue <= 80: return 'yellow'
    if 150 <= hue <= 210: return 'cyan'
    if hue <= 40 or hue >= 330: return 'red'
    if 80 < hue < 150: return 'green'
    return 'other'
