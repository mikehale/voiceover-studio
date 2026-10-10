"""Voice plugins: .zip packages that add a cloned voice (a reference recording + manifest).

Zip layout (flat, no folders):
  manifest.json   format_version, id, name, version, author, accent, sample_rate, reference, files {name: sha256}
  <reference>.wav and any other files listed in manifest["files"]
A plugin is accepted only if every listed file matches its SHA-256 (integrity check against damaged or partial
copies), nothing else is in the zip and the manifest fields are valid. Plugins are not signed. See docs/voice-plugins.md.
Standard library only, so the app, the clone venv and tools/ can all use it."""
import hashlib, io, json, os, re, shutil, wave, zipfile

FORMAT_VERSION = 1
ACCENTS = ('original', 'german_v3')
ID_RE = re.compile(r'^[a-z0-9][a-z0-9_]{0,39}$')
FILE_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
MAX_FILE = 20 * 1024 * 1024          # per file
MAX_TOTAL = 40 * 1024 * 1024
MANIFEST = 'manifest.json'

class PluginError(ValueError): pass

def sha256(b): return hashlib.sha256(b).hexdigest()

def _check_manifest(m):
    if not isinstance(m, dict): raise PluginError('manifest.json is not an object')
    if m.get('format_version') != FORMAT_VERSION: raise PluginError(f"unsupported format_version {m.get('format_version')!r} (this app reads {FORMAT_VERSION})")
    for k in ('id', 'name', 'version', 'author', 'accent', 'sample_rate', 'reference', 'files'):
        if k not in m: raise PluginError(f'manifest.json is missing "{k}"')
    if not isinstance(m['id'], str) or not ID_RE.match(m['id']): raise PluginError('id must be 1-40 of a-z, 0-9, _')
    for k in ('name', 'version', 'author'):
        if not isinstance(m[k], str) or not m[k].strip() or len(m[k]) > 80: raise PluginError(f'"{k}" must be a short non-empty string')
    if m['accent'] not in ACCENTS: raise PluginError(f"unknown accent {m['accent']!r} (known: {', '.join(ACCENTS)})")
    if not isinstance(m['sample_rate'], int) or not 8000 <= m['sample_rate'] <= 96000: raise PluginError('bad sample_rate')
    files = m['files']
    if not isinstance(files, dict) or not files: raise PluginError('"files" must list the plugin files with their SHA-256')
    for n, h in files.items():
        if not FILE_RE.match(n) or n == MANIFEST: raise PluginError(f'bad file name {n!r}')
        if not isinstance(h, str) or not re.match(r'^[0-9a-f]{64}$', h): raise PluginError(f'bad SHA-256 for {n}')
    if m['reference'] not in files or not m['reference'].lower().endswith('.wav'): raise PluginError('"reference" must be a .wav listed in "files"')

def _check_wav(b, m):
    try:
        with wave.open(io.BytesIO(b)) as w: sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    except Exception as e: raise PluginError(f'reference is not a readable PCM WAV ({e})')
    if sr != m['sample_rate']: raise PluginError(f'reference sample rate {sr} does not match the manifest ({m["sample_rate"]})')
    dur = n / sr
    if not 3.0 <= dur <= 60.0: raise PluginError(f'reference is {dur:.1f} s; it must be 3-60 s (10-20 s of one speaker works best)')
    return dict(duration=round(dur, 2), channels=ch)

def verify(read, names):
    """read(name) -> bytes; names = everything in the package. Returns (manifest, info). Raises PluginError."""
    if MANIFEST not in names: raise PluginError('not a voice plugin: manifest.json is missing')
    try: m = json.loads(read(MANIFEST))
    except Exception: raise PluginError('manifest.json is not valid JSON')
    _check_manifest(m)
    extra = set(names) - {MANIFEST} - set(m['files'])
    if extra: raise PluginError(f"files not listed in the manifest: {', '.join(sorted(extra))}")
    total = 0
    for n, h in m['files'].items():
        if n not in names: raise PluginError(f'missing file {n}')
        b = read(n); total += len(b)
        if len(b) > MAX_FILE or total > MAX_TOTAL: raise PluginError('plugin files are too large')
        if sha256(b) != h: raise PluginError(f'hash mismatch for {n}: the file is damaged or was changed after packaging')
    return m, _check_wav(read(m['reference']), m)

def verify_zip(path):
    try: z = zipfile.ZipFile(path)
    except Exception: raise PluginError('not a .zip file')
    with z:
        infos = z.infolist()
        for i in infos:
            if i.is_dir() or '/' in i.filename or '\\' in i.filename or not (FILE_RE.match(i.filename) or i.filename == MANIFEST):
                raise PluginError(f'unexpected entry in the zip: {i.filename!r} (plugins are flat zips)')
            if i.file_size > MAX_FILE: raise PluginError('plugin files are too large')
        names = [i.filename for i in infos]
        if len(set(names)) != len(names): raise PluginError('duplicate entries in the zip')
        m, info = verify(z.read, names)
        return m, info, {n: z.read(n) for n in names}

def verify_dir(path):
    names = [f for f in os.listdir(path) if not f.startswith('.')]
    def read(n):
        with open(os.path.join(path, n), 'rb') as f: return f.read()
    return verify(read, names)

def install(zip_path, plugins_dir):
    """Verify, then extract to plugins_dir/<id>/ (replacing an older copy atomically). Returns the manifest."""
    m, info, files = verify_zip(zip_path)
    os.makedirs(plugins_dir, exist_ok=True)
    dest = os.path.join(plugins_dir, m['id']); tmp = dest + '.installing'; old = dest + '.old'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for n, b in files.items():
        with open(os.path.join(tmp, n), 'wb') as f: f.write(b)
    verify_dir(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(dest): os.replace(dest, old)
    os.replace(tmp, dest); shutil.rmtree(old, ignore_errors=True)
    return m

def scan(plugins_dir):
    """([(manifest, info, folder)] for every valid installed plugin, [(folder, error)] for invalid ones)."""
    ok, bad = [], []
    if not os.path.isdir(plugins_dir): return ok, bad
    for d in sorted(os.listdir(plugins_dir)):
        p = os.path.join(plugins_dir, d)
        if not os.path.isdir(p) or d.endswith(('.installing', '.old')): continue
        try:
            m, info = verify_dir(p)
            if m['id'] != d: raise PluginError(f'folder name {d} does not match plugin id {m["id"]}')
            ok.append((m, info, p))
        except Exception as e: bad.append((d, str(e)))
    return ok, bad

def build(out_zip, reference_wav, id, name, version, author, accent, extra_files=()):
    """Create a plugin zip (used by tools/make_voice_plugin.py)."""
    files = {}
    for p in [reference_wav] + list(extra_files):
        with open(p, 'rb') as f: files[os.path.basename(p)] = f.read()
    ref = os.path.basename(reference_wav)
    with wave.open(io.BytesIO(files[ref])) as w: sr = w.getframerate()
    m = dict(format_version=FORMAT_VERSION, id=id, name=name, version=version, author=author, accent=accent,
             sample_rate=sr, reference=ref, files={n: sha256(b) for n, b in sorted(files.items())})
    _check_manifest(m)
    tmp = out_zip + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(m, indent=1, sort_keys=True) + '\n')
        for n, b in sorted(files.items()): z.writestr(n, b)
    verify_zip(tmp)                                       # self-check before handing it out
    os.replace(tmp, out_zip)
    return m
