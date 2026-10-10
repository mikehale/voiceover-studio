#!/usr/bin/env python3
"""Voiceover Studio backend: first-run setup, persistent download/voice-over queue, small JSON API + static UI.

Standard library only (it runs before the heavy dependencies are installed). The voice-over itself runs as
`python -m lotgh_vo` subprocesses in the private venv.
"""
import argparse, hashlib, json, os, re, secrets, shutil, signal, subprocess, sys, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

APP_NAME = 'Voiceover Studio'
VERSION = '0.4.1-dev'
KOKORO_FILES = [
    ('kokoro-v1.0.fp16.onnx', 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx', 177464787),
    ('voices-v1.0.bin', 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin', 28214398),
]
OLD_MODEL_FILES = ['kokoro-v1.0.onnx']            # fp32 model used by v1.0.0; replaced by the fp16 one
DENO_URL = 'https://github.com/denoland/deno/releases/latest/download/deno-aarch64-apple-darwin.zip'
STAGE_WEIGHTS = [('ocr', 0.40), ('separate', 0.18), ('tts', 0.30), ('mix', 0.07), ('mux', 0.05)]
CLONE_STAGE_WEIGHTS = [('ocr', 0.14), ('separate', 0.06), ('clone', 0.74), ('tts', 0.02), ('mix', 0.02), ('mux', 0.02)]
# optional cloned voices (Chatterbox): separate venv + model, only when the user opts in
CLONE_REPO = 'ResembleAI/chatterbox'
CLONE_FILES_BYTES = 3191366992            # ve + t3_cfg + s3gen + tokenizer + conds at the pinned revision
# German accent (Chatterbox Multilingual V3): extra T3 model + tokenizer in the same model folder, same pinned
# revision, downloaded on first use. Used by the accent option and by German-accent voice plugins.
DE_FILES = {'t3_mtl23ls_v3.safetensors': 2143989928, 'grapheme_mtl_merged_expanded_v1.json': 69989}
DE_FILES_BYTES = sum(DE_FILES.values())
DE_MARKER = 'clone_de_v3_done.json'
ACCENTS = ('original', 'german_v3')
# v0.3.x: accent id 'german' (an older Multilingual model) -> migrated at start; its files are removed
LEGACY_ACCENTS = {'german': 'german_v3'}
OLD_DE_FILES = ('t3_mtl23ls_v2.safetensors', 'clone_de_done.json')
JP_OFF = -41                              # Japanese-voices slider at its minimum = off (stem left out, no ducking)
VIDEO_EXT = ('.mkv', '.webm', '.mp4', '.mov', '.m4v', '.avi')

# ----------------------------------------------------------------------------------------------------------------
class Paths:
    def __init__(self, a):
        self.res = os.path.abspath(a.resources)                   # Contents/Resources
        self.app = os.path.join(self.res, 'app')
        self.pipeline = os.path.join(self.app, 'pipeline')
        self.ui = os.path.join(self.app, 'ui')
        self.data = os.path.abspath(os.path.expanduser(a.data))   # ~/Library/Application Support/Voiceover Studio
        self.venv = os.path.join(self.data, 'venv')
        self.py = os.path.join(self.venv, 'bin', 'python')
        self.cvenv = os.path.join(self.data, 'clone-venv')            # optional Chatterbox venv (opt-in)
        self.cpy = os.path.join(self.cvenv, 'bin', 'python')
        self.cmodel = os.path.join(self.data, 'models', 'chatterbox')       # plain files, removable as a unit
        self.plugins = os.path.join(self.data, 'plugins')           # installed voice plugins, one folder each
        self.bin = os.path.join(self.res, 'bin')                  # bundled uv, ffmpeg, ffprobe
        self.dbin = os.path.join(self.data, 'bin')                # downloaded deno
        self.models = os.path.join(self.data, 'models')
        self.torch = os.path.join(self.data, 'torch')
        self.downloads = os.path.join(self.data, 'downloads')
        self.work = os.path.join(self.data, 'work')
        self.logs = os.path.join(self.data, 'logs')
        for d in (self.data, self.dbin, self.models, self.torch, self.downloads, self.work, self.logs, self.plugins):
            os.makedirs(d, exist_ok=True)
        self.dev = a.dev

    def tool(self, name):
        for d in (self.bin, self.dbin):
            p = os.path.join(d, name)
            if os.path.exists(p): return p
        return shutil.which(name) if self.dev else os.path.join(self.bin, name)

P = None
LOCK = threading.RLock()
STATE = {'setup': {}, 'items': [], 'settings': {}, 'paused': False, 'clone': {'state': 'absent'}}
CHANGED = threading.Event()

def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    try:
        with open(os.path.join(P.logs, 'server.log'), 'a') as f: f.write(line + '\n')
    except Exception: pass

def atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def load(path, default):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def default_settings():
    return dict(output_dir=os.path.expanduser('~/Movies/Voiceover'), voice_dialogue='bm_daniel', voice_narrator='bm_lewis',
                voice_lyrics='skip', jp_db=JP_OFF, duck_db=-9, tts_db=0, keep_song_vocals=True, keep_source=False, video_codec='h264',
                cookies='none', max_height=720, clone_default=False, clone_optin=False, clone_accent='original',
                clone_voice_narrator='', clone_voice_dialogue='')

# ------------------------------------------- voice plugins ---------------------------------------------------------
# A voice plugin is a zip (manifest.json with SHA-256 file hashes + a 10-20 s reference recording of one speaker; see
# docs/voice-plugins.md). Installed plugins live in <data>/plugins/<id>/ and are re-checked on every scan; narrator
# and/or dialogue lines can use one instead of the voices cloned from the video (needs cloned voices installed; a
# German-accent plugin also needs the German model).
VOICE_ROLES = ('narrator', 'dialogue')
PLUGINS = {'voices': [], 'errors': []}

def _vp():
    if P.pipeline not in sys.path: sys.path.insert(0, P.pipeline)
    from lotgh_vo import voiceplugin
    return voiceplugin

def scan_plugins():
    ok, bad = _vp().scan(P.plugins)
    with LOCK:
        PLUGINS['voices'] = [dict(id=m['id'], name=m['name'], accent=m['accent'], version=m['version'], author=m['author'],
                                  duration=info['duration'], path=os.path.join(d, m['reference']))
                             for m, info, d in ok]
        PLUGINS['errors'] = [dict(id=d, error=e) for d, e in bad]
    for d, e in bad: log(f'voice plugin {d} ignored: {e}')
    CHANGED.set()

def custom_voices():
    with LOCK: return list(PLUGINS['voices'])

def custom_voice(vid):
    return next((v for v in custom_voices() if v['id'] == vid), None) if vid else None

def install_plugin(path):
    vp = _vp(); path = os.path.abspath(os.path.expanduser(path or ''))
    if not os.path.isfile(path): raise ValueError('file not found: ' + path)
    try: m = vp.install(path, P.plugins)
    except vp.PluginError as e:
        log(f'voice plugin rejected ({path}): {e}'); raise ValueError(f'Voice plugin rejected: {e}')
    log(f"voice plugin installed: {m['name']} ({m['id']} {m['version']}, accent {m['accent']}) from {path}")
    scan_plugins()
    if m['accent'] == 'german_v3': start_de_install()
    return m['id']

def remove_plugin(vid):
    if not re.match(r'^[a-z0-9_]+$', vid or '') or not os.path.isdir(os.path.join(P.plugins, vid)): raise ValueError('no such voice plugin')
    shutil.rmtree(os.path.join(P.plugins, vid)); log(f'voice plugin removed: {vid}')
    scan_plugins()
    with LOCK:
        clean_voice_ids(STATE['settings'], 'clone_voice_')
        for it in STATE['items']:
            if it['status'] not in ('done', 'processing', 'downloading'): clean_voice_ids(it, 'cvoice_')
    save_settings(); save_queue()

def clean_voice_ids(d, prefix):
    """Unknown custom-voice ids (or any without cloned voices installed) become '' (= voices cloned from the video)."""
    ok = {v['id'] for v in custom_voices()} if clone_ready() else set()
    for r in VOICE_ROLES:
        if d.get(prefix + r) not in ok: d[prefix + r] = ''

def start_voice_models(d, prefix):
    """Starts the one-time accent download a chosen custom voice needs."""
    if any((custom_voice(d.get(prefix + r)) or {}).get('accent') == 'german_v3' for r in VOICE_ROLES): start_de_install()

def migrate_accents():
    """Old accent ids in settings and queue -> current ones; old German model files removed."""
    s = STATE['settings']
    s['clone_accent'] = LEGACY_ACCENTS.get(s.get('clone_accent'), s.get('clone_accent') or 'original')
    for it in STATE['items']: it['accent'] = LEGACY_ACCENTS.get(it.get('accent'), it.get('accent') or 'original')
    for f in OLD_DE_FILES:
        for d in (P.cmodel, P.data):
            fp = os.path.join(d, f)
            if os.path.exists(fp): os.remove(fp); log(f'removed old German-accent file {f}')

def save_queue():
    with LOCK:
        atomic_write(os.path.join(P.data, 'queue.json'), {'items': STATE['items'], 'paused': STATE['paused']})
    CHANGED.set()

def save_settings():
    with LOCK:
        atomic_write(os.path.join(P.data, 'settings.json'), STATE['settings'])
        s = STATE['settings']
        voices = {'_help': 'Written by Voiceover Studio from Settings.',
                  'classes': {'yellow': s['voice_dialogue'], 'cyan': s['voice_narrator'],
                              'white': s['voice_lyrics'], 'red': s['voice_dialogue'],
                              'green': s['voice_dialogue'], 'other': s['voice_dialogue']},
                  'speeds': s.get('speeds', {})}
        atomic_write(os.path.join(P.data, 'voices.json'), voices)
    CHANGED.set()

def env_for_tools():
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join([P.bin, P.dbin, os.path.join(P.venv, 'bin'), '/usr/bin', '/bin', '/usr/sbin', '/sbin']
                                  + ([os.environ.get('PATH', '')] if P.dev else []))
    env['LOTGH_FFMPEG'] = P.tool('ffmpeg') or 'ffmpeg'
    env['LOTGH_FFPROBE'] = P.tool('ffprobe') or 'ffprobe'
    env['TORCH_HOME'] = P.torch
    env['HF_HOME'] = os.path.join(P.data, 'hf')
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['PYTHONPATH'] = P.pipeline
    env['PYTHONUNBUFFERED'] = '1'
    env['UV_NO_CACHE'] = '1'                 # uv uses a throw-away temp cache; nothing persists
    env['UV_LINK_MODE'] = 'copy'             # venv files are real copies, never links into a cache
    env.pop('UV_CACHE_DIR', None)
    env['UV_PYTHON_INSTALL_DIR'] = os.path.join(P.data, 'python')
    env['HF_HUB_DISABLE_TELEMETRY'] = '1'
    env['TQDM_DISABLE'] = '1'
    env['LOTGH_CLONE_MODEL'] = P.cmodel
    cert = certifi_path()
    if cert: env['SSL_CERT_FILE'] = cert
    return env

_CERT = None
def certifi_path():
    global _CERT
    if _CERT and os.path.exists(_CERT): return _CERT
    for root, _, files in os.walk(os.path.join(P.venv, 'lib')):
        if root.endswith(os.path.join('certifi')) and 'cacert.pem' in files:
            _CERT = os.path.join(root, 'cacert.pem'); return _CERT
    return None

# ------------------------------------------------- first-run setup ------------------------------------------------
STEPS = [('packages', 'Python packages (PyTorch, Demucs, Kokoro, OCR)'),
         ('ytdlp', 'yt-dlp downloader'),
         ('deno', 'JavaScript runtime for YouTube (deno)'),
         ('kokoro', 'Kokoro voice model (205 MB)'),
         ('demucs', 'Demucs vocal-separation model (~170 MB)'),
         ('check', 'Self-test')]

def setup_state(): return STATE['setup']

def set_step(key, **kw):
    with LOCK:
        st = STATE['setup']['steps'][key]; st.update(kw)
    CHANGED.set()

def run_logged(cmd, key, env=None, cwd=None):
    """Run a setup command, streaming its last output line into the step detail."""
    logf = open(os.path.join(P.logs, 'setup.log'), 'a')
    logf.write(f"\n$ {' '.join(cmd)}\n"); logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env or env_for_tools(),
                         cwd=cwd, bufsize=1)
    last = ''
    for line in p.stdout:
        logf.write(line); line = line.strip()
        if line: last = line; set_step(key, detail=line[-160:])
    p.wait(); logf.close()
    if p.returncode != 0: raise RuntimeError(f'{os.path.basename(cmd[0])} failed: {last[-300:]}')

def curl_download(url, dest, key, expected=0):
    tmp = dest + '.part'
    p = subprocess.Popen(['/usr/bin/curl', '-fL', '--retry', '3', '-C', '-', '-s', '-S', '-o', tmp, url],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while p.poll() is None:
        if expected and os.path.exists(tmp):
            set_step(key, pct=min(99.0, 100 * os.path.getsize(tmp) / expected),
                     detail=f'{os.path.getsize(tmp) / 1e6:.0f} / {expected / 1e6:.0f} MB')
        time.sleep(0.5)
    if p.returncode != 0:
        raise RuntimeError(f'download failed ({url}): {p.stderr.read().strip()[-300:]}')
    if expected and os.path.getsize(tmp) != expected:
        raise RuntimeError(f'download incomplete: {os.path.getsize(tmp)} of {expected} bytes')
    os.replace(tmp, dest)

def _site_packages():
    import glob
    return glob.glob(os.path.join(P.venv, 'lib', 'python3*', 'site-packages'))

def lock_hash():
    with open(os.path.join(P.app, 'requirements.lock'), 'rb') as f: return hashlib.sha1(f.read()).hexdigest()[:12]

def do_setup():
    marks_path = os.path.join(P.data, 'setup_done.json')
    marks = load(marks_path, {})
    with LOCK:
        STATE['setup'] = {'state': 'running', 'error': '', 'steps': {k: {'label': l, 'state': 'pending', 'pct': None,
                                                                          'detail': ''} for k, l in STEPS},
                          'order': [k for k, _ in STEPS]}
    CHANGED.set()
    uv = P.tool('uv')
    def mark(k, v=True):
        marks[k] = v; atomic_write(marks_path, marks); set_step(k, state='done', pct=100.0, detail='')
    try:
        # 1. pinned python packages into the private venv
        set_step('packages', state='running')
        if marks.get('packages') != lock_hash():
            # sync = install the pinned set and remove anything else (e.g. wordfreq and its deps from v1.0.0);
            # yt-dlp is re-added by the next step.
            run_logged([uv, 'pip', 'sync', '--python', P.py, os.path.join(P.app, 'requirements.lock')], 'packages')
            mark('packages', lock_hash())
        else: mark('packages', lock_hash())
        # venvs created by v1.0.0 were seeded with pip/setuptools/wheel (uv sync keeps those); nothing uses them
        seeded = [n for n in ('pip', 'setuptools', 'wheel')
                  if any(os.path.isdir(os.path.join(sp, n)) for sp in _site_packages())]
        if seeded: run_logged([uv, 'pip', 'uninstall', '--python', P.py] + seeded, 'packages')
        # 2. yt-dlp (unpinned on purpose: YouTube changes often; 'Update yt-dlp' in Settings re-runs this)
        set_step('ytdlp', state='running')
        if not marks.get('ytdlp') or not os.path.exists(os.path.join(P.venv, 'bin', 'yt-dlp')):
            run_logged([uv, 'pip', 'install', '--python', P.py, '-U', 'yt-dlp[default]'], 'ytdlp')
        mark('ytdlp')
        # 3. deno (yt-dlp needs a JS runtime for YouTube)
        set_step('deno', state='running')
        deno = os.path.join(P.dbin, 'deno')
        if not os.path.exists(deno) and not P.dev:
            z = os.path.join(P.dbin, 'deno.zip')
            curl_download(DENO_URL, z, 'deno')
            subprocess.run(['/usr/bin/ditto', '-x', '-k', z, P.dbin], check=True); os.remove(z)
            os.chmod(deno, 0o755)
        mark('deno')
        # 4. Kokoro model files
        set_step('kokoro', state='running')
        tot = sum(s for _, _, s in KOKORO_FILES); done = 0
        for name, url, size in KOKORO_FILES:
            dest = os.path.join(P.models, name)
            if not (os.path.exists(dest) and not os.path.islink(dest) and os.path.getsize(dest) == size):
                if os.path.islink(dest): os.remove(dest)
                curl_download(url, dest, 'kokoro', size)
            done += size; set_step('kokoro', pct=100 * done / tot)
        for old in OLD_MODEL_FILES:
            op = os.path.join(P.models, old)
            if os.path.lexists(op): os.remove(op); log(f'removed old model {old}')
        mark('kokoro')
        # 5. Demucs weights (torch hub cache inside our data dir)
        set_step('demucs', state='running', detail='downloading htdemucs weights')
        if not marks.get('demucs'):
            run_logged([P.py, '-c', "from demucs.pretrained import get_model; get_model('htdemucs'); print('ok')"], 'demucs')
        mark('demucs')
        # 6. self-test
        set_step('check', state='running')
        run_logged([P.py, '-c', 'import torch, demucs, kokoro_onnx, rapidocr_onnxruntime, cv2, soundfile; from lotgh_vo.textfix import word_zipf; assert word_zipf("the") > 7;'
                    'print("torch", torch.__version__, "MPS", torch.backends.mps.is_available())'], 'check')
        ff = subprocess.run([P.tool('ffmpeg'), '-hide_banner', '-decoders'], capture_output=True, text=True).stdout
        if 'libdav1d' not in ff and 'av1' not in ff: raise RuntimeError('bundled ffmpeg cannot decode AV1')
        mark('check')
        if STATE['settings'].get('clone_optin') and not clone_ready():
            with LOCK:
                STATE['setup']['steps']['clone'] = {'label': 'Cloned voices (Chatterbox, optional)', 'state': 'running',
                                                    'pct': None, 'detail': ''}
                STATE['setup']['order'].append('clone')
            do_clone_install()        # never fails base setup; errors show in Settings > Voices
        with LOCK: STATE['setup']['state'] = 'ready'
        log('setup complete')
    except Exception as e:
        log('setup failed: ' + traceback.format_exc())
        with LOCK:
            STATE['setup']['state'] = 'error'; STATE['setup']['error'] = str(e)
            for st in STATE['setup']['steps'].values():
                if st['state'] == 'running': st['state'] = 'error'
    CHANGED.set()

def setup_ready(): return STATE['setup'].get('state') == 'ready'

# ------------------------------------------- optional cloned voices -----------------------------------------------
def clone_lock_hash():
    with open(os.path.join(P.app, 'clone-requirements.lock'), 'rb') as f: return hashlib.sha1(f.read()).hexdigest()[:12]

def clone_ready():
    m = load(os.path.join(P.data, 'clone_done.json'), {})
    return bool(m) and m.get('lock') == clone_lock_hash() and os.path.exists(P.cpy) and \
        os.path.exists(os.path.join(P.cmodel, 't3_cfg.safetensors'))

def dir_size(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                try: tot += os.path.getsize(fp)
                except OSError: pass
    return tot

def de_ready():
    return clone_ready() and os.path.exists(os.path.join(P.data, DE_MARKER)) and all(
        os.path.exists(os.path.join(P.cmodel, f)) and os.path.getsize(os.path.join(P.cmodel, f)) == n for f, n in DE_FILES.items())

def de_refresh():
    with LOCK:
        cur = STATE['clone'].get('de') or {}
        if cur.get('state') == 'installing': return
        if de_ready():
            de = {'state': 'ready', 'size': sum(DE_FILES.values()),
                  'info': load(os.path.join(P.data, DE_MARKER), {}).get('selftest', '')}
        else:
            de = {'state': 'error' if cur.get('state') == 'error' else 'absent', 'error': cur.get('error', ''),
                  'size': DE_FILES_BYTES}
        STATE['clone']['de'] = de
    CHANGED.set()

def clone_refresh():
    with LOCK:
        if STATE['clone'].get('state') in ('installing', 'removing'): return
        if clone_ready():
            STATE['clone'] = {'state': 'ready', 'detail': '', 'size': dir_size(P.cvenv) + dir_size(P.cmodel),
                              'info': load(os.path.join(P.data, 'clone_done.json'), {}).get('selftest', ''),
                              'de': STATE['clone'].get('de') or {}}
        else:
            partial = os.path.exists(P.cvenv) or os.path.exists(P.cmodel)
            STATE['clone'] = {'state': 'error' if STATE['clone'].get('state') == 'error' else 'absent',
                              'error': STATE['clone'].get('error', ''), 'partial': partial,
                              'size': (dir_size(P.cvenv) + dir_size(P.cmodel)) if partial else 0, 'de': {'state': 'absent'}}
    if STATE['clone'].get('state') == 'ready': de_refresh()
    CHANGED.set()

def cset(**kw):
    with LOCK:
        STATE['clone'].update(kw)
        st = STATE['setup'].get('steps', {}).get('clone')
        if st and STATE['setup'].get('state') == 'running':
            st.update({k: v for k, v in kw.items() if k in ('pct', 'detail')})
            if kw.get('state') in ('ready', 'error'): st['state'] = 'done' if kw['state'] == 'ready' else 'error'
    CHANGED.set()

def _clone_logged(cmd, env=None, set_status=None):
    logf = open(os.path.join(P.logs, 'clone_setup.log'), 'a')
    logf.write(f"\n$ {' '.join(cmd)}\n"); logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env or env_for_tools(), bufsize=1)
    last = ''
    for line in p.stdout:
        logf.write(line); line = line.strip()
        if line: last = line; (set_status or cset)(detail=line[-160:])
    p.wait(); logf.close()
    if p.returncode != 0: raise RuntimeError(f'{os.path.basename(cmd[0])} failed: {last[-300:]}')
    return last

CLONE_LOCK = threading.Lock()
def do_clone_install():
    """Separate venv + pinned packages + model files + self-test. The base venv is never touched."""
    if not CLONE_LOCK.acquire(blocking=False): return
    try:
        cset(state='installing', error='', pct=None, detail='creating separate Python environment')
        log('cloned voices: install started')
        uv = P.tool('uv')
        if not os.path.exists(P.cpy):
            _clone_logged([uv, 'venv', '--python', '3.11', P.cvenv])
        cset(detail='installing Chatterbox and PyTorch (about 1 GB)')
        _clone_logged([uv, 'pip', 'sync', '--python', P.cpy, os.path.join(P.app, 'clone-requirements.lock')])
        cset(detail='downloading the Chatterbox model (3.2 GB)', pct=0.0)
        done = threading.Event(); xet = os.path.join(P.data, 'hf', 'xet'); x0 = dir_size(xet)
        def watch():
            while not done.wait(1.0):
                sz = dir_size(P.cmodel) + max(0, dir_size(xet) - x0)
                cset(pct=min(99.0, 100 * sz / CLONE_FILES_BYTES), detail=f'downloading the Chatterbox model: '
                     f'{sz / 1e9:.1f} / {CLONE_FILES_BYTES / 1e9:.1f} GB')
        threading.Thread(target=watch, daemon=True).start()
        try: _clone_logged([P.cpy, '-m', 'lotgh_vo.clone_worker', '--download', P.cmodel])
        finally: done.set()
        cset(pct=None, detail='self-test (loading the model and speaking one sentence)')
        env = env_for_tools(); env['HF_HUB_OFFLINE'] = '1'
        out = _clone_logged([P.cpy, '-m', 'lotgh_vo.clone_worker', '--selftest'], env=env)
        out = (re.findall(r'chatterbox ok[^\r\n]*', out) or [out])[-1]
        atomic_write(os.path.join(P.data, 'clone_done.json'), {'lock': clone_lock_hash(), 'selftest': out, 'time': time.time()})
        log('cloned voices installed: ' + out)
        cset(state='ready', pct=100.0, detail='')
        clone_refresh()
    except Exception as e:
        log('cloned voices install failed: ' + traceback.format_exc())
        cset(state='error', error=str(e)[:400], pct=None, detail='')
        clone_refresh()
    finally:
        CLONE_LOCK.release()

DE_LOCK = threading.Lock()
def dset(**kw):
    with LOCK: STATE['clone'].setdefault('de', {}).update(kw)
    CHANGED.set()

def do_de_install():
    """German accent (Multilingual V3): ~2.1 GB of extra model files into the cloned-voices model folder + a self-test.
    Blocks until done if another thread is already installing it. Returns True when ready."""
    if de_ready(): return True
    if not clone_ready(): return False
    if not DE_LOCK.acquire(blocking=False):
        with DE_LOCK: return de_ready()
    try:
        if de_ready(): return True
        dset(state='installing', error='', pct=0.0, detail='downloading the German-accent model')
        log('German accent: install started')
        done = threading.Event(); xet = os.path.join(P.data, 'hf', 'xet'); x0 = dir_size(xet); m0 = dir_size(P.cmodel)
        def watch():
            while not done.wait(1.0):
                sz = max(0, dir_size(P.cmodel) - m0) + max(0, dir_size(xet) - x0)
                dset(pct=min(99.0, 100 * sz / DE_FILES_BYTES),
                     detail=f'downloading the German-accent model: {sz / 1e9:.1f} / {DE_FILES_BYTES / 1e9:.1f} GB')
        threading.Thread(target=watch, daemon=True).start()
        try: _clone_logged([P.cpy, '-m', 'lotgh_vo.clone_worker', '--download-de', P.cmodel], set_status=dset)
        finally: done.set()
        dset(pct=None, detail='self-test (loading the German-accent model and speaking one sentence)')
        env = env_for_tools(); env['HF_HUB_OFFLINE'] = '1'
        out = _clone_logged([P.cpy, '-m', 'lotgh_vo.clone_worker', '--selftest', 'auto', 'german_v3'], env=env, set_status=dset)
        out = (re.findall(r'chatterbox ok[^\r\n]*', out) or [out])[-1]
        atomic_write(os.path.join(P.data, DE_MARKER), {'model': 'multilingual-v3', 'selftest': out, 'time': time.time()})
        log('German accent installed: ' + out)
        dset(state='ready', pct=100.0, detail='')
        clone_refresh(); return True                  # also updates the total size shown for cloned voices
    except Exception as e:
        log('German accent install failed: ' + traceback.format_exc())
        dset(state='error', error=str(e)[:400], pct=None, detail='')
        return False
    finally:
        DE_LOCK.release()

def start_de_install():
    if clone_ready() and not de_ready() and not DE_LOCK.locked():
        threading.Thread(target=do_de_install, daemon=True).start()

def do_clone_remove():
    if not CLONE_LOCK.acquire(blocking=False): return
    try:
        cset(state='removing', detail='removing cloned voices', pct=None)
        for p in (os.path.join(P.data, 'clone_done.json'), os.path.join(P.data, DE_MARKER)):
            if os.path.exists(p): os.remove(p)
        shutil.rmtree(P.cvenv, ignore_errors=True); shutil.rmtree(P.cmodel, ignore_errors=True)
        with LOCK:
            STATE['settings']['clone_default'] = False; STATE['settings']['clone_optin'] = False
            STATE['settings']['clone_accent'] = 'original'
            STATE['settings']['clone_voice_narrator'] = STATE['settings']['clone_voice_dialogue'] = ''
            for it in STATE['items']:
                if it['status'] in ('queued', 'error', 'stopped') and it.get('clone'): it['clone'] = False
                if it['status'] != 'done': it['accent'] = 'original'; it['cvoice_narrator'] = it['cvoice_dialogue'] = ''
        save_settings(); save_queue()
        log('cloned voices removed')
        with LOCK: STATE['clone'] = {'state': 'idle'}
        clone_refresh()
    finally:
        CLONE_LOCK.release()

# --------------------------------------------------- queue ------------------------------------------------------
def new_item(**kw):
    it = dict(id=uuid.uuid4().hex[:10], kind='url', url='', path='', title='', start='', end='', status='queued',
              stage='', stage_pct=0.0, pct=0.0, message='', output='', added=time.time(), updated=time.time(), lines=0,
              clone=False, accent='original', cvoice_narrator='', cvoice_dialogue='')
    it.update(kw); return it

def find(iid):
    for it in STATE['items']:
        if it['id'] == iid: return it
    return None

def upd(it, **kw):
    with LOCK:
        it.update(kw); it['updated'] = time.time()
    save_queue()

URL_RE = re.compile(r'https?://\S+')

def add_text(text, start='', end='', clone=False, accent='original', cv=None):
    cv = cv or {}
    added = []
    with LOCK:
        for raw in text.splitlines():
            s = raw.strip().strip('"\'')
            if not s: continue
            if URL_RE.match(s):
                added.append(new_item(kind='url', url=s, title=s, status='expanding', message='Reading video info...',
                                      start=start, end=end, clone=clone, accent=accent, **cv))
            elif os.path.exists(os.path.expanduser(s)):
                pth = os.path.abspath(os.path.expanduser(s))
                added.append(new_item(kind='file', path=pth, title=os.path.basename(pth), start=start, end=end, clone=clone,
                                      accent=accent, **cv))
        STATE['items'].extend(added)
    save_queue()
    return len(added)

def add_files(paths, start='', end='', clone=False, accent='original', cv=None):
    cv = cv or {}
    with LOCK:
        for pth in paths:
            if os.path.isfile(pth):
                STATE['items'].append(new_item(kind='file', path=pth, title=os.path.basename(pth), start=start, end=end,
                                               clone=clone, accent=accent, **cv))
    save_queue()

def ytdlp_base():
    s = STATE['settings']
    cmd = [os.path.join(P.venv, 'bin', 'yt-dlp'), '--no-colors', '--ffmpeg-location', os.path.dirname(P.tool('ffmpeg'))]
    deno = P.tool('deno')
    if deno and os.path.exists(deno): cmd += ['--js-runtimes', f'deno:{deno}']
    if s.get('cookies', 'none') != 'none': cmd += ['--cookies-from-browser', s['cookies']]
    return cmd

def expander_loop():
    """Turns pasted URLs into queue items; playlists become one item per video."""
    while True:
        time.sleep(0.5)
        if not setup_ready(): continue
        with LOCK: todo = [it for it in STATE['items'] if it['status'] == 'expanding']
        for it in todo:
            try:
                r = subprocess.run(ytdlp_base() + ['--flat-playlist', '-J', it['url']], capture_output=True, text=True,
                                   env=env_for_tools(), timeout=300)
                if r.returncode != 0: raise RuntimeError((r.stderr.strip().splitlines() or ['yt-dlp failed'])[-1])
                info = json.loads(r.stdout)
                entries = [e for e in (info.get('entries') or []) if e] if info.get('_type') == 'playlist' else None
                if entries:
                    new = []
                    for e in entries:
                        url = e.get('url') or e.get('webpage_url') or e.get('id')
                        if url and not url.startswith('http') and e.get('ie_key') == 'Youtube':
                            url = f'https://www.youtube.com/watch?v={url}'
                        new.append(new_item(kind='url', url=url, title=e.get('title') or url, start=it['start'],
                                            end=it['end'], playlist=info.get('title') or '', clone=it.get('clone', False),
                                            accent=it.get('accent', 'original'),
                                            **{f'cvoice_{r}': it.get(f'cvoice_{r}', '') for r in VOICE_ROLES}))
                    with LOCK:
                        idx = STATE['items'].index(it)
                        STATE['items'][idx:idx + 1] = new
                    log(f"playlist {it['url']} -> {len(new)} items")
                    save_queue()
                else:
                    upd(it, status='queued', title=info.get('title') or it['url'], message='',
                        duration=info.get('duration') or 0)
            except Exception as e:
                upd(it, status='error', message=f'Could not read URL: {e}')

# --------------------------------------------------- worker ------------------------------------------------------
CURRENT = {'item': None, 'proc': None, 'cancel': None}

def to_sec(t):
    try:
        v = 0.0
        for part in str(t).strip().split(':'): v = v * 60 + float(part)
        return v
    except ValueError: return 0.0

def safe_name(s):
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', ' ', s).strip().strip('.')
    return re.sub(r'\s+', ' ', s)[:150] or 'video'

def unique(path):
    base, ext = os.path.splitext(path); i = 2
    while os.path.exists(path): path = f'{base} {i}{ext}'; i += 1
    return path

def run_proc(it, cmd, on_line, logname):
    logf = open(os.path.join(P.logs, f"{it['id']}.log"), 'a')
    logf.write(f"\n=== {time.strftime('%H:%M:%S')} {logname}: {' '.join(cmd)}\n"); logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_for_tools(),
                         bufsize=1, start_new_session=True, cwd=P.data)
    CURRENT['proc'] = p
    tail = []
    for line in p.stdout:
        logf.write(line); logf.flush()
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        try: on_line(line)
        except Exception: pass
    p.wait(); logf.close(); CURRENT['proc'] = None
    return p.returncode, tail

def kill_current():
    p = CURRENT.get('proc')
    if p and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            for _ in range(50):
                if p.poll() is not None: break
                time.sleep(0.1)
            if p.poll() is None: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass

def download(it):
    s = STATE['settings']; h = int(s.get('max_height', 720))
    d = os.path.join(P.downloads, it['id']); os.makedirs(d, exist_ok=True)
    have = [f for f in os.listdir(d) if f.lower().endswith(VIDEO_EXT) and '.part' not in f and not re.search(r'\.f\d+\.', f)]
    if have and os.path.exists(os.path.join(d, '.complete')): return os.path.join(d, have[0])
    cmd = ytdlp_base() + ['-f', f'bv*[height<={h}]+ba/b[height<={h}]/bv*+ba/b', '--merge-output-format', 'mkv',
                          '-o', '%(title).120B [%(id)s].%(ext)s', '-P', d, '--newline', '--no-playlist',
                          '--progress-template', 'download:VSPROG %(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s']
    if it.get('start') or it.get('end'):
        cmd += ['--download-sections', f"*{it.get('start') or '0'}-{it.get('end') or 'inf'}"]
    cmd += [it['url']]
    state = {'part': 0, 'last': 0.0}
    sec_len = 0.0
    if it.get('start') or it.get('end'):
        a0 = to_sec(it.get('start') or '0'); a1 = to_sec(it.get('end')) if it.get('end') else float(it.get('duration') or 0)
        sec_len = max(0.0, a1 - a0)
    def on_line(line):
        m = re.search(r'VSPROG\s+([\d.]+)%\|([^|]*)\|(.*)', line)
        if m:
            pct = float(m.group(1))
            if pct + 5 < state['last']: state['part'] += 1          # next format (video, then audio)
            state['last'] = pct
            frac = min(1.0, (state['part'] * 100 + pct) / 200 if state['part'] < 2 else 1.0)
            upd(it, stage='download', stage_pct=pct, pct=round(20 * frac, 1),
                message=f"Downloading {pct:.0f}%  {m.group(2).strip()}  ETA {m.group(3).strip()}")
        elif sec_len and (tm := re.search(r'time=(\d+):(\d+):([\d.]+)', line)):
            t = int(tm.group(1)) * 3600 + int(tm.group(2)) * 60 + float(tm.group(3))
            pct = min(100.0, 100 * t / sec_len)
            upd(it, stage='download', stage_pct=pct, pct=round(20 * pct / 100, 1), message=f'Downloading section {pct:.0f}%')
        elif '[Merger]' in line or '[ffmpeg]' in line:
            upd(it, message='Merging video and audio...')
        elif 'ERROR' in line:
            upd(it, message=line[-200:])
    upd(it, status='downloading', stage='download', message='Starting download...')
    rc, tail = run_proc(it, cmd, on_line, 'yt-dlp')
    if CURRENT['cancel']: return None
    if rc != 0:
        err = next((l for l in reversed(tail) if 'ERROR' in l), tail[-1] if tail else 'yt-dlp failed')
        hint = ''
        if re.search(r'Sign in|age|cookies|confirm you', err, re.I): hint = ' (try Settings > Browser cookies)'
        raise RuntimeError(err[-300:] + hint)
    have = [f for f in os.listdir(d) if f.lower().endswith(VIDEO_EXT) and '.part' not in f and not re.search(r'\.f\d+\.', f)]
    if not have: raise RuntimeError('download finished but no video file found')
    open(os.path.join(d, '.complete'), 'w').close()
    return os.path.join(d, max(have, key=lambda f: os.path.getsize(os.path.join(d, f))))

def jp_arg(v):
    """Japanese-voices level for the pipeline: 'off' at the slider minimum (or unset), else dB."""
    try: v = float(v)
    except (TypeError, ValueError): return 'off'
    return 'off' if v <= JP_OFF else f'{v:g}'

def voiceover(it, src, base_pct):
    s = STATE['settings']
    out_dir = os.path.expanduser(s['output_dir']); os.makedirs(out_dir, exist_ok=True)
    if not it.get('output'):
        title = it.get('title') or os.path.basename(src)
        if it['kind'] == 'file' or title.lower().endswith(VIDEO_EXT): title = os.path.splitext(title)[0]
        if it.get('start') or it.get('end'): title += f" [{(it.get('start') or '0').replace(':', '.')}-{(it.get('end') or 'end').replace(':', '.')}]"
        it['output'] = unique(os.path.join(out_dir, safe_name(title) + ' (English VO).mp4'))
        save_queue()
    out = it['output']
    wd = os.path.join(P.work, it['id'])
    cmd = [P.py, '-m', 'lotgh_vo', src, '-o', out, '--workdir', wd,
           '--voices', os.path.join(P.data, 'voices.json'), '--corrections', os.path.join(P.data, 'corrections.json'),
           '--models', P.models, '--jp-db', jp_arg(s.get('jp_db')), '--duck-db', str(s['duck_db']), '--tts-db', str(s.get('tts_db', 0)),
           '--keep-song-vocals' if s.get('keep_song_vocals', True) else '--no-keep-song-vocals',
           '--video-codec', s.get('video_codec', 'h264')]
    if s.get('voice_lyrics', 'skip') != 'skip': cmd += ['--no-skip-white']
    use_clone = bool(it.get('clone'))
    if use_clone and not clone_ready():
        use_clone = False; log(f"item {it['id']}: cloned voices requested but not installed; using Kokoro")
    if use_clone: cmd += ['--clone', '--clone-python', P.cpy]
    if use_clone and it.get('accent') == 'german_v3':
        if not de_ready():
            upd(it, status='processing', message='Downloading the German-accent model (2.1 GB)...')
            do_de_install()
        if de_ready(): cmd += ['--clone-accent', 'german_v3']
        else: log(f"item {it['id']}: German accent requested but its model is not available; using the original accent")
    for role in VOICE_ROLES if use_clone else ():
        vid = it.get(f'cvoice_{role}')
        if not vid: continue
        v = custom_voice(vid)
        if not v:
            log(f"item {it['id']}: custom {role} voice {vid!r} not found; using the voices from the video"); continue
        acc = v.get('accent', 'original')
        if acc == 'german_v3' and not de_ready():
            upd(it, status='processing', message='Downloading the German-accent model (2.1 GB)...')
            if not do_de_install():
                log(f"item {it['id']}: {v['name']} needs the German-accent model, which is not available; original accent"); acc = 'original'
        cmd += ['--clone-voice', f"{role}={v['name']}|{acc}|{v['path']}"]
    if it['kind'] == 'file' and (it.get('start') or it.get('end')):
        if it.get('start'): cmd += ['--start', it['start']]
        if it.get('end'): cmd += ['--end', it['end']]
    sw = CLONE_STAGE_WEIGHTS if use_clone else STAGE_WEIGHTS
    weights = dict(sw); order = [k for k, _ in sw]
    labels = {'ocr': 'Reading subtitles (OCR)', 'separate': 'Separating voices', 'tts': 'Generating English speech',
              'clone': 'Cloning voices (Chatterbox)',
              'mix': 'Mixing', 'mux': 'Writing video'}
    st = {'stage': 'ocr', 'pct': 0.0}
    progress = dict.fromkeys(order, 0.0)
    def overall():
        frac = sum(weights[k] * progress[k] / 100 for k in order)
        return round(base_pct + (100 - base_pct) * frac, 1)
    def on_line(line):
        m = re.match(r'^\[\d+:\d+:\d+\] (\w+)\s+(.*)$', line)
        if not m: return
        stage, msg = m.group(1), m.group(2)
        if stage in weights:
            if stage not in ('ocr', 'separate'):
                for previous in order[:order.index(stage)]: progress[previous] = 100.0
            st['stage'], st['pct'] = stage, progress[stage]
            pm = re.search(r'(\d+)/(\d+) \(\s*([\d.]+)%\)', msg)
            if pm and stage == st['stage']: st['pct'] = float(pm.group(3))
            if 'cached' in msg and stage == st['stage'] and re.search(r'\b0 to do', msg): st['pct'] = 100.0
            progress[stage] = max(progress[stage], st['pct'])
            nl = re.search(r'(\d+) dialogue lines', msg)
            if nl: it['lines'] = int(nl.group(1))
            extra = ''
            em = re.search(r'ETA\s+([\d.]+) min', msg)
            if em: extra = f' (ETA {float(em.group(1)):.0f} min)' if float(em.group(1)) >= 1 else ''
            text = labels[stage] + (f' {st["pct"]:.0f}%' if pm else '') + extra
            if stage == 'mux' and 'encoding English' in msg: text = 'Encoding audio'
            if stage == 'clone' and not pm:
                if 'grouping' in msg: text = 'Cloning voices: grouping speakers'
                elif 'loading' in msg: text = 'Cloning voices: loading Chatterbox'
                else: text = it.get('message') or labels['clone']
            if stage == 'mux' and 'writing' in msg: text = 'Writing video' + (' (H.264 encode)' if s.get('video_codec') == 'h264' else '')
            upd(it, stage=st['stage'], stage_pct=st['pct'], pct=overall(), message=text)
        elif stage == 'voices':
            vm = re.search(r'voicing (\d+) lines', msg)
            if vm: upd(it, message=f'{vm.group(1)} lines to voice')
    upd(it, status='processing', stage='ocr', message='Starting voice-over...', pct=base_pct)
    rc, tail = run_proc(it, cmd, on_line, 'voiceover')
    if CURRENT['cancel']: return False
    if rc == 3:
        base = os.path.splitext(out)[0]
        for suf in ('.en.ocr.srt', '.voiced.srt', '.captions.srt'):
            if os.path.exists(base + suf): os.remove(base + suf)
        shutil.rmtree(wd, ignore_errors=True); it['output'] = ''
        upd(it, status='no_hardsubs', message='No hardsubs found: OCR found almost no burned-in English subtitle lines.',
            stage='', pct=0); return False
    if rc != 0:
        err = next((l for l in reversed(tail) if re.search(r'Error|error|failed', l)), tail[-1] if tail else 'failed')
        raise RuntimeError(err[-300:])
    # tidy: subtitles into Subtitles/, drop work dir
    base = os.path.splitext(out)[0]; sd = os.path.join(os.path.dirname(out), 'Subtitles'); os.makedirs(sd, exist_ok=True)
    for suf in ('.en.ocr.srt', '.voiced.srt', '.captions.srt'):
        if os.path.exists(base + suf): os.replace(base + suf, os.path.join(sd, os.path.basename(base) + suf))
    shutil.rmtree(wd, ignore_errors=True)
    return True

def process(it):
    CURRENT['item'], CURRENT['cancel'] = it, None
    upd(it, message='', status='downloading' if it['kind'] == 'url' else 'processing')
    try:
        if it['kind'] == 'url':
            src = download(it)
            if src is None: return
            ok = voiceover(it, src, 20.0)
        else:
            if not os.path.exists(it['path']): raise RuntimeError('source file not found: ' + it['path'])
            src = it['path']; ok = voiceover(it, src, 0.0)
        if ok:
            if it['kind'] == 'url':
                d = os.path.join(P.downloads, it['id'])
                if STATE['settings'].get('keep_source'):
                    sdir = os.path.join(os.path.dirname(it['output']), 'Sources'); os.makedirs(sdir, exist_ok=True)
                    shutil.move(src, unique(os.path.join(sdir, os.path.basename(src))))
                shutil.rmtree(d, ignore_errors=True)
            upd(it, status='done', stage='', stage_pct=100, pct=100, message=f"Done - {it.get('lines', 0)} subtitle lines",
                finished=time.time())
    except Exception as e:
        if not CURRENT['cancel']:
            log(f"item {it['id']} failed: {traceback.format_exc()}")
            upd(it, status='error', message=str(e)[:400])
    finally:
        if CURRENT['cancel']:
            upd(it, status=CURRENT['cancel'], message='Stopped' if CURRENT['cancel'] == 'stopped' else '')
        CURRENT['item'] = None

def worker_loop():
    while True:
        time.sleep(0.5)
        if not setup_ready() or STATE['paused']: continue
        with LOCK: nxt = next((it for it in STATE['items'] if it['status'] == 'queued'), None)
        if nxt: process(nxt)

def cleanup_item(it):
    shutil.rmtree(os.path.join(P.work, it['id']), ignore_errors=True)
    shutil.rmtree(os.path.join(P.downloads, it['id']), ignore_errors=True)

# --------------------------------------------------- HTTP --------------------------------------------------------
TOKEN = secrets.token_urlsafe(16)

LOG_TAIL = 20000            # characters shown when the log view opens (and after a reset)
LOG_CHUNK = 512 * 1024      # most bytes returned by one incremental read

def read_log(pth, offset=None):
    """Without offset: the last LOG_TAIL characters + the byte offset to continue from. With offset: the complete
    lines written since then + the new offset; {'reset': True} with a fresh tail when the log was truncated or
    replaced (offset past its end) or when more was written than one read returns."""
    try: size = os.path.getsize(pth)
    except OSError: return {'log': '', 'offset': 0, 'reset': offset is not None}
    try: off = int(offset) if offset is not None else None
    except ValueError: off = None
    with open(pth, 'rb') as f:
        if off is None or off > size or size - off > LOG_CHUNK:
            f.seek(max(0, size - 4 * LOG_TAIL)); txt = f.read(size - f.tell()).decode('utf-8', 'replace')[-LOG_TAIL:]
            return {'log': txt, 'offset': size, 'reset': off is not None}
        f.seek(off); b = f.read(size - off)
    end = b.rfind(b'\n') + 1                     # whole lines only (never splits a line or a UTF-8 character)
    return {'log': b[:end].decode('utf-8', 'replace'), 'offset': off + end, 'reset': False}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(b))); self.end_headers()
        self.wfile.write(b)
    def _auth(self):
        if self.headers.get('X-Token') == TOKEN: return True
        self._json({'error': 'forbidden'}, 403); return False
    def do_GET(self):
        u = urlparse(self.path)
        if u.path.startswith('/api/'):
            if not self._auth(): return
            if u.path == '/api/state':
                with LOCK:
                    return self._json({'setup': STATE['setup'], 'items': STATE['items'], 'settings': STATE['settings'],
                                       'paused': STATE['paused'], 'current': CURRENT['item']['id'] if CURRENT['item'] else None,
                                       'version': VERSION, 'data_dir': P.data, 'clone': STATE['clone'],
                                       'custom_voices': [{k: v[k] for k in ('id', 'name', 'accent', 'version', 'author')}
                                                         for v in custom_voices()],
                                       'plugin_errors': PLUGINS['errors']})
            m = re.match(r'/api/log/(\w+)$', u.path)
            if m:                                          # ?offset=N: only what was written since (live log view)
                q = parse_qs(u.query)
                return self._json(read_log(os.path.join(P.logs, f'{m.group(1)}.log'), q['offset'][0] if 'offset' in q else None))
            return self._json({'error': 'not found'}, 404)
        name = 'index.html' if u.path in ('/', '/index.html') else u.path.lstrip('/')
        fp = os.path.normpath(os.path.join(P.ui, name))
        if not fp.startswith(P.ui) or not os.path.isfile(fp):
            self.send_response(404); self.end_headers(); return
        ctype = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css',
                 '.png': 'image/png', '.svg': 'image/svg+xml'}.get(os.path.splitext(fp)[1], 'application/octet-stream')
        b = open(fp, 'rb').read()
        self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        if not self._auth(): return
        n = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(n) or b'{}') if n else {}
        path = urlparse(self.path).path
        try:
            r = handle_post(path, body)
            self._json(r if r is not None else {'ok': True})
        except Exception as e:
            log('api error ' + traceback.format_exc()); self._json({'error': str(e)}, 400)

def handle_post(path, b):
    cv = {f'cvoice_{r}': b.get(f'cvoice_{r}') or '' for r in VOICE_ROLES}
    if path in ('/api/add', '/api/add_files'):
        clean_voice_ids(cv, 'cvoice_')
        if any(cv.values()): b['clone'] = True              # a custom voice is a cloned voice
    want_clone = bool(b.get('clone')) and clone_ready()
    accent = b.get('accent') if b.get('accent') in ACCENTS and want_clone else 'original'
    if path in ('/api/add', '/api/add_files'):
        if accent != 'original': start_de_install()
        start_voice_models(cv, 'cvoice_')
    if path == '/api/add':
        return {'added': add_text(b.get('text', ''), b.get('start', ''), b.get('end', ''), want_clone, accent, cv)}
    if path == '/api/add_files':
        add_files(b.get('paths', []), b.get('start', ''), b.get('end', ''), want_clone, accent, cv); return None
    if path == '/api/plugins/install':                     # {path}: a voice plugin zip
        return {'id': install_plugin(b.get('path', ''))}
    if path == '/api/plugins/remove':                      # {id}
        remove_plugin(b.get('id', '')); return None
    if path == '/api/clone/install_de':
        if not clone_ready(): raise ValueError('install cloned voices first')
        start_de_install(); return None
    if path == '/api/clone/optin':                       # checkbox on the first-run screen
        with LOCK: STATE['settings']['clone_optin'] = bool(b.get('on'))
        save_settings()
        if b.get('on') and setup_ready() and not clone_ready():
            threading.Thread(target=do_clone_install, daemon=True).start()
        return None
    if path == '/api/clone/install':
        if not setup_ready(): raise ValueError('finish the basic setup first')
        with LOCK: STATE['settings']['clone_optin'] = True
        save_settings(); threading.Thread(target=do_clone_install, daemon=True).start(); return None
    if path == '/api/clone/remove':
        if CURRENT['item'] is not None and CURRENT['item'].get('clone'):
            raise ValueError('a video using cloned voices is being processed; stop it first')
        if DE_LOCK.locked(): raise ValueError('the German-accent model is still downloading; try again when it has finished')
        threading.Thread(target=do_clone_remove, daemon=True).start(); return None
    if path == '/api/pause':
        STATE['paused'] = bool(b.get('paused')); save_queue(); return None
    if path == '/api/settings':
        with LOCK:
            for k, v in b.items():
                if k in default_settings() and k != 'clone_optin': STATE['settings'][k] = v
            STATE['settings']['keep_song_vocals'] = bool(STATE['settings'].get('keep_song_vocals', True))
            if STATE['settings'].get('clone_accent') not in ACCENTS: STATE['settings']['clone_accent'] = 'original'
            if not clone_ready(): STATE['settings']['clone_default'] = False; STATE['settings']['clone_accent'] = 'original'
            clean_voice_ids(STATE['settings'], 'clone_voice_')
        if STATE['settings']['clone_accent'] != 'original': start_de_install()     # first time German is chosen
        start_voice_models(STATE['settings'], 'clone_voice_')
        save_settings(); return None
    if path == '/api/setup/retry':
        if STATE['setup'].get('state') != 'running': threading.Thread(target=do_setup, daemon=True).start()
        return None
    if path == '/api/update_ytdlp':
        def upd_y():
            r = subprocess.run([P.tool('uv'), 'pip', 'install', '--python', P.py, '-U', 'yt-dlp[default]'],
                               capture_output=True, text=True, env=env_for_tools())
            log('yt-dlp update: ' + (r.stdout + r.stderr)[-500:])
        threading.Thread(target=upd_y, daemon=True).start(); return None
    if path == '/api/clear_done':
        with LOCK: STATE['items'] = [i for i in STATE['items'] if i['status'] != 'done']
        save_queue(); return None
    m = re.match(r'/api/item/(\w+)/(\w+)$', path)
    if m:
        iid, act = m.groups()
        with LOCK:
            it = find(iid)
            if not it: raise ValueError('no such item')
            items = STATE['items']; i = items.index(it)
        running = CURRENT['item'] is it
        if act == 'remove':
            if running: CURRENT['cancel'] = 'removed'; kill_current()
            with LOCK: items.remove(it)
            cleanup_item(it)
        elif act == 'stop' and running:
            CURRENT['cancel'] = 'stopped'; kill_current()
        elif act == 'retry' and not running:
            upd(it, status='expanding' if it['kind'] == 'url' and it.get('title') == it.get('url') else 'queued',
                message='', pct=0, stage='')
        elif act in ('clone_on', 'clone_off') and not running and it['status'] != 'done':
            if act == 'clone_on' and not clone_ready(): raise ValueError('cloned voices are not installed')
            with LOCK: it['clone'] = act == 'clone_on'
        elif act in ('accent_german_v3', 'accent_original') and not running and it['status'] != 'done':
            if act == 'accent_german_v3' and not clone_ready(): raise ValueError('cloned voices are not installed')
            with LOCK: it['accent'] = act[len('accent_'):]
            if act == 'accent_german_v3': start_de_install()
        elif act == 'top':
            # top of the pending queue: right after the running item (which keeps going), else the very top
            if running or it['status'] == 'done': raise ValueError('only waiting items can be moved to the top')
            with LOCK:
                cur = CURRENT['item']; r = items.index(cur) if cur is not None and cur in items else -1
                items.pop(i)
                items.insert(r + 1 if 0 <= r < i else 0, it)
        elif act in ('up', 'down'):
            with LOCK:
                items.pop(i)
                j = max(0, i - 1) if act == 'up' else min(len(items), i + 1)
                items.insert(j, it)
        save_queue(); return None
    raise ValueError('unknown endpoint')

# --------------------------------------------------- main --------------------------------------------------------
def watchdog(ppid):
    while True:
        time.sleep(2)
        if ppid and os.getppid() != ppid:
            log('parent app exited; shutting down'); shutdown()

def shutdown(*_):
    CURRENT['cancel'] = CURRENT['cancel'] or 'queued'     # interrupted item resumes next launch
    kill_current()
    try: save_queue()
    except Exception: pass
    os._exit(0)

def main():
    global P
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--port', type=int, default=0); ap.add_argument('--dev', action='store_true')
    ap.add_argument('--watch-parent', action='store_true')
    a = ap.parse_args()
    P = Paths(a)
    saved = load(os.path.join(P.data, 'settings.json'), {})
    s = default_settings(); s.update(saved)
    if saved and not saved.get('mix_v3'):
        # v0.3.0: the Japanese voices are left out of the English mix by default. Users still on the old default
        # (-15 dB) move to Off; a level they chose themselves is kept.
        if saved.get('jp_db', -15) == -15: s['jp_db'] = JP_OFF
    s['mix_v3'] = True
    STATE['settings'] = s; save_settings()
    if not os.path.exists(os.path.join(P.data, 'corrections.json')):
        shutil.copy(os.path.join(P.pipeline, 'config', 'corrections.json'), os.path.join(P.data, 'corrections.json'))
    q = load(os.path.join(P.data, 'queue.json'), {})
    STATE['items'] = q.get('items', []); STATE['paused'] = q.get('paused', False)
    for it in STATE['items']:                                    # resume interrupted work
        if it['status'] in ('downloading', 'processing'): it['status'] = 'queued'; it['message'] = 'Resuming...'
    save_queue()
    signal.signal(signal.SIGTERM, shutdown); signal.signal(signal.SIGINT, shutdown)
    old_cache = os.path.join(P.data, 'uv-cache')                # left by v1.0.0 installs (~1 GB)
    if os.path.isdir(old_cache):
        threading.Thread(target=lambda: (shutil.rmtree(old_cache, ignore_errors=True), log('removed old uv-cache')), daemon=True).start()
    for it in STATE['items']:
        it.setdefault('clone', False); it.setdefault('accent', 'original'); it.setdefault('cvoice_narrator', ''); it.setdefault('cvoice_dialogue', '')
    scan_plugins(); migrate_accents(); save_settings(); save_queue()
    clone_refresh()
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    for fn in (do_setup, expander_loop, worker_loop):
        threading.Thread(target=fn, daemon=True).start()
    if a.watch_parent: threading.Thread(target=watchdog, args=(os.getppid(),), daemon=True).start()
    log(f'{APP_NAME} {VERSION} data={P.data}')
    print(f'VS_READY http://127.0.0.1:{srv.server_address[1]}/?t={TOKEN}', flush=True)
    srv.serve_forever()

if __name__ == '__main__':
    main()
