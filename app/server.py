#!/usr/bin/env python3
"""Voiceover Studio backend: first-run setup, persistent download/voice-over queue, small JSON API + static UI.

Standard library only (it runs before the heavy dependencies are installed). The voice-over itself runs as
`python -m lotgh_vo` subprocesses in the private venv.
"""
import argparse, hashlib, json, os, re, secrets, shutil, signal, subprocess, threading, time, traceback, uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

APP_NAME = 'Voiceover Studio'
VERSION = '1.0.0'
KOKORO_FILES = [
    ('kokoro-v1.0.fp16.onnx', 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx', 177464787),
    ('voices-v1.0.bin', 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin', 28214398),
]
OLD_MODEL_FILES = ['kokoro-v1.0.onnx']            # fp32 model used by v1.0.0; replaced by the fp16 one
DENO_URL = 'https://github.com/denoland/deno/releases/latest/download/deno-aarch64-apple-darwin.zip'
STAGE_WEIGHTS = [('ocr', 0.40), ('separate', 0.18), ('tts', 0.30), ('mix', 0.07), ('mux', 0.05)]
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
        self.bin = os.path.join(self.res, 'bin')                  # bundled uv, ffmpeg, ffprobe
        self.dbin = os.path.join(self.data, 'bin')                # downloaded deno
        self.models = os.path.join(self.data, 'models')
        self.torch = os.path.join(self.data, 'torch')
        self.downloads = os.path.join(self.data, 'downloads')
        self.work = os.path.join(self.data, 'work')
        self.logs = os.path.join(self.data, 'logs')
        for d in (self.data, self.dbin, self.models, self.torch, self.downloads, self.work, self.logs):
            os.makedirs(d, exist_ok=True)
        self.dev = a.dev

    def tool(self, name):
        for d in (self.bin, self.dbin):
            p = os.path.join(d, name)
            if os.path.exists(p): return p
        return shutil.which(name) if self.dev else os.path.join(self.bin, name)

P = None
LOCK = threading.RLock()
STATE = {'setup': {}, 'items': [], 'settings': {}, 'paused': False}
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
                voice_lyrics='skip', jp_db=-15, duck_db=-9, tts_db=0, keep_source=False, video_codec='h264',
                cookies='none', max_height=720)

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

# --------------------------------------------------- queue ------------------------------------------------------
def new_item(**kw):
    it = dict(id=uuid.uuid4().hex[:10], kind='url', url='', path='', title='', start='', end='', status='queued',
              stage='', stage_pct=0.0, pct=0.0, message='', output='', added=time.time(), updated=time.time(), lines=0)
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

def add_text(text, start='', end=''):
    added = []
    with LOCK:
        for raw in text.splitlines():
            s = raw.strip().strip('"\'')
            if not s: continue
            if URL_RE.match(s):
                added.append(new_item(kind='url', url=s, title=s, status='expanding', message='Reading video info...',
                                      start=start, end=end))
            elif os.path.exists(os.path.expanduser(s)):
                pth = os.path.abspath(os.path.expanduser(s))
                added.append(new_item(kind='file', path=pth, title=os.path.basename(pth), start=start, end=end))
        STATE['items'].extend(added)
    save_queue()
    return len(added)

def add_files(paths, start='', end=''):
    with LOCK:
        for pth in paths:
            if os.path.isfile(pth):
                STATE['items'].append(new_item(kind='file', path=pth, title=os.path.basename(pth), start=start, end=end))
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
                                            end=it['end'], playlist=info.get('title') or ''))
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
           '--models', P.models, '--jp-db', str(s['jp_db']), '--duck-db', str(s['duck_db']), '--tts-db', str(s.get('tts_db', 0)),
           '--video-codec', s.get('video_codec', 'h264')]
    if s.get('voice_lyrics', 'skip') != 'skip': cmd += ['--no-skip-white']
    if it['kind'] == 'file' and (it.get('start') or it.get('end')):
        if it.get('start'): cmd += ['--start', it['start']]
        if it.get('end'): cmd += ['--end', it['end']]
    weights = dict(STAGE_WEIGHTS); order = [k for k, _ in STAGE_WEIGHTS]
    labels = {'ocr': 'Reading subtitles (OCR)', 'separate': 'Separating voices', 'tts': 'Generating English speech',
              'mix': 'Mixing', 'mux': 'Writing video'}
    st = {'stage': 'ocr', 'pct': 0.0}
    def overall():
        done = sum(weights[k] for k in order[:order.index(st['stage'])])
        frac = done + weights[st['stage']] * st['pct'] / 100
        return round(base_pct + (100 - base_pct) * frac, 1)
    def on_line(line):
        m = re.match(r'^\[\d+:\d+:\d+\] (\w+)\s+(.*)$', line)
        if not m: return
        stage, msg = m.group(1), m.group(2)
        if stage in weights:
            if order.index(stage) > order.index(st['stage']): st['stage'], st['pct'] = stage, 0.0
            pm = re.search(r'(\d+)/(\d+) \(\s*([\d.]+)%\)', msg)
            if pm and stage == st['stage']: st['pct'] = float(pm.group(3))
            if 'cached' in msg and stage == st['stage'] and re.search(r'\b0 to do', msg): st['pct'] = 100.0
            nl = re.search(r'(\d+) dialogue lines', msg)
            if nl: it['lines'] = int(nl.group(1))
            extra = ''
            em = re.search(r'ETA\s+([\d.]+) min', msg)
            if em: extra = f' (ETA {float(em.group(1)):.0f} min)' if float(em.group(1)) >= 1 else ''
            text = labels[stage] + (f' {st["pct"]:.0f}%' if pm else '') + extra
            if stage == 'mux' and 'encoding English' in msg: text = 'Encoding audio'
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
                                       'version': VERSION, 'data_dir': P.data})
            m = re.match(r'/api/log/(\w+)$', u.path)
            if m:
                pth = os.path.join(P.logs, f'{m.group(1)}.log')
                txt = open(pth, errors='replace').read()[-20000:] if os.path.exists(pth) else ''
                return self._json({'log': txt})
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
    if path == '/api/add':
        return {'added': add_text(b.get('text', ''), b.get('start', ''), b.get('end', ''))}
    if path == '/api/add_files':
        add_files(b.get('paths', []), b.get('start', ''), b.get('end', '')); return None
    if path == '/api/pause':
        STATE['paused'] = bool(b.get('paused')); save_queue(); return None
    if path == '/api/settings':
        with LOCK:
            for k, v in b.items():
                if k in default_settings(): STATE['settings'][k] = v
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
        elif act in ('up', 'down', 'top'):
            with LOCK:
                items.pop(i)
                j = 0 if act == 'top' else max(0, i - 1) if act == 'up' else min(len(items), i + 1)
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
    s = default_settings(); s.update(load(os.path.join(P.data, 'settings.json'), {}))
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
    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    for fn in (do_setup, expander_loop, worker_loop):
        threading.Thread(target=fn, daemon=True).start()
    if a.watch_parent: threading.Thread(target=watchdog, args=(os.getppid(),), daemon=True).start()
    log(f'{APP_NAME} {VERSION} data={P.data}')
    print(f'VS_READY http://127.0.0.1:{srv.server_address[1]}/?t={TOKEN}', flush=True)
    srv.serve_forever()

if __name__ == '__main__':
    main()
