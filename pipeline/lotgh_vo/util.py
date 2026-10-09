import hashlib, json, os, platform, shutil, subprocess, sys, time

_T0 = time.time()
def log(stage, msg):
    el = time.time() - _T0
    print(f"[{int(el // 3600):02d}:{int(el % 3600 // 60):02d}:{int(el % 60):02d}] {stage:<8} {msg}", flush=True)

class Progress:
    """Tiny progress/ETA printer (no external deps)."""
    def __init__(self, stage, total, every=5.0):
        self.stage, self.total, self.done, self.t0, self.last, self.every = stage, max(1, total), 0, time.time(), 0.0, every
    def step(self, n=1, extra=''):
        self.done += n; now = time.time()
        if now - self.last >= self.every or self.done >= self.total:
            self.last = now; el = now - self.t0
            eta = el / self.done * (self.total - self.done) if self.done else 0
            log(self.stage, f"{self.done}/{self.total} ({100 * self.done / self.total:4.1f}%)  elapsed {el / 60:5.1f} min  "
                            f"ETA {eta / 60:5.1f} min {extra}")

def which_ffmpeg(name='ffmpeg'):
    env = os.environ.get(f'LOTGH_{name.upper()}')          # app bundle passes its own static binaries
    if env and os.path.exists(env): return env
    for p in (f'/opt/homebrew/bin/{name}', f'/usr/local/bin/{name}'):
        if os.path.exists(p): return p
    p = shutil.which(name)
    if not p: sys.exit(f'{name} not found (brew install ffmpeg)')
    return p
FFMPEG, FFPROBE = which_ffmpeg('ffmpeg'), which_ffmpeg('ffprobe')

def run(cmd, quiet=True):
    r = subprocess.run(cmd, stdout=subprocess.PIPE if quiet else None, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r

def probe(path):
    j = json.loads(subprocess.check_output([FFPROBE, '-v', 'error', '-show_entries',
                                            'stream=index,codec_type,codec_name,width,height:format=duration',
                                            '-of', 'json', path]))
    v = next(s for s in j['streams'] if s['codec_type'] == 'video')
    return dict(width=v['width'], height=v['height'], vcodec=v['codec_name'], duration=float(j['format']['duration']),
                has_audio=any(s['codec_type'] == 'audio' for s in j['streams']))

def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def load_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default

def h(*parts):
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:16]

def is_mac(): return platform.system() == 'Darwin'

def cpu_count():
    try:
        if is_mac():   # performance cores only; efficiency cores slow down CPU-bound pools
            return int(subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.physicalcpu']).strip())
    except Exception: pass
    return os.cpu_count() or 4

_HWDEC = {}
def hwdec_args(mode, video):
    """ffmpeg input args for hardware decode. 'auto' = VideoToolbox on macOS if it decodes this file."""
    if mode == 'none': return []
    if mode == 'auto' and not is_mac(): return []
    key = (mode, video)
    if key not in _HWDEC:
        args = ['-hwaccel', 'videotoolbox']
        try:
            r = subprocess.run([FFMPEG, '-v', 'error', *args, '-t', '2', '-i', video, '-an', '-f', 'null', '-'],
                               capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0 and 'Failed' not in r.stderr and 'not supported' not in r.stderr.lower()
        except Exception: ok = False
        _HWDEC[key] = args if ok else []
        log('hwdec', 'VideoToolbox decode ' + ('enabled' if ok else 'unavailable for this codec/GPU -> software decode'))
    return _HWDEC[key]

# ---------------- onnxruntime providers ----------------
def ort_session(model_path, provider='cpu', threads=0):
    import onnxruntime as rt
    so = rt.SessionOptions(); so.log_severity_level = 3
    if threads: so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    if provider == 'coreml':
        for opts in ({'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}, {}):
            try:
                s = rt.InferenceSession(model_path, sess_options=so,
                                        providers=[('CoreMLExecutionProvider', opts), 'CPUExecutionProvider'])
                if 'CoreMLExecutionProvider' in s.get_providers(): return s
            except Exception: pass
        raise RuntimeError('CoreML provider failed to initialise')
    return rt.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

def coreml_available():
    try:
        import onnxruntime as rt
        return 'CoreMLExecutionProvider' in rt.get_available_providers()
    except Exception: return False

def pick_provider(name, pref, bench):
    """pref: auto|coreml|cpu. bench(provider)->seconds. auto = CoreML only if present, working and faster."""
    if pref == 'cpu' or (pref != 'coreml' and not coreml_available()):
        log(name, 'onnxruntime provider: CPU' + ('' if coreml_available() else ' (CoreML EP not available)')); return 'cpu'
    try:
        tc = bench('coreml')
    except Exception as e:
        log(name, f'CoreML failed ({str(e)[:120]}) -> CPU'); return 'cpu'
    if pref == 'coreml':
        log(name, f'onnxruntime provider: CoreML (forced), {tc:.2f}s/bench'); return 'coreml'
    tcpu = bench('cpu')
    choice = 'coreml' if tc < tcpu * 0.9 else 'cpu'
    log(name, f'benchmark CoreML {tc:.2f}s vs CPU {tcpu:.2f}s -> {choice.upper()}')
    return choice
