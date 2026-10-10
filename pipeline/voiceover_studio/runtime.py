"""Locate the app's code, tools and private environments without a Git checkout."""
import os
import json
from pathlib import Path
import plistlib
import shutil

CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path.home() / 'Library/Application Support/Voiceover Studio'


def data_dir():
    return Path(os.environ.get('VOICEOVER_STUDIO_DATA_DIR', DEFAULT_DATA)).expanduser().resolve()


def version(root=CODE_ROOT):
    for plist in (root.parent.parent / 'Info.plist', root / 'swift/Info.plist'):
        if plist.is_file():
            with plist.open('rb') as stream:
                return plistlib.load(stream)['CFBundleShortVersionString']
    return 'development'


def tool(name, root=CODE_ROOT):
    # In a bundle always use that bundle's tools, including when the app was moved.
    if root.parent.name == 'Resources':
        return str(root.parent / 'bin' / name)
    override = os.environ.get('LOTGH_' + name.upper())
    if override:
        return override
    for candidate in (root / 'cache' / name,
                      Path('/Applications/Voiceover Studio.app/Contents/Resources/bin') / name):
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name) or name


def environment(root=CODE_ROOT, data=None, tool_root=None):
    data = data or data_dir()
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONUSERBASE', 'PYTHONINSPECT', 'PYTHONSTARTUP', 'VO_BENCH_RESTARTS'):
        env.pop(key, None)
    env.update(PYTHONPATH=str(root / 'pipeline'), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               PYTHONPYCACHEPREFIX=str(data / 'pycache'), VOICEOVER_STUDIO_DATA_DIR=str(data),
               HF_HOME=str(data / 'hf'), TORCH_HOME=str(data / 'torch'), HF_HUB_OFFLINE='1',
               LOTGH_CLONE_MODEL=str(data / 'models/chatterbox'), PYTORCH_ENABLE_MPS_FALLBACK='1',
               TOKENIZERS_PARALLELISM='false', TQDM_DISABLE='1', HF_HUB_DISABLE_TELEMETRY='1')
    for name in ('ffmpeg', 'ffprobe'):
        env['LOTGH_' + name.upper()] = tool(name, tool_root or root)
    env['PATH'] = os.pathsep.join([str(Path(env['LOTGH_FFMPEG']).parent), str(data / 'bin'),
                                   str(data / 'venv/bin'), '/usr/bin', '/bin', '/usr/sbin', '/sbin'])
    return env


def diagnostics(root=CODE_ROOT, data=None):
    data = data or data_dir()
    try:
        setup = json.loads((data / 'setup_done.json').read_text())
    except (OSError, ValueError):
        setup = {}
    if not isinstance(setup, dict):
        setup = {}
    checks = {'python': (data / 'venv/bin/python').is_file(),
              'ffmpeg': os.access(tool('ffmpeg', root), os.X_OK),
              'ffprobe': os.access(tool('ffprobe', root), os.X_OK),
              'kokoro_model': any((data / 'models' / n).is_file() for n in ('kokoro-v1.0.fp16.onnx', 'kokoro-v1.0.onnx')),
              'kokoro_voices': (data / 'models/voices-v1.0.bin').is_file(),
              'setup_complete': bool(setup.get('check'))}
    return {'version': version(root), 'code': str(root), 'data': str(data), 'checks': checks,
            'ready': all(checks.values()),
            'clone_installed': (data / 'clone-venv/bin/python').is_file() and (data / 'clone_done.json').is_file(),
            'german_installed': (data / 'clone_de_v3_done.json').is_file()}
