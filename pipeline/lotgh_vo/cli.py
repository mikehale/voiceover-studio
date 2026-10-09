"""Entry point: python -m lotgh_vo INPUT [-o OUTPUT] [options]"""
import argparse, os, re, shutil, sys, time
from .util import FFMPEG, log, probe, run, cpu_count, load_json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGES = ['ocr', 'separate', 'tts', 'mix', 'mux']          # (+ 'clone' before tts with --clone)


SONG_CLASSES = ('white',)        # OP/ED lyric subtitles
def song_spans(subs, dur, gap=6.0, pad=1.0):
    """Time spans of songs from lyric subtitles: merged when < gap apart, padded on both sides."""
    out = []
    for s, e, text, cls in sorted(subs, key=lambda x: x[0]):
        if cls not in SONG_CLASSES: continue
        if out and s - out[-1][1] < gap: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return [[round(max(0.0, x - pad), 3), round(min(dur, y + pad), 3)] for x, y in out]

def _jp_level(v):
    return None if str(v).strip().lower() in ('off', 'none', '') else float(v)

def parse_args(argv=None):
    cores = cpu_count()
    ap = argparse.ArgumentParser(prog='lotgh_voiceover', description='English voice-over from burned-in subtitles.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', help='video file (e.g. .webm with burned-in English subs)')
    ap.add_argument('-o', '--output', help='output video (.mp4 or .mkv); default <input>_en_voiceover.mp4')
    ap.add_argument('--workdir', help='cache dir (resumable); default <output>.work/')
    g = ap.add_argument_group('range (for samples)')
    g.add_argument('--start', type=str, help='start time (seconds or hh:mm:ss); clip is cut by stream copy')
    g.add_argument('--end', type=str, help='end time (seconds or hh:mm:ss)')
    g = ap.add_argument_group('subtitles')
    g.add_argument('--srt', help='use this SRT instead of OCR (times relative to the input/clip)')
    g.add_argument('--corrections', default=os.path.join(HERE, 'config', 'corrections.json'))
    g.add_argument('--ocr-fps', type=float, default=5); g.add_argument('--ocr-y0', type=int, default=530)
    g.add_argument('--ocr-y1', type=int, default=0, help='0 = frame bottom')
    g.add_argument('--min-lines-per-min', type=float, default=0.5,
                   help='if OCR finds fewer dialogue lines than this (and at least 3), stop with exit code 3 '
                        '("no hardsubs found") instead of producing a silent voice-over; 0 disables')
    g.add_argument('--ocr-thresh', type=float, default=0.28); g.add_argument('--ocr-chunk', type=float, default=300)
    g = ap.add_argument_group('voices')
    g.add_argument('--voices', default=os.path.join(HERE, 'config', 'voices.json'), help='colour -> voice map')
    g.add_argument('--voice', action='append', default=[], metavar='CLASS=VOICE',
                   help='override, e.g. --voice cyan=bm_lewis --voice yellow=bm_george (VOICE=skip to mute)')
    g.add_argument('--skip-white', action=argparse.BooleanOptionalAction, default=True,
                   help='skip white (song lyric) lines; --no-skip-white voices them with the yellow voice')
    g.add_argument('--voice-speed', action='append', default=[], metavar='VOICE=X',
                   help='base speaking speed per voice, e.g. bm_daniel=1.12 (also "speeds" in voices.json)')
    g.add_argument('--max-tempo', type=float, default=1.4, help='max speed-up to fit a line in its slot')
    g = ap.add_argument_group('cloned voices (optional, needs the Chatterbox venv)')
    g.add_argument('--clone', action='store_true',
                   help="voice lines with Chatterbox in the original actors' voices (much slower; Kokoro fallback per line)")
    g.add_argument('--clone-python', help='python of the venv that has chatterbox-tts installed')
    g.add_argument('--clone-device', choices=['auto', 'mps', 'cpu'], default='auto', help='Chatterbox device')
    g.add_argument('--clone-accent', choices=['original', 'german'], default='original',
                   help='accent of the cloned voices: original (as the reference) or german (Chatterbox Multilingual)')
    g = ap.add_argument_group('mix levels (dB)')
    g.add_argument('--jp-db', type=_jp_level, default=None,
                   help="Japanese vocals level in dB, or 'off' (default): the vocals stem is left out of the English mix "
                        "and nothing is ducked. The original Japanese audio is always kept as the second audio track.")
    g.add_argument('--duck-db', type=float, default=-9, help='extra Japanese attenuation while English plays')
    g.add_argument('--tts-db', type=float, default=0, help='English voice level')
    g.add_argument('--keep-song-vocals', action=argparse.BooleanOptionalAction, default=True,
                   help='keep the vocals stem at full level (no ducking) during songs, i.e. around white lyric subtitles '
                        '(merged when < 6 s apart, padded 1 s with short fades), whatever --jp-db is')
    g.add_argument('--duck-hold', type=float, default=0.15); g.add_argument('--duck-smooth', type=float, default=0.3)
    g = ap.add_argument_group('performance')
    g.add_argument('--jobs', type=int, default=max(1, cores), help='OCR worker processes')
    g.add_argument('--tts-jobs', type=int, default=max(1, min(4, cores // 2)), help='Kokoro worker processes')
    g.add_argument('--device', choices=['auto', 'mps', 'cpu'], default='auto', help='Demucs device')
    g.add_argument('--onnx-provider', choices=['auto', 'coreml', 'cpu'], default='cpu',
                   help='RapidOCR: auto = CoreML only if available AND faster in a quick benchmark')
    g.add_argument('--tts-provider', choices=['auto', 'coreml', 'cpu'], default='cpu',
                   help='Kokoro: CPU by default (CoreML fails on its dynamic shapes)')
    g.add_argument('--hwdec', choices=['auto', 'videotoolbox', 'none'], default='auto')
    g.add_argument('--sep-chunk', type=float, default=300, help='Demucs/mix chunk length (s); bounds RAM use')
    g = ap.add_argument_group('output')
    g.add_argument('--video-codec', choices=['copy', 'h264'], default='copy',
                   help='copy keeps AV1 (QuickTime needs M3+ for AV1; use IINA/VLC or h264)')
    g.add_argument('--abitrate', default='192k')
    g.add_argument('--keep-stems', action='store_true', help='also write per-chunk English-only tracks')
    g.add_argument('--keep-temp', action='store_true', help='keep float chunk mixes after encoding')
    g.add_argument('--force', default='', help=f'comma list of stages to recompute: {",".join(STAGES)}')
    g.add_argument('--until', choices=STAGES, help='stop after this stage')
    g.add_argument('--models', default=os.path.join(HERE, 'models'))
    a = ap.parse_args(argv)
    a.force = {f'force_{s.strip()}' for s in a.force.split(',') if s.strip()}
    a.cpu_cores = cores
    a.kokoro_model = os.path.join(a.models, 'kokoro-v1.0.fp16.onnx')          # fp16 (170 MB); fp32 still accepted
    if not os.path.exists(a.kokoro_model): a.kokoro_model = os.path.join(a.models, 'kokoro-v1.0.onnx')
    a.kokoro_voices = os.path.join(a.models, 'voices-v1.0.bin')
    return a

def tsec(x):
    if x is None: return None
    p = [float(v) for v in str(x).split(':')]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))

def voice_map(a):
    cfg = load_json(a.voices, {}) or {}
    cm = dict(cfg.get('classes', {}))
    for ov in a.voice:
        k, _, v = ov.partition('=')
        cm[k.strip()] = v.strip()
    if not a.skip_white and cm.get('white', 'skip') == 'skip': cm['white'] = cm.get('yellow', 'bm_george')
    if a.skip_white: cm['white'] = 'skip'
    return cm

def lang_for(voice): return 'en-us' if voice.startswith('a') else 'en-gb'

def parse_srt(path):
    from .textfix import color_class
    txt = open(path, encoding='utf-8-sig').read().replace('\r', '')
    out = []
    for b in re.split(r'\n\s*\n', txt):
        L = b.strip().split('\n')
        for i, l in enumerate(L):
            if '-->' in l:
                s, e = [tsec(x.strip().replace(',', '.')) for x in l.split('-->')]
                body = ' '.join(L[i + 1:])
                m = re.search(r'<font color="(#[0-9a-fA-F]{6})"', body)
                body = re.sub(r'<[^>]+>', '', body).strip()
                if body: out.append([s, e, body, color_class(m.group(1)) if m else 'other'])
                break
    return out

def main(argv=None):
    a = parse_args(argv)
    if not os.path.exists(a.kokoro_model): sys.exit(f'Kokoro model missing in {a.models}; run ./setup.sh')
    src = os.path.abspath(a.input)
    base = os.path.splitext(src)[0]
    rng = ''
    if a.start or a.end:
        rng = f"_{int(tsec(a.start) or 0)}-{int(tsec(a.end)) if a.end else 'end'}"
    out = os.path.abspath(a.output or f'{base}{rng}_en_voiceover.mp4')
    wd = os.path.abspath(a.workdir or os.path.splitext(out)[0] + '.work'); os.makedirs(wd, exist_ok=True)
    log('setup', f'input {src}\n{"":17}output {out}\n{"":17}workdir {wd} (delete it to start over)')
    video = src
    if a.start or a.end:
        clip = os.path.join(wd, 'clip' + os.path.splitext(src)[1])
        if not os.path.exists(clip):
            cmd = [FFMPEG, '-v', 'error', '-y']
            if a.start: cmd += ['-ss', str(tsec(a.start))]
            if a.end: cmd += ['-to', str(tsec(a.end))]
            run(cmd + ['-i', src, '-map', '0', '-c', 'copy', clip + '.part' + os.path.splitext(src)[1]])
            os.replace(clip + '.part' + os.path.splitext(src)[1], clip)
        video = clip
        log('setup', f'working on clip {a.start or 0} -> {a.end or "end"} (stream copy; starts on a keyframe)')
    info = probe(video)
    log('setup', f"{info['vcodec']} {info['width']}x{info['height']}, {info['duration'] / 60:.1f} min, "
                 f"{a.cpu_cores} perf cores, OCR jobs {a.jobs}, TTS jobs {a.tts_jobs}")
    stop = lambda s: a.until == s
    T = {}
    # 1. subtitles
    t = time.time()
    if a.srt:
        dsrt = a.srt; log('ocr', f'using provided SRT {a.srt}')
    else:
        from .ocr import run_ocr
        dsrt, csrt = run_ocr(a, video, info, wd)
        shutil.copy(dsrt, os.path.splitext(out)[0] + '.en.ocr.srt')
        shutil.copy(csrt, os.path.splitext(out)[0] + '.captions.srt')
    T['ocr'] = time.time() - t
    subs = parse_srt(dsrt)
    if not a.srt and a.min_lines_per_min > 0:
        need = max(3, int(a.min_lines_per_min * info['duration'] / 60))
        if len(subs) < need:
            log('ocr', f'NO_HARDSUBS: only {len(subs)} subtitle lines found (need >= {need}); '
                       'this video does not seem to have burned-in English subtitles')
            sys.exit(3)
    if stop('ocr'): return
    if a.srt and a.corrections:
        from .textfix import load_corrections, apply_corrections
        rules = load_corrections(a.corrections); subs = [[s, e, apply_corrections(x, rules, 'dialogue'), c] for s, e, x, c in subs]
    a.song_spans = song_spans(subs, info['duration']) if a.keep_song_vocals else []
    if a.song_spans:
        log('voices', 'song vocals kept in ' + ', '.join(f'{x:.1f}-{y:.1f}s' for x, y in a.song_spans))
    vm = voice_map(a)
    lines, skipped, classes = [], {}, []
    for s, e, text, cls in subs:
        v = vm.get(cls, vm.get('other', 'bm_george'))
        if not v or v == 'skip': skipped[cls] = skipped.get(cls, 0) + 1; continue
        if s >= info['duration'] - 0.3: continue
        lines.append([s, e, text, v, lang_for(v)]); classes.append(cls)
    used = {}
    for l in lines: used[l[3]] = used.get(l[3], 0) + 1
    log('voices', f'map {vm}; voicing {len(lines)} lines {used}; skipped {skipped}')
    from .ocr import write_srt
    write_srt([[l[0], min(l[1], lines[i + 1][0]) if i + 1 < len(lines) else l[1], l[2]] for i, l in enumerate(lines)],
              os.path.splitext(out)[0] + '.voiced.srt', with_color=False)
    # 2. separation
    t = time.time()
    from .separate import run_separation
    chunks = run_separation(a, video, info, wd); T['separate'] = time.time() - t
    if stop('separate'): return
    # 3. TTS (optionally cloned voices first; Kokoro for everything else)
    external = None
    if a.clone:
        t = time.time()
        if a.clone_python and os.path.exists(a.clone_python):
            from .clone import run_clone
            external = run_clone(a, lines, classes, chunks, info['duration'], wd)
        else:
            log('clone', f'Chatterbox venv not found ({a.clone_python}); using Kokoro for every line')
        T['clone'] = time.time() - t
    t = time.time()
    from .tts import run_tts
    place = run_tts(a, lines, info['duration'], wd, external) if external else run_tts(a, lines, info['duration'], wd)
    T['tts'] = time.time() - t
    if stop('tts'): return
    # 4. mix (streaming)
    t = time.time()
    from .mix import run_mix, encode_mix, mix_cached, mux
    if mix_cached(a, chunks, place, wd):
        log('mix', 'cached')
    else:
        mix_chunks, gain = run_mix(a, chunks, place, wd)
        encode_mix(a, chunks, place, mix_chunks, gain, wd)
    T['mix'] = time.time() - t
    if stop('mix'): return
    t = time.time()
    mux(a, video, info, wd, out); T['mux'] = time.time() - t
    log('done', f'{out}\n{"":17}stage times: ' + ', '.join(f'{k} {v / 60:.1f} min' for k, v in T.items()))
