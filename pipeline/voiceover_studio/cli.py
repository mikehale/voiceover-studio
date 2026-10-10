"""Public CLI dispatch; importing help or diagnostics never loads inference libraries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import runtime


def parser():
    ap = argparse.ArgumentParser(prog='voiceover-studio', description='Process videos and measure Voiceover Studio performance.')
    ap.add_argument('--version', action='version', version='Voiceover Studio ' + runtime.version())
    ap.add_argument('--data-dir', type=Path, default=runtime.data_dir(), help='app data directory (before the command)')
    commands = ap.add_subparsers(dest='command', required=True)
    commands.add_parser('process', add_help=False, help='process a local video; process --help lists all pipeline options')
    commands.add_parser('benchmark', add_help=False, help='run, clone, or compare repeatable benchmarks')
    doctor = commands.add_parser('doctor', help='check app setup and optional voice installations')
    doctor.add_argument('--json', action='store_true', help='machine-readable setup status')
    commands.add_parser('install', add_help=False, help='add the bundled command to PATH (use the native executable)')
    return ap


def pipeline_command(arguments, data, root=runtime.CODE_ROOT):
    defaults = {'--models': str(data / 'models'), '--clone-python': str(data / 'clone-venv/bin/python')}
    # User settings are appropriate for ordinary processing. Benchmarks use fixed repo defaults.
    for option, name in (('--voices', 'voices.json'), ('--corrections', 'corrections.json')):
        if (data / name).is_file():
            defaults[option] = str(data / name)
    supplied = {arg.split('=', 1)[0] for arg in arguments if arg.startswith('--')}
    extra = [part for flag, value in defaults.items() if flag not in supplied for part in (flag, value)]
    return [str(data / 'venv/bin/python'), '-u', '-m', 'lotgh_vo', *extra, *arguments]


def main(argv=None):
    ap = parser()
    args, remaining = ap.parse_known_args(argv)
    data = args.data_dir.expanduser().resolve()
    if args.command == 'doctor':
        if remaining:
            ap.error('unrecognized arguments: ' + ' '.join(remaining))
        report = runtime.diagnostics(data=data)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print('Voiceover Studio ' + report['version'])
            print('Data: ' + report['data'])
            for name, passed in report['checks'].items():
                print(f"{'OK' if passed else 'MISSING'} {name}")
            print('Cloned voices: ' + ('installed' if report['clone_installed'] else 'not installed (optional)'))
            if not report['ready']:
                print('Open Voiceover Studio to complete setup.')
        return 0 if report['ready'] else 1
    if args.command == 'install':
        ap.error('use the app’s Contents/MacOS/voiceover-studio executable to install the command')
    if args.command == 'benchmark':
        from . import benchmark
        # The native launcher and Python entrypoint select the same private data directory.
        return benchmark.main(remaining, default_data=data)
    if not remaining:
        ap.error('process requires a video; use process --help')
    cmd = pipeline_command(remaining, data)
    if not Path(cmd[0]).is_file():
        raise RuntimeError('Private Python is missing. Open Voiceover Studio to complete setup.')
    if '--clone' in remaining and not (data / 'clone-venv/bin/python').is_file():
        raise RuntimeError('Cloned voices are not installed. Install them in Settings > Voices.')
    os.execve(cmd[0], cmd, runtime.environment(data=data))


def entrypoint():
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'voiceover-studio: {exc}', file=sys.stderr)
        raise SystemExit(1)
