#!/usr/bin/env python3
"""Build and check Voiceover Studio voice plugins (see docs/voice-plugins.md).

  tools/make_voice_plugin.py build --wav ref.wav --id example_narrator --name "Example Narrator" \\
      --accent original --version 1.0.0 --author "Your Name" --out example_narrator.vsvoice.zip
      ref.wav: 10-20 s of one speaker, PCM WAV (mono 24 kHz recommended), no music
  tools/make_voice_plugin.py verify PLUGIN.zip
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))
from lotgh_vo import voiceplugin as vp

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    for a in ('wav', 'id', 'name', 'version', 'author', 'out'): b.add_argument('--' + a, required=True)
    b.add_argument('--accent', default='original', choices=vp.ACCENTS)
    v = sub.add_parser('verify'); v.add_argument('zip')
    a = ap.parse_args()
    if a.cmd == 'build':
        m = vp.build(a.out, a.wav, a.id, a.name, a.version, a.author, a.accent)
        print(f"built {a.out}: {m['name']} ({m['id']} {m['version']}, accent {m['accent']})")
    else:
        m, info, _ = vp.verify_zip(a.zip)
        print(f"OK: {m['name']} ({m['id']} {m['version']}, accent {m['accent']}, {info['duration']} s reference)")

if __name__ == '__main__':
    try: main()
    except vp.PluginError as e: sys.exit(f'error: {e}')
