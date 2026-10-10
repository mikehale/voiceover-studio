#!/usr/bin/env python3
"""Compatibility entrypoint; the supported CLI is `voiceover-studio benchmark`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pipeline'))
from voiceover_studio.cli import entrypoint

if __name__ == '__main__':
    sys.argv.insert(1, 'benchmark')
    entrypoint()
