# Voiceover Studio command line

Every app build includes `Contents/MacOS/voiceover-studio`. It uses that app's bundled code and FFmpeg plus the same private Python environments and models as the GUI. No checkout, Git, Homebrew or system Python is needed. The command does not open a GUI or start the queue server.

## Start here

```sh
"/Applications/Voiceover Studio.app/Contents/MacOS/voiceover-studio" --help
"/Applications/Voiceover Studio.app/Contents/MacOS/voiceover-studio" --version
"/Applications/Voiceover Studio.app/Contents/MacOS/voiceover-studio" doctor
```

Open the app once to complete setup. `doctor --json` reports machine-readable setup status and exits nonzero if required files are missing. It checks installation files, not model inference quality. Install optional cloned voices and German models through Settings > Voices.

Choose **Help > Install Command-Line Tool…** in the app, or add the command from Terminal:

```sh
"/Applications/Voiceover Studio.app/Contents/MacOS/voiceover-studio" install
```

This creates `~/.local/bin/voiceover-studio`. Add `~/.local/bin` to PATH if necessary. `install --bin-dir DIR` chooses another directory. Installation never replaces a different command or edits shell profiles. Help, version and command installation work before Python setup. The symlink points at this app: install from the app's permanent location, and recreate the link if you move it. Running the executable directly works from any app location, including paths containing spaces.

## Process a video

```sh
voiceover-studio process /path/video.mkv -o /path/english.mp4
voiceover-studio process /path/video.mkv --start 00:02:00 --end 00:17:00 \
  --clone --clone-accent german_v3 --metrics-json /path/timings.json
voiceover-studio process --help
```

Processing accepts all existing pipeline options, including ranges, subtitle input, voices, mix levels, work directory and partial stages (`--until`). Local video input is required; URL downloads remain available through the GUI. By default the CLI copies the source video stream and keeps a resumable work directory beside the output. It reads the app's saved voice map and corrections when available; remaining settings use pipeline defaults. Explicit flags override defaults. `--metrics-json` writes completed-stage timings and status, including partial runs and Python failures; a forced termination may prevent the final timing write.

Select another private data directory **before** the command:

```sh
voiceover-studio --data-dir /path/isolated-data doctor --json
```

The directory must already be set up; the CLI does not download dependencies automatically. `VOICEOVER_STUDIO_DATA_DIR` provides the same default for scripted use.

## Benchmark

```sh
voiceover-studio benchmark run /path/video.mkv --start 120 --duration 900 \
  --run-dir /path/results/pipeline-baseline
voiceover-studio benchmark clone --job /path/clone/job.json --lines 320 \
  --run-dir /path/results/clone-baseline
voiceover-studio benchmark compare /path/results/baseline /path/results/candidate
```

The benchmark uses fresh private caches and records timings, source/build identity, memory samples, retries and fallbacks. It refuses another active voiceover job unless `--wait-idle` is used. Its memory/time guards stop only its own process group. See [the benchmark guide](benchmarking.md) (also bundled as `Resources/docs/benchmarking.md`).

## Development

From a checkout, use `PYTHONPATH=pipeline python3 -m voiceover_studio --help`. For actual processing, the command selects the app's private environment. `tools/benchmark.py` remains a compatibility wrapper; implementation lives in the bundled `voiceover_studio` package. Use `benchmark ... --repo /path/candidate` to benchmark a different checkout; the harness remains the one you invoked. Bundled reports use build metadata and source hashes without invoking Git.

`build_app.sh` compiles and signs the CLI, copies its package and guides into the app, embeds the build commit, and checks help/version before creating the DMG and ZIP. Build artifacts remain in this checkout's `dist`; building does not replace the installed app.

If a synced folder keeps adding Finder metadata during signing, set `VOICEOVER_DIST_DIR` to a directory outside it (for example a temporary directory) when running `build_app.sh`. Signature verification failures stop the build.
