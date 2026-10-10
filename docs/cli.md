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

For multi-chunk videos, Demucs stays loaded throughout separation and is released before speech synthesis. This is enabled by default; it preserves the existing model, overlap, shift count and output encoding. A three-chunk short test showed less startup overhead, but does not predict whole-episode savings.

```sh
voiceover-studio process /path/video.mkv --sep-reuse-model -o /path/english.mp4
```

Select another private data directory **before** the command:

```sh
voiceover-studio --data-dir /path/isolated-data doctor --json
```

The directory must already be set up; the CLI does not download dependencies automatically. `VOICEOVER_STUDIO_DATA_DIR` provides the same default for scripted use.

## Benchmark

`process --metrics-json` records pipeline and stage elapsed times. Use `benchmark` when you also need sampled memory, guard limits and comparable run reports. Ordinary processing can reuse its work directory; performance comparisons need fresh caches.

```sh
voiceover-studio benchmark run /path/video.mkv --start 210 --duration 60 \
  --run-dir /path/results/pipeline-baseline
voiceover-studio benchmark clone --job /path/clone/job.json --lines 4 \
  --run-dir /path/results/clone-baseline
voiceover-studio benchmark compare /path/results/baseline /path/results/candidate
```

The benchmark uses fresh private caches and records timings, source/build identity, memory samples, retries and fallbacks. It refuses another active voiceover job unless `--wait-idle` is used. Its memory/time guards stop only its own process group; by default, any observed growth in system swap aborts the run. Use small fixed samples with memory headroom. See [the benchmark guide](benchmarking.md) (also bundled as `Resources/docs/benchmarking.md`).

Choose a new `--run-dir` for every run; existing directories are refused. `summary.json` contains total wall time and sampled memory peaks, `pipeline.json` contains stage times, and `memory.jsonl` contains the individual samples. The default sampling interval is half a second. Default benchmarks use a one-minute video sample or four cloned lines, a 10 GiB process-memory limit, and a 60% free-memory floor. RSS covers the process group and can double-count shared pages; it does not include every GPU allocation. System swap is machine-wide. Compare matching workloads with other heavy jobs idle.

### Exact ten-second dialogue check (02:30–02:40)

The pipeline's `--start` and `--end` options cut by stream copy, so the clip can include extra footage around a keyframe. For an exact sample, first cut it with the app's bundled FFmpeg. Replace the input path and use a new sample directory:

```sh
APP="/Applications/Voiceover Studio.app"
SAMPLE="$HOME/Movies/voiceover-check-0230"
mkdir "$SAMPLE"
"$APP/Contents/Resources/bin/ffmpeg" \
  -ss 150 -i /path/video.mkv -t 10 -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -preset fast -crf 18 -threads 2 \
  -c:a aac -b:a 192k "$SAMPLE/source.mp4"

"$APP/Contents/MacOS/voiceover-studio" process "$SAMPLE/source.mp4" \
  --device cpu --jobs 2 --tts-jobs 1 --min-lines-per-min 0 \
  --video-codec copy --workdir "$SAMPLE/cache" \
  --metrics-json "$SAMPLE/pipeline.json" -o "$SAMPLE/output.mp4"
```

This reads the actual subtitles and uses standard voices from your saved voice map. `--min-lines-per-min 0` disables the minimum-subtitle guard for this tiny sample, which may contain fewer than the usual minimum of three lines. Check the recognized subtitles and listen to the output. Leave the guard enabled for normal videos. Clip preparation is separate from the recorded pipeline time.

The current `benchmark run` command does not expose `--device`, `--tts-jobs` or `--min-lines-per-min`. Use the example above for the tiny functional check, and a longer dialogue sample for the public benchmark command. The processing example writes timings only; it does not collect memory samples.

On October 10, 2026, this exact source range on an Apple M3 Max (36 GiB) voiced two cyan narration lines with Kokoro `bm_lewis`. A separate run using the bundled benchmark supervisor recorded **15.922 seconds total**, **15.495 seconds in the pipeline**, and **1.849 GiB peak sampled process-group RSS**, with no observed swap growth. Another episode was processing at the time. These results are a functional baseline for CPU separation and standard voices; they do not measure clone performance or long-run memory stability. The repository keeps the detailed report and sanitized samples under `docs/benchmarks/2026-10-10-dialogue-0230.*`.

## Performance defaults

CPU OCR and MPS separation overlap when at least 60% system free memory is reported at startup. CPU-only separation, supplied SRT files, OCR-only runs, more than four requested OCR workers and unavailable memory telemetry use sequential preparation. Separation completes and its process exits before speech synthesis starts. Cancellation or OCR failure stops its worker and active tool. Stage timings may overlap; use total wall time for end-to-end comparisons.

OCR uses up to four workers, never more than the number of pending chunks. Resolution and the 5 fps sampling rate are unchanged. `--jobs` still overrides the worker count. Use `--no-overlap-preparation` for sequential stages and `--no-sep-reuse-model` for separate Demucs launches.

Cloned voices use six CFM vocoder steps, following the listening comparison. Guidance and model precision are unchanged. Generated-audio cache keys include the step count; random seeds retain their previous identity. Old generated clips are regenerated, while OCR, stems and speaker references remain reusable.

## Common issues

- **Command not found:** use the full app executable path, or add `~/.local/bin` to your shell's PATH after installation.
- **Missing environment or models:** open the app, finish setup, then run `doctor` again.
- **Too few subtitles on a tiny sample:** use the short-sample command above and verify the OCR output before interpreting timings.
- **Benchmark detects an active job:** let that job finish, or use `--wait-idle`; the benchmark will not stop it.
- **Unexpectedly fast rerun:** use a fresh processing work directory or a new benchmark run directory to avoid measuring cached work.

## Development

From a checkout, use `PYTHONPATH=pipeline python3 -m voiceover_studio --help`. For actual processing, the command selects the app's private environment. `tools/benchmark.py` remains a compatibility wrapper; implementation lives in the bundled `voiceover_studio` package. Use `benchmark ... --repo /path/candidate` to benchmark a different checkout; the harness remains the one you invoked. Bundled reports use build metadata and source hashes without invoking Git.

`build_app.sh` compiles and signs the CLI, copies its package and guides into the app, embeds the build commit, and checks help/version before creating the DMG and ZIP. Build artifacts remain in this checkout's `dist`; building does not replace the installed app.

If a synced folder keeps adding Finder metadata during signing, set `VOICEOVER_DIST_DIR` to a directory outside it (for example a temporary directory) when running `build_app.sh`. Signature verification failures stop the build.
