# Benchmark CLI

Run `python3 tools/benchmark.py --help` from a checkout. The wrapper uses the installed app's Python environments, models and bundled FFmpeg by default. It runs the checkout's pipeline code. It does not install packages or change the installed app.

## Full workflow

```sh
python3 tools/benchmark.py run /absolute/path/video.mkv \
  --start 120 --duration 900 --run-dir /absolute/path/results/pipeline-baseline
```

Add `--clone --accent german_v3` to include cloned voices. Use `--until ocr`, `separate`, `clone`, `tts` or `mix` to measure a partial pipeline. Video is copied by default. Start/end are seconds; the pipeline cuts on a keyframe, so the actual clip duration is recorded. Default: 15 minutes, four OCR workers. Check the actual voiced-line count; a 15-minute video is not necessarily a sufficient clone endurance test.

## Long cloning sample

Reuse a saved `work/<item>/clone/job.json` to avoid repeating OCR and separation. The wrapper reads stem files but generates audio, references and caches in a new private directory.

```sh
python3 tools/benchmark.py clone \
  --job '/path/to/saved/clone/job.json' --lines 320 --offset 0 \
  --run-dir /absolute/path/results/clone-baseline
```

The sample is 320 chronological lines by default. Speaker grouping runs on that selected sample, so its speaker references may differ from the full episode. Keep the same job, offset and line count for comparisons. Generation retains the worker's normal speaker ordering, seed, accent, retry and fallback behavior. Models are loaded cold once per worker; generation cache is cold at the beginning. Supported worker recycling re-executes a clean process and preserves this run's caches; counts and timings are aggregated across workers. A 320-line run measures sustained behavior, but will not exercise a 500-line scheduled recycle unless a memory threshold triggers one earlier. Use at least 600 lines to test that scheduled boundary.

Use `--repo /path/to/another/checkout` to benchmark a candidate revision with the same harness. The selected repo must support `--metrics-json` for full pipeline runs; focused clone runs work with the previous worker and the pending recycling implementation.

## Reports and comparisons

Every run requires a **new** `--run-dir`. An existing path is refused rather than silently measuring cached output. The source video, installed app and original job cache are never deleted.

Outputs:

- `manifest.json`: command, workload, source commit, dirty status, Python source hashes, runtime package versions and guard limits.
- `summary.json`: completion/abort status, elapsed time, sampled memory peaks, clone attempts, retries, fallbacks and stage timings when available.
- `pipeline.json`: precise completed-stage timings from full/partial pipeline runs.
- `memory.jsonl`: elapsed time, system swap/free-memory percentage and RSS of all processes in the benchmark's process group.
- `events.jsonl`: clone generation durations, audio durations and MPS allocation samples, including restarts.
- `run.log`: complete diagnostic output.
- `cache/`: preserved intermediate files and audio for listening comparisons.

```sh
python3 tools/benchmark.py compare /path/results/baseline /path/results/candidate
```

Comparison refuses different workloads, failed/aborted runs and initial generated-audio cache hits. Source changes are expected, but workloads must match. Source identity uses full SHA-256 hashes of Python files; media/stems are identified by absolute path, size and modification time, not a full media hash. Runtime package versions are recorded; review changes in runtime or model files before interpreting a comparison. The hardware should be idle except for normal background activity.

Cloning `synth_rtf_including_retries` is synchronized synthesis time divided by all audio produced by successful synthesis attempts, including discarded retries. It excludes model loading, speaker grouping, reference preparation, fitting and restart overhead. Compare **wall time** and quality as well as RTF. Failed attempts still cost wall time. `generated_lines` counts final usable cloned lines; `line_events` also includes failures. Final fallback flags can be lost across worker recycling by the underlying engine, so review the full log as well.

MPS live/driver peaks are **samples at event boundaries**, not guaranteed transient peaks. RSS excludes some GPU/driver allocations and can double-count shared pages across processes. Do not add RSS and MPS driver memory together. Swap and free-memory percentage are system-wide; unrelated applications affect them. Full pipeline mode records process memory but does not inject the focused worker's per-line MPS instrumentation. GPU synchronization adds measurement overhead; use the same harness for baseline and candidate.

## Memory guards and job contention

macOS is required for memory telemetry. Default guards: system swap above 8 GiB, process-group RSS above 24 GiB, free memory below 15%, or elapsed time above 120 minutes. These stop only the benchmark's own process group and preserve diagnostics. Sampling happens every three seconds; guards cannot prevent an allocation spike between samples. Adjust using `--max-swap-gib`, `--max-rss-gib`, `--min-free-percent`, `--timeout-minutes` and `--interval`.

The wrapper refuses to start when it detects another heavy voiceover job. `--wait-idle` waits for it to finish, with a separate wait bounded by `--timeout-minutes`. It never stops another job. Missing memory telemetry stops the benchmark instead of running unmonitored.

## Acceptance protocol

1. Run one 15-minute pipeline baseline and a separate 320+ line clone baseline.
2. Change one optimization, retaining identical sample settings and generation seeds.
3. Repeat the focused benchmark, then compare elapsed time, sampled memory, retry and fallback counts.
4. Blind-listen to roughly 20 matching lines, including long dialogue and narration. Faster truncation or worse voices are not wins.
5. Before accepting a memory fix, run 600+ lines through a worker-recycle boundary, followed by a full episode when practical. Repeat promising comparisons to distinguish gains from run-to-run variation.

Tests: `python3 -m unittest discover -s tests -v`. The process cleanup test needs permission to inspect/terminate its own subprocesses.
