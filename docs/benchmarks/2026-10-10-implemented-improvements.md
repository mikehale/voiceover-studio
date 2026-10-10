# Implemented performance improvements

October 10, 2026 — `perf/voiceover-optimizations`

Enabled in both the app pipeline and CLI:

- Reuse Demucs across chunks, then release its model before speech synthesis.
- Overlap CPU OCR and MPS separation when startup free memory is at least 60% and no more than four OCR workers are requested. Otherwise run sequentially. Supplied subtitles and OCR-only runs avoid the background worker.
- Own and reap the separation worker, including its running tool, on cancellation or OCR failure. Keep it in the app/benchmark process group for cancellation and memory accounting.
- Track app progress independently for overlapping stages.
- Use six CFM steps for cloned speech. Generated-audio cache identities change, while random seeds retain their earlier values. OCR, separation and speaker caches remain reusable.
- Bound default OCR concurrency to four workers and avoid spawning workers without pending chunks. Keep the original resolution, 5 fps sampling, guidance and model precision.
- Default benchmarks to one minute of video or four cloned lines, half-second memory samples, a 10 GiB RSS ceiling, a 60% free-memory floor, and abort on any observed swap growth.

Use `--no-sep-reuse-model` or `--no-overlap-preparation` to disable either optimization for comparison.

## Validation

21 automated tests passed, including real worker/tool cancellation, cleanup on failure, unchanged speech seeds with invalidated audio caches, and GUI progress during overlap.

The implemented full pipeline processed the requested one-minute dialogue range (62.288 seconds after keyframe-aligned clipping), voiced 13 lines, and wrote output video in **27.64 seconds**, with **2.50 GiB peak sampled process-group RSS**, **76% minimum system free memory**, and **zero observed swap growth**. The run used two OCR workers and four Kokoro workers. Stage times overlap and must not be added to estimate wall time.

The generated line from the implemented six-step clone path contained 108,480 samples identical to the approved CFM6 sample (maximum sample difference 0). The four-line verification stopped at the conservative free-memory guard after one line, without swap growth. An unrelated source edit also invalidated that run's timing; it is not used as a performance measurement.

The earlier controlled small-sample comparisons established approximately 25% less separation time from model reuse and 17% less preparation time from overlap. These are stage/workload-specific measurements, not a claim of equivalent whole-episode improvement.

The subsequent end-to-end release comparison passed; see [release validation](2026-10-10-release-validation.md).
