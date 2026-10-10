# Small-sample performance options — October 10, 2026

Historical screening results. The later implementation and approval of six CFM steps are recorded in [implemented improvements](2026-10-10-implemented-improvements.md).

These trials prioritize tool settings, model lifetime and bounded concurrency. Each candidate ran alone with fresh caches. No endurance tests were used to screen the options.

## Findings

- **Keep Demucs model reuse as an opt-in option.** Across the repeated 90-second, three-chunk tests, reused-model runs took about 6.53 seconds; separate-launch runs took 8.12–9.19 seconds. This removes repeated model loading without changing separation settings. A fixed-seed ten-second comparison produced exactly identical vocals and background samples. The implementation releases the model at the end of the stage, including failure paths.
- **OCR/separation overlap is promising but remains a prototype.** Two runs took about 18.4 seconds versus 22.1 seconds sequentially, with matching subtitle text. Sampled process memory rose moderately. Production integration still needs cancellation/error handling; the prototype is not enabled in the app.
- **Four OCR workers are not worthwhile on this short sample.** 17.91 versus 18.36 seconds, with 3.03 versus 2.20 GiB peak sampled RSS. Both extracted the same 13 subtitle lines. The sample only forms two or three work chunks, so this says nothing about optimal worker count for a long episode.
- **Reject the tested 720p OCR setting.** It took 25.54 seconds versus 18.36 and merged two subtitles, missing “The stars are good.” The trial scaled to 720p and used the bottom 190 pixels; other crop/detection tuning is a separate experiment.
- **Keep 5 fps as the OCR default.** Four fps took 16.84 seconds and kept 13 lines, but changed recognized text on two lines, including losing a letter in “left.” This small speed gain does not establish a safe default.

- **Leave the vocoder at 10 steps.** Six CFM steps took 45.34 versus 46.89 seconds overall (about 3%); synchronized synthesis took 22.92 versus 24.61 seconds (about 7%). All four lines completed with equal audio durations and no retries/fallbacks, but sampled memory did not improve. Audible quality has not been accepted; listen to the saved paired samples before considering this setting.
- **Do not adopt zero guidance from this trial.** It was aborted at 29.31 seconds when system free memory fell below the conservative 60% floor (lowest sample 56%), before a line completed. No swap growth was observed. This is an incomplete measurement, not evidence of a speed gain; the guard was not relaxed or retried.

## Measurements

Wall time includes process startup and cleanup. RSS is sampled process-group resident memory, not total GPU/driver memory. It can double-count shared pages. System swap is machine-wide; pre-existing allocated swap was not cleared. Every completed screening run below recorded zero observed swap growth.

| Trial | Wall seconds | Peak sampled RSS GiB | Observed swap growth GiB | Status |
|---|---:|---:|---:|---|
| small-baseline-jobs2 | 18.36 | 2.20 | 0.000 | completed |
| small-baseline-jobs4 | 17.91 | 3.03 | 0.000 | completed |
| small-ocr720-jobs2 | 25.54 | 2.03 | 0.000 | completed |
| small-ocr4fps-jobs2 | 16.84 | 2.07 | 0.000 | completed |
| separation-baseline | 9.19 | 1.10 | 0.000 | completed |
| separation-persistent-demucs | 6.56 | 1.28 | 0.000 | completed |
| separation-baseline-repeat1 | 8.12 | 1.06 | 0.000 | completed |
| separation-reuse-repeat1 | 6.53 | 1.28 | 0.000 | completed |
| separation-baseline-repeat2 | 8.70 | 1.05 | 0.000 | completed |
| separation-reuse-repeat2 | 6.53 | 1.30 | 0.000 | completed |
| pipeline-baseline-small | 22.16 | 2.19 | 0.000 | completed |
| pipeline-overlap-small | 18.45 | 2.54 | 0.000 | completed |
| pipeline-overlap-repeat | 18.38 | 2.41 | 0.000 | completed |
| pipeline-baseline-repeat | 22.14 | 2.20 | 0.000 | completed |
| stem-equivalence | 4.39 | 0.94 | 0.000 | completed |
| clone-option-baseline-4 | 46.89 | 4.73 | 0.000 | completed |
| clone-option-cfm6-4 | 45.34 | 4.89 | 0.000 | completed |
| clone-option-cfg0-4 | 29.31 | 4.62 | 0.000 | aborted |

## Workloads and limits

- Machine: Apple M3 Max, 36 GiB unified memory.
- OCR/overlap: the same source video, requested range 03:30–04:30. Stream-copy clipping produced 62.288 seconds. CPU OCR, VideoToolbox decode, two workers except the explicit four-worker trial. Overlap uses MPS separation concurrently with CPU OCR. All caches were fresh.
- Separation: 90 seconds of audio from 02:00, three 30-second chunks with five-second padding. Original and reusable paths use htdemucs, MPS, one random shift, 0.25 internal overlap and the same PCM16 intermediate WAVs / PCM24 final FLACs. Smaller-than-normal chunks expose startup overhead; do not extrapolate the percentage to a whole episode.
- Speech-option screening: four actual lines (two narration, two dialogue), German V3, fixed reference clips, seeds and cold generated-audio caches. Model precision remains unchanged. These tests screen settings, not voice quality acceptance.
- Sampling: 0.5 seconds plus telemetry overhead; abort on any observed swap growth, RSS over 10 GiB, free memory below 60%, or three minutes. Small fixed inputs and headroom are the primary controls. A sampled guard cannot stop an OS allocation before it happens.
- Early clone exploration: a 24-line baseline completed before the user clarified the small-sample requirement; a subsequent attention trial was interrupted. Neither is used for the option comparisons. Precision/attention changes were set aside in favor of tool settings and concurrency.

## Branch changes and reproducibility

The `perf/voiceover-optimizations` worktree adds opt-in `--sep-reuse-model` to `process` and `benchmark run`, a benchmark abort on observed swap growth, documentation, and lifecycle/guard tests. The app's default separation path is unchanged. No app rebuild, merge, or push was performed for these experiments.

Raw per-run manifests, timings, logs and memory samples are preserved in the local `outputs/performance` folders. Each run also has a `memory.csv` for inspecting growth over time. Candidate patches against the original baseline are saved alongside the local report. The machine's normal applications stayed open; short timings are evidence for further decisions, not whole-episode speed guarantees.

Speech setting baseline: 4 usable lines; 24.61 synthesis seconds; 15.88 audio seconds including retries; 0 length retries; 0 failed lines; 0 accent fallbacks.

Speech setting cfm6: 4 usable lines; 22.92 synthesis seconds; 15.88 audio seconds including retries; 0 length retries; 0 failed lines; 0 accent fallbacks.

Speech setting cfg0: 0 usable lines; 0.00 synthesis seconds; 0.00 audio seconds including retries; 0 length retries; 0 failed lines; 0 accent fallbacks.
