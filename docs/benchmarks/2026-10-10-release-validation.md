# Release validation — October 10, 2026

The committed improvements passed a fresh-cache, end-to-end cloned-dialogue comparison, using the same source at 3:30–3:50. Keyframe-aligned clipping produced 22.288 seconds. Both runs used two OCR workers, the German v3 accent, the same runtime/models, and conservative memory guards.

| Measurement | Baseline ca86cb6 | Optimized 1ca11e1 |
| --- | ---: | ---: |
| Total time | 66.29 s | 61.86 s |
| Peak sampled process-group RSS | 4.88 GiB | 4.81 GiB |
| Minimum system free memory | 62% | 63% |
| Observed swap growth | 0 | 0 |
| Cloned lines | 5 | 5 |
| Kokoro fallback lines | 0 | 0 |

This single short comparison was 6.7% faster. It is a release smoke test, not a whole-episode speed guarantee; the small memory difference is not evidence of a substantial memory reduction. Demucs references are regenerated independently and can vary between runs. Existing system swap was allocated before testing; neither run increased it.

Subtitles matched exactly. Both videos decoded successfully, and all five generated WAV files in each run were finite and non-silent. Six-step speech quality was approved in the earlier listening comparison; these checks do not replace listening assessment.

The initial 2:30–2:40 attempt stopped at the existing minimum of three subtitle lines because it contained two. It was excluded from the comparison. No memory guard was relaxed.

21 automated tests passed before committing. The benchmark harness detects supported options in older revisions and preserves memory-guard reasons if source files change during a run.

Raw results are in `release-baseline-20s/`, `release-optimized-20s/`, and `release-validation.json` alongside the output copy of this report.
