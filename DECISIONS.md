# Decision Log

Every judgement call made while building the project, recorded **before** acting on it.
Format: what had to be decided → what we chose → why (in plain English).

---

## D1 · Order of Stage 1 (separation) and Stage 2 (diarization)
**Decided:** 2026-09-25
**Choice:** Build **both** orders and let the numbers pick the winner.
- **Order A (proposal):** separate the whole recording first → diarize the separated tracks → transcribe.
- **Order B (alternative):** diarize the original recording first → detect overlap regions → separate only those regions → transcribe.

**Why:** The proposal specifies Order A, but pretrained separators (Conv-TasNet) are trained on clean 2-speaker
lab mixtures and can add artefacts to long, noisy, real recordings. Rather than silently changing the professor's design,
we run both on identical test audio and justify the final choice with measured DER / WER / runtime.

## D2 · Clustering method for diarization
**Decided:** 2026-09-25
**Choice:** Implement **both** Spectral Clustering and GMM (the proposal says "Spectral Clustering *or* GMMs"),
compare them, and also run pyannote's own pipeline as an industry reference point.
**Why:** Costs little extra, and gives a "we evaluated the options" slide instead of an arbitrary pick.

## D3 · Evaluate on synthetic conversations built from real Hindi speech
**Decided:** 2026-09-25
**Choice:** Build test "conversations" by mixing real IndicVoices Hindi utterances from different speakers
(controlled overlap %, added real background noise — originally planned from Vaani's noise dataset, see D11 for the change).
**Why:** Real radio clips have no ground-truth labels, so they cannot be scored. With mixtures we built ourselves, we
know exactly who spoke what and when, so every stage can be measured. Real (unlabelled) audio is still used for a
qualitative demo.

## D4 · Language scope
**Decided:** 2026-09-25 (by the team)
**Choice:** Hindi + English code-switching (Hinglish) only, for midsem.

## D5 · Compute
**Decided:** 2026-09-25
**Choice:** Develop and test on the team laptop (RTX 3050, 6 GB); pretrained models only, no fine-tuning of large
models. Provide a Google Colab notebook so teammates can run it in a browser.
**Why:** Lets every run be verified locally; 6 GB is enough for inference but not for training big models.

## D6 · Git
**Decided:** 2026-09-25
**Choice:** Commit locally as work progresses; **do not push** to GitHub automatically. Large data and model files are
git-ignored.
**Why:** Safe and reversible; the team pushes when they are ready.

## D7 · Datasets from the proposal
**Decided:** 2026-09-25
| Corpus | Used? | Reason |
|---|---|---|
| IndicVoices (AI4Bharat) | ✅ Main source | Spontaneous Hindi speech with transcripts + speaker info → builds the test conversations |
| Project Vaani (IISc/ARTPARK) | ✅ | Real-world noisy Hindi (transcribed part) for ASR testing, plus its English-word tags for the Hinglish lexicon. Not used for background noise: too little speech-free audio (D11) |
| Nirantar (AI4Bharat) | ⚠️ ASR cross-check only | Distributed only as one ~198 GB `.tgz` (5 × 39.6 GB parts) with all 22 languages shuffled in one flat folder, so the Hindi files (135.6 h, ~8.5 GB) cannot be pulled out without streaming all of it. Its manifests (checked 2026-09-28) show the Hindi part comes from the same collection as IndicVoices: identical fields, and 165 of the 514 speakers in IndicVoices' Hindi *valid* split (our test speakers, D13) also appear in it. So it would add little new data and risks test-speaker leakage. A teammate streamed it on Colab and kept only Hindi (64,597 clips, 135.6 h, 490 speakers). 923 of those clips are the exact IndicVoices valid files. On an 800-clip sample with our test speakers removed (406 clips), IndicConformer scores 11.0 % WER and IndicWav2Vec 30.2 %, so the model choice (D9) is unchanged. It is reported as a cross-check in RESULTS.md, not used to build test conversations. The full 8.5 GB Hindi set stays on the teammate's Drive and isn't downloaded. We train no model (D5), so more speech from the same collection has nothing to feed into, and 406 clips already separate the two ASR models by 19 WER points. Building a second test set from the ~325 non-test speakers is logged as a possible extension (RESULTS.md, "what we will do next") |
| AIR-RS-DB | ❌ | No public source found under this name (the proposal's citation for it is blank) |

## D8 · Proposal named "Conv-TasNet or Demucs"
**Decided:** 2026-09-25
**Choice:** Conv-TasNet.
**Why:** The widely available Demucs checkpoints separate *music* (vocals vs. instruments), not one speaker from
another. Conv-TasNet is trained exactly for speaker separation.

## D9 · Proposal named "IndicASR or IndicWav2Vec"
**Decided:** 2026-09-25
**Choice:** Run **both** (IndicConformer = AI4Bharat's current "IndicASR" model; IndicWav2Vec-Hindi), keep the better one.

## D10 · Output script
**Decided:** 2026-09-25
**Choice:** Output both the native Devanagari transcript (used for scoring) and a romanised Hinglish version.
**Why:** The proposal allows "native scripts **or** standardized Latin representations"; giving both covers either reading.

## D11 · Background-noise source: DEMAND + ESC-50 instead of Vaani
**Decided:** 2026-09-25
**Tried first:** harvesting speech-free audio from Project Vaani (both the main corpus and the Noise-Event release).
**Finding:** Vaani clips are trimmed tightly around the speaker — ~1 s of speech-free audio per ~500 s — and noise
that happens *during* speech cannot be separated from the voice.
**Choice:** two soundscapes that match the proposal's "ambient village or market noise":
- *village* = DEMAND field/park/home ambience + ESC-50 cows, roosters, hens, dogs, crows, insects, birds
- *market*  = DEMAND traffic/square/station/bus ambience + ESC-50 horns, engines, sirens, trains, bells, footsteps
DEMAND is CC-BY-4.0; ESC-50 is CC-BY-NC-3.0 (fine for coursework). Noise is not one of the proposal's speech corpora,
so this does not change which speech datasets we use. Vaani is still used for real-world speech (see D7).

## D12 · Test-set design: paired conditions
**Decided:** 2026-09-25
**Choice:** each group of 2–3 real speakers is rendered under all 9 conditions:
overlap {none, low ≈3 %, high ≈12 %} × noise {clean, village @ 10 dB SNR, market @ 5 dB SNR}.
**Why:** differences between conditions are then caused by the condition, not by which speakers were drawn.
High-overlap ≈12 % matches lively real meetings (AMI-style corpora report ~10–15 %).

## D13 · Dev (tuning) speakers are disjoint from test speakers
**Decided:** 2026-09-25
**Finding:** IndicVoices' Hindi *valid* split shares 91 speakers with the first *train* shard.
**Choice:** test set = valid split; dev set = train-shard speakers that never appear in valid. All thresholds are tuned
on dev only; test numbers are reported once.

## D14 · Tight reference segments
**Decided:** 2026-09-25
**Finding:** IndicVoices utterance files include leading/trailing silence; counting it as "speech" inflated every
system's missed-speech error by ~7 % (pyannote included).
**Choice:** trim edge silence (>35 dB below the utterance peak, 50 ms kept) before placing utterances.

## D15 · Separator checkpoint & sample rate
**Decided:** 2026-09-25
**Finding:** the Asteroid Conv-TasNet "16k" checkpoints report `sample_rate=8000` in their metadata, but running them
at 16 kHz gives 12.6–14.0 dB SI-SDR improvement on Hindi 2-speaker mixtures vs 5.0 dB at 8 kHz → they are 16 kHz models.
SpeechBrain SepFormer-WHAMR16k scored only ~4.5 dB on the same mixtures (it was trained on reverberant audio).
**Choice:** Conv-TasNet (proposal's model), `sepnoisy` vs `sepclean` checkpoints compared in the Stage-1 benchmark;
SepFormer kept only as a reference row.

## D16 · Separator outputs: per-source gain fit + stitching
**Decided:** 2026-09-25
**Finding:** separators trained with a scale-invariant loss return each voice with an arbitrary gain *and sign*;
naive window stitching then swapped speakers between windows (−3 to −7 dB on 30 s audio).
**Choice:** fit per-source gains so the outputs add back up to the input (least squares, small ridge term), then align
consecutive windows by cosine similarity on their shared 2 s before crossfading.

## D17 · Cluster clean-up step
**Decided:** 2026-09-25
**Finding:** raw spectral/GMM clustering over-counted speakers on 3-speaker and overlapped audio (4–5 found for 3).
**Choice:** after clustering, absorb clusters holding < 3 % of windows into the nearest real cluster, and merge
clusters whose mean voice fingerprints are near-identical (cosine > threshold). Thresholds tuned on the dev set only.

## D18 · Separator checkpoint chosen on dev: Conv-TasNet `sepnoisy`
**Decided:** 2026-09-25 (from `results/separation_dev_summary.csv`)
| SI-SDR improvement (dB) | clean | village 10 dB | market 5 dB | all |
|---|---|---|---|---|
| overlap regions — Conv-TasNet sepnoisy | 7.05 | 5.98 | 6.26 | **6.43** |
| overlap regions — Conv-TasNet sepclean | 8.46 | 4.90 | 2.79 | 5.42 |
| overlap regions — SepFormer WHAMR | 0.26 | 0.25 | 0.47 | 0.33 |
| whole recording — Conv-TasNet sepnoisy | 3.05 | 3.60 | 0.56 | 2.40 |
**Choice:** `sepnoisy` — slightly worse than `sepclean` on clean audio but far more robust in noise, which is the
proposal's setting. **First evidence for the Order question:** the same model helps ~2.7× more when it is only asked
to separate the overlapping stretches (6.4 dB) than when it must process the whole recording (2.4 dB).

## D19 · Choosing the ASR model on Vaani, not IndicVoices
**Decided:** 2026-09-25
**Finding:** IndicConformer was trained on IndicVoices (other utterances of the same speakers), so on IndicVoices-based
audio it has a home advantage over IndicWav2Vec (trained before IndicVoices existed).
**Choice:** pick the Stage-3 model by WER on Project Vaani's transcribed Hindi *test* split — real phone recordings from
across India that neither model was trained on. IndicVoices numbers are reported alongside, with this caveat.

## D20 · Hinglish lexicon: hand-curated + mined from Vaani train annotations
**Decided:** 2026-09-25
**Finding:** Vaani annotators tag every English word spoken inside Hindi (`बिल्डिंग {building}`) — ~8,600 tagged words in
two test shards alone. The hand-curated lexicon (456 entries from IndicVoices' 2,500 most frequent words) spelled 94 %
of the words it contained correctly but covered only 43 % of Vaani's English tokens; the rules alone got 7 % of the rest.
**Choice:** mine additional pairs from Vaani **train** transcripts only (25 of 193 shards, 66k transcripts; test split
untouched for measurement). Keep a pair only if seen ≥ 2×, ≥ 70 % agreement on spelling, tagged English in ≥ 50 % of its
appearances, and not one of the frequent IndicVoices words judged Hindi (blocks e.g. `बस` "enough" → "bus").
Result: +2,147 entries.
**Also decided (tooling):** the official AI4Bharat transliterator (IndicXlit) needs `fairseq`, which does not install on
Python 3.12/Windows; the rule-based romaniser + lexicon is deterministic, testable and runs anywhere.

## D21 · Large data lives outside OneDrive
**Decided:** 2026-09-25 (requested by the team: OneDrive quota is full)
**Choice:** the 4.7 GB of corpora + simulated conversations moved to `C:\whospoke-data`. All code resolves the data folder
through `whospoke.paths.DATA` (`WHOSPOKE_DATA` env var → one-line `data_location.txt` → default `project/data`), so fresh
clones and Colab keep working with no setting. `index.csv` paths are stored relative to the data folder.

## D22 · Diarization settings frozen on dev (re-tuned against exact references)
**Decided:** 2026-09-25
**Finding:** after the references were rebuilt from exact speech activity (D14), the grid search over clustering
settings (spectral p-percentile ∈ {0.1–0.4}, GMM PCA dims ∈ {4, 8, 16}, merge threshold, small-cluster absorption,
overlap-aware assignment) on the **dev** conversations gave:

| system | dev DER | without clean-up (D17) | speaker-count error |
|---|---|---|---|
| Order B · spectral | **0.163** | 0.173 | 0.83 |
| Order B · GMM | 0.184 | 0.277 | 0.83 |
| Order A · spectral | 0.240 | 0.247 | 1.25 |
| Order A · GMM | 0.262 | 0.308 | 1.28 |

**Choice:** freeze the best setting per system in `results/tuned_params.json`; the test set is then run once with
those settings, with no further changes. Order B + spectral clustering is the proposed system; the other three are
reported as comparisons. The clean-up step helps GMM far more than spectral clustering (spectral's eigengap already
estimates the speaker count reasonably).

## D23 · Python environment lives outside OneDrive
**Decided:** 2026-09-25
**Choice:** the virtual environment moved from `project/.venv` to `C:\whospoke-data\venv` (same reason as D21), and is
registered as the Jupyter kernel `whospoke`. Nothing in the code depends on where the environment lives; README gives
the setup steps for a fresh machine.


## D24 · Stage-3 model: IndicConformer (chosen on Vaani, as planned in D19)
**Decided:** 2026-09-25
**Finding:** 400 random single-speaker utterances per set (`scripts/eval_asr.py`, `results/asr_comparison.csv`):

| model | Vaani test WER | Vaani CER | English words recognised | IndicVoices WER | speed (× real time) | GPU peak |
|---|---|---|---|---|---|---|
| **IndicConformer 600M** (RNNT) | **21.0 %** | **10.0 %** | **78 %** of 775 | 13.9 % | 0.054 (18× faster than real time) | 3.7 GB |
| IndicWav2Vec Hindi (CTC, no LM) | 38.0 % | 16.8 % | 48 % of 775 | 36.1 % | 0.007 (135×) | 5.0 GB |

**Choice:** IndicConformer is the default. It wins on the independent Vaani audio by 17 WER points and on English
words spoken inside Hindi by 30 points, so the IndicVoices "home advantage" (D19) does not change the verdict.
IndicWav2Vec stays available (`--asr indicwav2vec`) as the fast option.
**Nirantar cross-check** (added 2026-09-28, `scripts/eval_asr_nirantar.py`): 406 Nirantar Hindi clips with our test
speakers removed give 11.0 % WER for IndicConformer and 30.2 % for IndicWav2Vec, the same ranking. Nirantar's Hindi is the
IndicVoices collection (D7), so it flatters IndicConformer the same way and does not replace Vaani as the selection set.
**Romaniser:** on all 8,630 English words tagged in the two Vaani *test* shards (never used to build the lexicon),
82.0 % of the Devanagari forms come out in the correct English spelling.

## D25 · Crash fix: SpeechBrain's global TorchScript setting vs the IndicConformer preprocessor
**Decided:** 2026-09-25
**Finding:** the first end-to-end test run crashed inside IndicConformer's feature extractor (a TorchScript module)
with an NVRTC compile error (`c10::complex` not found). Cause, reproduced in isolation: `import speechbrain` (pulled in
by pyannote and SepFormer) globally disables TorchScript's profiling executor; the legacy executor then fuses the
complex-valued STFT ops into a runtime-compiled CUDA kernel that does not compile on Windows. The ASR-only benchmark
never imported SpeechBrain, which is why it ran fine.
**Choice:** run the preprocessor with TorchScript's fusers switched off for that call only (`_no_jit_fusion` in
`asr_backends.py`, equivalent to `torch.jit.fuser("none")` but without its per-call deprecation warnings). This changes
speed only; the features are identical.

## D26 · Order B separates the *detected overlap regions*, splicing in only the overlapped stretch
**Decided:** 2026-09-25
**Finding (disclosed: found on the first test run):** in that run, the tuned Order-B diarizer (`overlap_aware=False`,
chosen on dev for DER) never produced overlapping turns, and the code only separated where two *turns* overlapped.
So Order B separated nothing (0 % of turns), which is not the design in D1 ("detect overlap regions → separate only
those"). Separately, with the true timeline, replacing a *whole* overlapping turn by its separated voice made the
transcripts slightly worse (cpWER 31.2 % vs 29.7 % on the plain mixture): the separator's artefacts spread over
speech that was never overlapped.
**Choice:** implement D1 as written. Overlap detection always runs. Order B separates every region where the detector
(or two turns) says two people speak, with some context. Inside that region only, it swaps in the separated voice
that best matches the turn's speaker, with 20 ms cross-fades; the rest of the turn stays the original audio. The
policy (no separation / whole turn / splice with 0.5 s or 1 s context) is chosen on the **dev** set only
(`scripts/tune_separation_policy.py`, `results/separation_policy_dev.csv`), and the test set is then re-run once with
the chosen policy. The first test run's numbers are kept in `results/eval_test_indicconformer_run1.csv` for
transparency.
**Dev result** (36 conversations, cpWER; `results/separation_policy_dev.csv`):

| audio given to the ASR | true timeline | our diarization (B-spectral) | true timeline, heavy overlap |
|---|---|---|---|
| mixture (no separation) | 35.0 % | 46.4 % | 44.1 % |
| whole turn separated (first design) | 37.1 % | 47.2 % | 43.6 % |
| **splice, 0.5 s context** | **31.4 %** | **46.4 %** | **36.2 %** |
| splice, 1.0 s context | 32.1 % | 46.5 % | 38.1 % |

**Chosen:** splice with 0.5 s context (default `sep_context_s=0.5`). It is the best or tied policy on both timelines.
With the true timeline it removes about a fifth of the heavy-overlap errors (44.1 % → 36.2 %). With our own diarization
the gain is small, because diarization mistakes (words credited to the wrong speaker) then dominate the error.

