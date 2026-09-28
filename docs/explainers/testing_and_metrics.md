# How We Test It — Simulated Conversations, Metrics & Evaluation

## (a) ELI5 — The Simple Explanation

**The fundamental problem:** To measure how well a system works, you need the right answers. But a real radio broadcast or phone call doesn't come with a perfect transcript — nobody has written down exactly who spoke every single word and when. Without a "right answer," you can't grade the system.

**Our solution:** We *build* conversations ourselves, with perfect answers.

Here's the recipe:
1. Take a real Hindi phone conversation from IndicVoices (real speech, real accents, real code-switching).
2. Separate the two sides of the call (Speaker A's audio and Speaker B's audio are already separate files).
3. Layer them on top of each other with controlled overlap (sometimes they talk at once, sometimes they don't).
4. Add real background noise (recorded in a village or a market).
5. Mix it all together into one noisy audio file.

Because *we* assembled the mixture, we know exactly:
- Where every word starts and ends (ground truth for transcription).
- Who says each word (ground truth for diarization).
- What each person's clean voice sounds like (ground truth for separation).

Now we can score every stage precisely.

**The metrics (what we measure):**

| Metric | Question it Answers | Analogy |
|---|---|---|
| **SI-SDR** (dB, higher = better) | How much cleaner is each voice after separation? | "+10 dB" means the interfering voice is 10× quieter |
| **DER** (%, lower = better) | How much speaking time is attributed to the wrong person, missed, or hallucinated? | Like a test score: 20% error means 80% of the timeline is correct |
| **WER** (%, lower = better) | What fraction of words are wrong? | A spelling test: 21% error means roughly 4 out of 5 words are right |
| **cpWER** (%, lower = better) | Who said what? — words wrong OR given to the wrong speaker | The ultimate test: you only get credit if both the word and the speaker are correct |

---

## (b) Full Technical Explanation

### Test Data: Simulated Conversations

**Why simulate?** Real broadcast audio has no ground-truth labels, so it cannot be scored. All prior work in speaker diarization and speech separation (LibriCSS, AMI-style simulation, Libri2Mix) uses the same approach: build mixtures from real speech with exact labels.

**Source material:**
- **Speech:** IndicVoices (AI4Bharat) — spontaneous conversational Hindi recorded over phone calls, with transcripts and speaker IDs. Each file is one side of a real conversation.
- **Noise:** DEMAND (CC-BY-4.0) for sustained environmental ambience + ESC-50 (CC-BY-NC) for sound events:
  - *Village soundscape:* DEMAND field/park/home + ESC-50 cows, roosters, dogs, birds, insects.
  - *Market soundscape:* DEMAND traffic/square/station/bus + ESC-50 horns, engines, sirens, trains.

**Construction (`synth.py`):**

1. **Turn planning:** Speakers alternate turns. Each turn consists of 1–3 consecutive utterances from one speaker (in their original temporal order). Turn changes are either:
   - **Gaps** (uniform random 0.15–0.8 s of silence), or
   - **Overlaps** (with probability `overlap_prob`; overlap length clamped to avoid three concurrent speakers).
2. **Rendering:** Each speaker's utterances are placed into a clean per-speaker track. Utterances are level-normalised (to a common speech RMS) with ±3 dB random per-speaker jitter.
3. **Noise mixing:** The clean speech sum is computed, then the noise is scaled to the target SNR using `scale_to_snr()`.
4. **Anti-clipping:** A single global gain keeps the mixture at ≤0.9 of full scale.
5. **Exact ground truth:** The code saves:
   - `mixture.wav` — the noisy, overlapped audio (what the pipeline hears).
   - `sources/*.wav` — each speaker's clean, isolated track (for SI-SDR scoring).
   - `reference.rttm` — exact speech activity derived from the clean tracks (not utterance boundaries — actual energy-based VAD on the clean audio, so intra-utterance pauses are excluded).
   - `reference.json` — complete metadata: segments with text, speaker IDs, overlap statistics, the `SynthConfig` that produced it.

**Code reference:** [`synth.py:simulate()`](../src/whospoke/synth.py), lines 195–269; [`synth.py:speech_activity()`](../src/whospoke/synth.py), lines 115–155

**Reference activity (`speech_activity`):** Utterance boundaries from IndicVoices include leading/trailing silence and internal pauses. The reference is therefore computed from the **clean source track**: within each utterance's time span, 20 ms frames louder than 40 dB below the utterance's peak count as speech. Pauses < 0.25 s are bridged; islands < 0.1 s are dropped. This is model-free, so no system is favoured (D14).

### Test Set Design (D12)

**Paired conditions:** Each group of 2–3 real speakers is rendered under **all 9 conditions:**

| Dimension | Levels |
|---|---|
| Overlap | none (0%), low (~3–5%), high (~11–18%) |
| Noise | clean, village (10 dB SNR), market (5 dB SNR) |

This "paired" design means that any difference between conditions is caused by the condition — not by different speakers being drawn. High-overlap ≈12% matches lively real meetings (AMI-style corpora report ~10–15%).

**Sizes:**
- **Test set:** 8 groups × 9 conditions = **72 conversations** (~83 minutes total).
- **Dev set:** 4 groups × 9 conditions = **36 conversations** (~42 minutes total).

**Speaker disjointness (D13):** Dev speakers never appear in the test set. IndicVoices' valid split shares 91 speakers with the first train shard, so the test set uses *valid* speakers and the dev set uses *train* speakers that never appear in valid. All tuning happens on dev only; the test set is run once.

### Metrics

#### Stage 1: SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

$$\text{SI-SDR}(e, s) = 10 \log_{10} \frac{\|\alpha s\|^2}{\|e - \alpha s\|^2}$$

where $\alpha = \frac{\langle e, s \rangle}{\|s\|^2}$ is the optimal scaling factor, $s$ is the reference (clean voice), and $e$ is the estimate (separated voice). Both are zero-mean.

**SI-SDR improvement** = SI-SDR(separated, clean) − SI-SDR(mixture, clean). It measures how many dB of separation the model achieves.

**Permutation-invariant:** With 2 outputs and 2 references, both assignments are tried and the better one is kept (pit_si_sdr).

**Code reference:** [`metrics.py:si_sdr()`](../src/whospoke/metrics.py), lines 22–29; [`metrics.py:pit_si_sdr()`](../src/whospoke/metrics.py), lines 32–39

#### Stage 2: DER (Diarization Error Rate)

$$\text{DER} = \frac{\text{missed} + \text{false alarm} + \text{confusion}}{\text{total reference speech}}$$

where:
- **Missed:** reference speech not covered by any hypothesis.
- **False alarm:** hypothesis speech where the reference has silence.
- **Confusion (speaker error):** hypothesis and reference both have speech, but attributed to different speakers.

**Parameters:** 0.25 s forgiveness collar at turn boundaries; overlapped speech is **included** (not skipped — skipping it would hide exactly the errors we care about).

**Speaker mapping:** The optimal one-to-one mapping between hypothesis and reference speaker labels is found by pyannote.metrics, which minimises confusion.

**JER (Jaccard Error Rate):** Same idea but averaged per speaker (so a speaker with little speech is weighted equally). Also computed but DER is the primary metric.

**Code reference:** [`metrics.py:der()`](../src/whospoke/metrics.py), lines 57–70

#### Stage 3: WER / CER

$$\text{WER} = \frac{\text{Levenshtein edit distance (word-level)}}{\text{number of reference words}}$$

Inputs are normalised: Unicode NFC, punctuation removed, lowercased, single-spaced.

**Code reference:** [`metrics.py:error_counts()`](../src/whospoke/metrics.py), lines 104–111

#### End-to-End: cpWER (Concatenated Polyglot Word Error Rate)

The "who said what" metric. It is the only score that requires both the correct words AND the correct speaker attribution.

**Algorithm:**
1. For each speaker (reference and hypothesis), concatenate all their words in time order into one string.
2. Build a cost matrix: for every possible pairing of a reference speaker to a hypothesis speaker, the cost is the Levenshtein edit distance between their concatenated texts.
3. Solve the optimal assignment using the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`).
4. Sum the costs under the optimal assignment; divide by the total number of reference words.

**Why cpWER is harsh:** A word given to the wrong speaker is counted as an error *twice* — it is missing from the correct speaker (a deletion) and extra for the wrong speaker (an insertion). So 10% speaker confusion can easily add 20+ WER points.

**Code reference:** [`metrics.py:cp_error()`](../src/whospoke/metrics.py), lines 133–151

### Confidence Intervals

All reported intervals are **95% bootstrap confidence intervals** over conversations. Comparisons between systems are **paired** (same conversations), which removes the large between-conversation variance and isolates the effect of the system change.

### Error Cascade Analysis

`scripts/evaluate.py` runs **oracle versions** of the pipeline that replace one stage with the ground truth:

| System | Diarization | Audio Given to ASR | What It Isolates |
|---|---|---|---|
| oracle-clean | true | each speaker's clean voice | the ASR's own error |
| oracle-mix | true | noisy overlapped mixture | + damage from noise and overlap |
| oracle-sep | true | mixture with separated overlaps | how much separation repairs |
| B-spectral | ours | mixture with separated overlaps | + diarization errors (full system) |

The differences between consecutive rows quantify the error contributed by each stage:

| Step | cpWER | What It Adds |
|---|---|---|
| ASR on clean voices, true timeline | 17.7% | ASR's intrinsic error |
| + noise and overlap | 29.7% | +12.0 pp from acoustics |
| + separation (true timeline) | 26.1% | −3.6 pp (separation helps) |
| + our diarization (full system) | **49.5%** | +23.4 pp from diarization |

### Key Files

| File | Role |
|---|---|
| [`src/whospoke/synth.py`](../src/whospoke/synth.py) | Conversation simulator |
| [`src/whospoke/noise.py`](../src/whospoke/noise.py) | Village/market soundscape generator |
| [`src/whospoke/metrics.py`](../src/whospoke/metrics.py) | All metrics: SI-SDR, DER, JER, WER, CER, cpWER |
| [`scripts/build_dataset.py`](../scripts/build_dataset.py) | Builds dev + test conversations |
| [`scripts/evaluate.py`](../scripts/evaluate.py) | End-to-end evaluation (all systems × all conversations) |
| [`scripts/make_report.py`](../scripts/make_report.py) | Generates tables and figures from CSVs |
| [`results/eval_test_indicconformer.csv`](../results/eval_test_indicconformer.csv) | Every system × every test conversation |
