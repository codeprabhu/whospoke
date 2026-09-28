# Stage 1 — Speech Separation (Conv-TasNet)

## (a) ELI5 — The Simple Explanation

Imagine two people are talking at the same time into one microphone. What you get is a single audio track where both voices are mixed together — like two songs playing simultaneously from one speaker. Stage 1's job is to "unmix" them: take that single messy recording and produce two separate, clean recordings — one for each voice.

We use a neural network called **Conv-TasNet** to do this. Think of it as a smart audio filter:

1. It listens to the raw waveform (not a spectrogram).
2. It learns to predict two "masks" — like two stencils — one shaped like each person's voice.
3. It applies each stencil to pull out each voice individually.

**Two key problems we had to solve:**

- **The volume problem.** The model outputs audio at random volumes, sometimes even upside-down (inverted). We mathematically fix the volume so the two outputs add back up to the original input.
- **The swapping problem.** The model only works on 8-second chunks. Between chunks, it randomly decides which voice goes on Track 1 and which goes on Track 2. Without fixing this, the speakers flip back and forth. We fix it by overlapping chunks by 2 seconds and using cosine similarity to match the voices across consecutive chunks.

**The punchline:** In the final pipeline (Order B), we *don't* run this on the whole recording. We only run it on the exact moments where two people overlap. Running it on everything else (silence, single speakers, noise) actually makes the audio *worse*.

---

## (b) Full Technical Explanation

### Model Architecture

Conv-TasNet (Luo & Mesgarani, 2019) operates entirely in the time domain — it processes raw waveform samples, not spectrograms.

| Component | Role |
|---|---|
| **1-D Convolutional Encoder** | Maps the input waveform into a learned high-dimensional feature representation (a "basis"), analogous to a learnable STFT |
| **Temporal Convolutional Network (TCN)** | A stack of dilated 1-D convolutions with increasing receptive field. Predicts one multiplicative mask per source |
| **Decoder** | A transposed convolution that inverts the encoder, converting each masked representation back into a time-domain waveform |

**Checkpoint:** `JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k` from the Asteroid library. Trained on Libri2Mix (2-speaker English mixtures with noise), 16 kHz. This is the `sepnoisy` variant — chosen over `sepclean` because it retains its separation quality in noisy conditions (see Results below). The model has only **5.1 M parameters**.

**Alternative tested:** SepFormer (`speechbrain/sepformer-whamr16k`), a Transformer-based separator trained on reverberant noisy mixtures (WHAMR!). It scored far worse on our Hindi data (+2.2 dB on test vs +9.6 dB for Conv-TasNet), likely because its training regime (reverberant) does not match our test conditions (anechoic phone recordings with additive noise). Demucs was ruled out entirely — its public checkpoints separate musical instruments, not voices (D8).

### Engineering Details

#### Gain Fitting (Mixture Consistency)

Separators trained with **Scale-Invariant SDR (SI-SDR)** as their loss function produce outputs with arbitrary gain and sign (the loss is invariant to both). Naively using these outputs yields inconsistent volume and occasional phase inversions.

**Solution:** For each window, after obtaining the raw separator outputs $y_0, y_1$, we solve for per-source gains $g_0, g_1$ such that:

$$g_0 \cdot y_0 + g_1 \cdot y_1 \approx x \quad (\text{the input mixture})$$

This is a 2×2 least-squares problem. The Gram matrix $Y Y^T$ is regularised with a small ridge term ($10^{-3} \cdot \text{trace}$) to prevent a near-silent output from being amplified to infinity.

**Code reference:** [`separation.py:_separate_window()`](../src/whospoke/separation.py), lines 72–84.

#### Windowed Processing & Speaker-Consistent Stitching

Recordings are processed in **8-second windows** with a **2-second overlap** (hop = 6 s). The separator's two outputs can swap order between windows (the model has no concept of "Speaker 1 stays on Track 1"). Without stitching, long outputs scored **−3 to −7 dB** (speakers randomly swapped mid-recording).

**The stitching algorithm:**

1. For each window after the first, compute the cosine similarity between the previous window's outputs (in the 2 s overlap region) and both possible orderings of the current window.
2. If the reversed ordering has higher similarity, swap the current window's outputs.
3. Crossfade using a linear ramp over the 2-second overlap.
4. A weighted accumulation buffer ensures smooth blending across all windows.

**Code reference:** [`separation.py:separate()`](../src/whospoke/separation.py), lines 86–110.

### Results

**Metric:** SI-SDR improvement (dB) — how much cleaner each separated voice is compared to the raw mixture. Higher is better.

| Where Conv-TasNet is applied | Clean | Village (10 dB SNR) | Market (5 dB SNR) | All |
|---|---|---|---|---|
| **Only overlapping stretches** (Order B) | +10.6 dB | +8.3 dB | +9.8 dB | **+9.6 dB** |
| Whole recording (Order A) | +9.7 dB | +1.7 dB | +2.0 dB | **+4.5 dB** |

**Checkpoint comparison (on dev, D18):**

| Checkpoint | Dev overlaps | Test overlaps |
|---|---|---|
| Conv-TasNet `sepnoisy` | +6.4 dB | **+9.6 dB** |
| Conv-TasNet `sepclean` | +5.4 dB | +7.3 dB |
| SepFormer WHAMR | — | +2.2 dB |

**Impact on transcripts (with the true timeline, D26):**

| Audio given to ASR | cpWER (all) | cpWER (heavy overlap) |
|---|---|---|
| Noisy mixture | 29.7 % | 35.5 % |
| **Mixture + separated overlaps spliced in** | **26.1 %** | **27.6 %** |

### Key Files

| File | Role |
|---|---|
| [`src/whospoke/separation.py`](../src/whospoke/separation.py) | The Separator class: model loading, gain fitting, windowed stitching |
| [`src/whospoke/pipeline.py`](../src/whospoke/pipeline.py) | How separated audio is fed to the ASR (splice mode vs turn mode) |
| [`scripts/eval_separation.py`](../scripts/eval_separation.py) | Measures SI-SDR improvement on dev/test |
| [`scripts/tune_separation_policy.py`](../scripts/tune_separation_policy.py) | Chooses how to feed separated audio to ASR (dev only) |
| [`results/separation_*.csv`](../results/) | Raw SI-SDR numbers per conversation |

### Key Decisions

- **D8:** Demucs ruled out (music-separation checkpoints, not speech).
- **D15:** The Asteroid checkpoint metadata says 8 kHz, but it is actually a 16 kHz model (verified by SI-SDR: 12.6–14.0 dB at 16 kHz vs 5.0 dB at 8 kHz).
- **D16:** Gain fitting + stitching are essential — without them, long audio scores −3 to −7 dB.
- **D18:** `sepnoisy` checkpoint chosen — slightly worse on clean audio but far more robust in noise.
- **D26:** Splicing separated audio into only the overlapped stretch (with 0.5 s context) is better than replacing the whole turn — separator artefacts spread over non-overlapped speech hurt the transcripts.
