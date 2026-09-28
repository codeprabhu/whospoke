# Why Order B Beats the Proposal's Order A

## (a) ELI5 — The Simple Explanation

The proposal says: "First, separate all the voices in the recording. Then, figure out who is who."

We tried it. It's worse. Here's why:

**The problem with separating first (Order A):**

Imagine you have a 60-second recording. Two people talk over each other for only about 5 seconds of it. The other 55 seconds are perfectly fine — one person talking, or silence, or just background noise.

If you run the separation model on the *whole* recording, it has to decide what to do with those 55 seconds of single-speaker audio. But it was *trained* on short clips where *two people are always talking*. When it hears just one person, it gets confused. It splits that person's voice across both tracks, adds weird robotic artifacts, and distorts the background noise. Now, when you try to take "voice fingerprints" of those messy tracks, they're less reliable — the diarization errors go up.

**The solution — diarize first, then separate only the overlaps (Order B):**

1. First, figure out where people are talking and where they overlap (using the overlap detector).
2. Run the separation model *only* on those 5 seconds of overlap.
3. Keep the original audio for everything else.

The separator now gets exactly the kind of audio it was trained on (short, fully overlapping two-speaker clips), so it works much better. And the diarizer gets the clean, undistorted original audio, so it makes fewer mistakes.

**The numbers:**

| | Order B (ours) | Order A (proposal) |
|---|---|---|
| Who said what (cpWER) | **49.5%** | 55.8% |
| Who spoke when (DER) | **20.3%** | 26.2% |
| Separation quality (SI-SDR) | **+9.6 dB** (overlaps only) | +4.5 dB (whole recording) |
| Speed | **16× real-time** | 9× real-time |

Order B makes fewer errors in **50 out of 72** test conversations.

---

## (b) Full Technical Explanation

### The Two Orders

```
ORDER A (proposal):
  mixture ──► Stage 1: separate whole file ──► Stage 2: diarize both tracks jointly ──► Stage 3: ASR

ORDER B (ours):
  mixture ──► Stage 2: diarize + detect overlaps ──► Stage 1: separate only overlaps ──► Stage 3: ASR
```

Both orders use exactly the same models, hyperparameters, and test data. The *only* difference is when and where separation is applied.

### Why Order A Fails — Three Independent Lines of Evidence

#### 1. Separation quality degrades on full recordings

Conv-TasNet was trained on short (~4 s), fully overlapping 2-speaker clips from Libri2Mix. A real conversation recording contains:
- Long stretches of single-speaker speech.
- Silence.
- Background noise with no speech at all.

The model has never seen these conditions. When forced to "separate" a single voice, it splits energy arbitrarily between its two outputs, creating cross-talk artifacts.

| Where Conv-TasNet is applied | Clean | Village 10 dB | Market 5 dB | All |
|---|---|---|---|---|
| Only overlapping stretches (Order B) | +10.6 dB | +8.3 dB | +9.8 dB | **+9.6 dB** |
| Whole recording (Order A) | +9.7 dB | +1.7 dB | +2.0 dB | **+4.5 dB** |

In clean audio, both are similar. But in noise (the proposal's setting), the gap is enormous: **+5.6 dB difference** (95% CI +3.8 to +7.6) on the same 23 two-speaker conversations. Order B wins in 22 of 23.

The separator was trained on *noise-free* or *controlled-noise* 2-speaker mixtures. Over a whole noisy recording, it must decide what to do with minutes of background noise — and it does so badly.

#### 2. Diarization accuracy drops when working on separated tracks

After whole-recording separation, each track carries:
- Separator artifacts (robotic distortion).
- Residual "leakage" of the other speaker's voice.

When the diarizer extracts voice fingerprints (WeSpeaker embeddings) from these tracks, the embeddings are less reliable. The result:

| System | DER | Missed Speech | Wrong Speaker |
|---|---|---|---|
| Order B — spectral clustering | **20.3%** | **6.7%** | 11.9% |
| Order A — spectral clustering | 26.2% | 10.3% | 12.8% |

Order A misses **3.6 percentage points more speech** (10.3% vs 6.7%). This is because separator artifacts cause the VAD/embedding step to fail on audio that would have been easy to process in the original mixture.

#### 3. The transcript quality (cpWER) is consistently worse

On 72 test conversations:

| System | cpWER (mean) | cpWER (heavy overlap) |
|---|---|---|
| Order B — spectral | **49.5%** | **53.6%** |
| Order A — spectral | 55.8% | 59.8% |

Per conversation, Order A makes on average **6.7 more cpWER points** of error (95% CI +4.0 to +9.4). It is worse in **50 of the 72** test conversations.

### Why Order B Works Better — The Technical Argument

1. **Domain match:** The overlap regions of a real conversation are short (0.5–3 s), fully overlapping, two-speaker segments — exactly what Conv-TasNet was trained on. Overlap-only separation is an in-distribution task; whole-recording separation is out-of-distribution.

2. **The diarizer works on clean audio:** In Order B, the diarizer processes the original mixture — noisy, but not distorted by separator artifacts. The WeSpeaker embeddings are therefore more reliable. This is the single biggest cause of Order B's lower DER.

3. **Selective splicing preserves clean speech:** In Order B's splice mode, the separated voice is substituted *only inside the overlapping stretch* (with 0.5 s of context and 20 ms cross-fades). Non-overlapped speech — which the separator would only damage — is never touched. In Order A, *all* speech goes through the separator.

4. **Speed:** Order B separates a few seconds of overlap. Order A separates the entire recording. The separation stage runs **4× faster** in Order B (RTF 0.005 vs 0.021), and the diarization stage runs **~2× faster** (one stream instead of two).

### The Separation Policy (D26)

Even within Order B, *how* the separated audio is used matters:

| Audio given to ASR | cpWER (true timeline) | cpWER (heavy overlap, true timeline) |
|---|---|---|
| Mixture (no separation) | 29.7% | 35.5% |
| Whole turn replaced by separated voice | 31.2% | — |
| **Splice: only the overlapped stretch replaced** | **26.1%** | **27.6%** |

Replacing the whole turn was *worse* than no separation — the separator's artifacts spread over non-overlapped speech within the turn. Splicing in only the overlapped stretch (the default) is the clear winner.

### Speed and Memory

| | Separation | Diarization | ASR | **Total** | Peak GPU |
|---|---|---|---|---|---|
| **Order B** | 0.005 RTF | 0.009 RTF | 0.048 RTF | **0.062** (16× real-time) | 4.5 GB |
| Order A | 0.021 RTF | 0.017 RTF | 0.076 RTF | 0.113 (9× real-time) | 4.4 GB |

Order B is **1.8× faster** overall. Cost scales linearly: 1 min of audio takes ~4 s; 8 min takes ~35 s.

### What We Told the Professor

The proposal's order (Order A) is fully implemented and fully reported alongside Order B. Every number exists for both orders on identical test data. The team drafted an email to the professor (before the review) explicitly flagging this design decision and offering to revert if preferred. The key line:

> *"We will present both and recommend the second, unless you would prefer we keep the original order as the main pipeline."*

### Summary Table

| Dimension | Order A (proposal) | Order B (ours) | Winner |
|---|---|---|---|
| cpWER (who said what) | 55.8% | **49.5%** | Order B by 6.3 pp |
| DER (who spoke when) | 26.2% | **20.3%** | Order B by 5.9 pp |
| SI-SDR improvement | +4.5 dB | **+9.6 dB** | Order B by 5.1 dB |
| Speed (RTF) | 0.113 | **0.062** | Order B 1.8× faster |
| Wins (of 72 conversations) | 22 | **50** | Order B |
