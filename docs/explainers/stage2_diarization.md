# Stage 2 — Speaker Diarization (Voice Fingerprints, Spectral Clustering & GMM)

## (a) ELI5 — The Simple Explanation

Diarization answers: **"Who spoke when?"**

Imagine you're listening to a phone call between two people. You can hear both voices on the same audio track, and you don't know who they are. Your job is to colour-code the timeline: every second that Person A speaks gets coloured blue, and every second Person B speaks gets coloured red.

Here's how the system does it:

1. **Find where speech is.** A neural network (pyannote) listens to the audio and marks every moment someone is talking vs silence.
2. **Take voice fingerprints.** The audio is sliced into 1.5-second windows. Each window is passed through another neural network (WeSpeaker) that converts it into a 256-number "fingerprint" — like a voiceprint. Windows from the same person will have similar fingerprints.
3. **Group the fingerprints.** This is the core AI step. We have a pile of fingerprints, and we need to sort them into groups (clusters). We wrote two algorithms from scratch:
   - **Spectral Clustering:** Build a graph where each fingerprint is a dot, and similar fingerprints are connected with strong lines. Split the graph by cutting the weakest lines. The math (eigenvalues of the graph Laplacian) automatically tells you how many speakers there are.
   - **Gaussian Mixture Model (GMM):** Assume the fingerprints form blob-shaped clouds in space. Fit 1 cloud, then 2 clouds, then 3, etc. The number of clouds with the best fit-vs-complexity trade-off (BIC score) is the number of speakers.
4. **Build the timeline.** Each 50 ms of audio gets the speaker label from the fingerprint windows that cover it. Short flickers (< 0.3 s) are smoothed away.
5. **Handle overlaps.** Where the overlap detector says two people talk at once, we add the nearest *other* speaker as a second label.

**How good is it?** Our from-scratch spectral clustering (20.3% DER) matches the industry-standard pyannote 3.1 pipeline (18.4% DER) within the margin of error.

---

## (b) Full Technical Explanation

### Step 1: Voice Activity Detection (VAD) and Overlap Detection

**Model:** pyannote `segmentation-3.0` — a neural network that outputs, every ~17 ms, the probability of each "powerset" class: silence, exactly 1 speaker, exactly 2 speakers, etc.

**Outputs derived:**
- **Speech regions:** frames where P(≥1 speaker) is above threshold → where anyone is talking.
- **Overlap regions:** frames where P(≥2 speakers) is above threshold → where two people talk at once.

Both thresholds use `min_duration_on = 0.0` and `min_duration_off = 0.0` (no duration-based filtering; all filtering happens later in the timeline construction).

**Code reference:** [`vad.py`](../src/whospoke/vad.py)

### Step 2: Speaker Embeddings

**Model:** WeSpeaker ResNet-34 (`pyannote/wespeaker-voxceleb-resnet34-LM`). Trained on VoxCeleb so that audio from the same person produces vectors pointing in similar directions (high cosine similarity), while different speakers are far apart.

**Windowing:** Speech regions are covered with **1.5 s windows, hopped every 0.75 s**. Windows shorter than 0.4 s are skipped. Each window is zero-padded to a common length, passed through WeSpeaker in batches of 32, and produces a **256-dimensional embedding**.

**Code reference:** [`diarization.py:embed()`](../src/whospoke/diarization.py), lines 90–108; [`diarization.py:sliding_windows()`](../src/whospoke/diarization.py), lines 111–126

### Step 3: Unsupervised Clustering (Written from Scratch)

Both algorithms take the same input (N embedding vectors) and produce the same output (a speaker label per vector), without knowing how many speakers there are.

#### Spectral Clustering (Ng, Jordan & Weiss 2001; Park et al. 2019)

1. **Affinity matrix:** Compute all pairwise cosine similarities between L2-normalised embeddings. This gives an N×N matrix $A$ where $A_{ij}$ measures how similar windows $i$ and $j$ sound.
2. **Row-wise pruning:** For each window, keep only the top $p$% most similar neighbours (default $p = 20\%$); set all other entries to zero. This removes noisy, weak connections. The matrix is then symmetrised: $\hat{A} = \frac{1}{2}(A + A^T)$, and negative values are clipped to zero.
3. **Graph Laplacian:** Compute the unnormalised Laplacian $L = D - \hat{A}$ where $D$ is the diagonal degree matrix.
4. **Eigengap for speaker count:** Compute the eigenvalues of $L$ in ascending order. The number of speakers $k$ is the position of the **largest gap** between consecutive eigenvalues (starting from `min_speakers`). If all eigenvalues are very close (i.e. all windows have high mutual similarity > `single_speaker_sim = 0.55`), it concludes there is only one speaker.
5. **k-Means on eigenvectors:** Take the first $k$ eigenvectors, L2-normalise each row, and run k-means (10 initialisations) to produce the final cluster labels.

**Code reference:** [`clustering.py:SpectralClustering`](../src/whospoke/clustering.py), lines 28–78

#### Gaussian Mixture Model (GMM)

1. **Normalise & reduce:** L2-normalise the embeddings, then project to a lower dimension with PCA (default 8 dimensions).
2. **Model selection:** Fit diagonal-covariance GMMs with $k = 1, 2, \ldots, K_{\max}$ components (each with 3 initialisations). For each $k$, compute the **Bayesian Information Criterion (BIC):** $\text{BIC} = -2 \ln \hat{L} + p \ln n$, where $\hat{L}$ is the maximised likelihood, $p$ is the number of parameters, and $n$ is the number of data points. BIC penalises complexity, so the $k$ with the lowest BIC balances fit and parsimony.
3. **Assign labels:** Predict cluster membership with the best GMM.

**Code reference:** [`clustering.py:GMMClustering`](../src/whospoke/clustering.py), lines 81–107

#### Cluster Clean-Up (Both Methods)

After clustering, two post-processing steps (D17):

1. **Absorb tiny clusters:** Any cluster holding < 3% of the total windows is not a real speaker — it is typically a noisy or heavily overlapped patch. Each of its windows is reassigned to the nearest large cluster (by cosine similarity to cluster centroids).
2. **Merge near-identical clusters:** If two clusters' mean embeddings have cosine similarity above `merge_sim` (default 0.6), they are the same person split in two. Merged iteratively, most-similar pair first.
3. **Relabel by first appearance:** Clusters are renamed 0, 1, 2, ... in the order they first appear, so "Speaker_A" is always the first voice heard.

**Code reference:** [`clustering.py:consolidate()`](../src/whospoke/clustering.py), lines 110–138

### Step 4: Timeline Construction

1. **Frame-level labelling:** Each 50 ms frame of speech is assigned the speaker label of the embedding windows covering it, using a **centre-weighted triangular vote** (windows contribute more at their centre than their edges).
2. **Smoothing:** Speaker runs shorter than `min_turn` (0.3 s) are absorbed into the longer neighbouring run (up to 3 passes).
3. **Turn extraction:** Consecutive frames with the same label are merged into Turn objects `(start, end, speaker)`.

**Code reference:** [`diarization.py:_frame_labels()`](../src/whospoke/diarization.py), lines 131–149; [`diarization.py:_smooth()`](../src/whospoke/diarization.py), lines 152–169

### Step 5: Overlap-Aware Assignment

Where the overlap detector fires, the second speaker is determined by finding the nearest *other* speaker in time (Bullock et al., 2020). Without this step, every overlap is counted as missed speech for the second speaker.

**Code reference:** [`diarization.py:add_overlap_speakers()`](../src/whospoke/diarization.py), lines 185–196

### Order A Variant: Joint Clustering of Separated Tracks

In Order A, the separator produces two tracks *before* diarization. The diarizer then:
1. Extracts embeddings from both tracks independently.
2. **Clusters all embeddings together** (so a speaker keeps one name even if the separator moves them between tracks).
3. If the same speaker is found on both tracks at the same time, or one track is much quieter (> 6 dB) than the other while both are active, the weaker copy is treated as **separator leakage** and dropped.

**Code reference:** [`diarization.py:Diarizer.assign_streams()`](../src/whospoke/diarization.py), lines 281–318

### Results

| System | DER | Missed | False Alarm | Wrong Speaker | Speaker-Count Error |
|---|---|---|---|---|---|
| **Our spectral clustering (Order B)** | **20.3%** (17.3–23.6) | 6.7% | 1.6% | 11.9% | 0.78 |
| Our GMM (Order B) | 20.3% (17.2–23.4) | 4.7% | 2.8% | 12.9% | 0.74 |
| pyannote 3.1 (reference) | 18.4% (15.5–21.4) | 3.9% | 2.5% | 12.0% | 0.46 |
| Order A — spectral | 26.2% (23.4–29.1) | 10.3% | 3.0% | 12.8% | 0.82 |
| True timeline (floor) | 8.7% | 0% | 8.7% | 0% | 0 |

**Key findings:**
- **Spectral vs GMM:** A tie on DER. Spectral is 1.2 cpWER points better but within noise. Spectral is the default because it needed less clean-up on dev (16.3% vs 18.4% DER; without clean-up GMM dev DER was 27.7%).
- **Two speakers are hardest:** DER 24.5% for 2 speakers vs 16.0% for 3 speakers. The eigengap finds only one speaker in 36% of two-speaker conversations and too many speakers 31% of the time. Fixing this is the most valuable next step.

### Key Files

| File | Role |
|---|---|
| [`src/whospoke/vad.py`](../src/whospoke/vad.py) | VAD + overlap detection (pyannote segmentation-3.0) |
| [`src/whospoke/diarization.py`](../src/whospoke/diarization.py) | Embeddings, timeline construction, Order A joint clustering |
| [`src/whospoke/clustering.py`](../src/whospoke/clustering.py) | Spectral clustering + GMM, both from scratch, plus clean-up |
| [`scripts/tune_diarization.py`](../scripts/tune_diarization.py) | Grid search over clustering settings (dev only) |
| [`results/tuned_params.json`](../results/tuned_params.json) | Frozen settings from the dev grid search |

### Key Decisions

- **D2:** Both clustering methods implemented and compared (the proposal says "or").
- **D17:** Cluster clean-up (absorb tiny clusters + merge near-identical ones) is essential — GMM's dev DER drops from 27.7% to 18.4% with it.
- **D22:** All thresholds (p-percentile, merge_sim, min_cluster_frac) frozen on dev set only; test set run once.
