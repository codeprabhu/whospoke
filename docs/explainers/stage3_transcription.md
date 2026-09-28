# Stage 3 — Hindi ASR + Hinglish Romanisation

## (a) ELI5 — The Simple Explanation

Stage 3 takes the cleaned-up audio and turns it into text. This is harder than it sounds because:

1. People speak **Hindi** with random **English words** mixed in (code-switching): "sir mere koi galti hui mere haath se wo mera **ID card** gum ho gaya hai sir".
2. The ASR model writes *everything* in Devanagari script — even English words. So "homework" comes out as `होमवर्क` and "meeting" as `मीटिंग`.
3. We need to produce text in *two* scripts: the original Devanagari, and a Roman-alphabet version (Hinglish) where English words are spelled correctly in English and Hindi words are transliterated the way people type them in WhatsApp chats.

**How it works:**

- **The ASR model:** We tested two models the proposal mentioned. IndicConformer (a powerful 600-million-parameter model) beat IndicWav2Vec by a huge margin on real-world audio (21% vs 38% word error rate). IndicConformer became the default.
- **The Hinglish romaniser:** After the ASR writes everything in Devanagari, our custom code converts it:
  - First, it checks a dictionary of ~2,600 English loanwords (e.g., `ऑफिस → office`).
  - If the word is not in the dictionary, it applies Hindi phonetic rules including *schwa deletion* (the silent "a" in Hindi) to produce natural-looking romanisations (e.g., `कमरा → kamra`, not `kamaraa`).

---

## (b) Full Technical Explanation

### ASR Models Compared

Both models were named as options in the project proposal.

#### IndicConformer 600M (AI4Bharat — "IndicASR")

- **Architecture:** A **Conformer** encoder (convolution + self-attention) producing frame-level features, with two decoder heads:
  - **CTC (Connectionist Temporal Classification):** Frame-by-frame character prediction with greedy decoding. Fast, batched.
  - **RNNT (Recurrent Neural Network Transducer):** An autoregressive decoder that conditions each output token on the previously emitted token. Slightly more accurate (~2 WER points), sequential (one utterance at a time), slower.
- **Parameters:** 600 M (multilingual, covering 22 Indian languages).
- **Inference stack:** TorchScript mel preprocessor → ONNX encoder + decoder heads. The ONNX sessions use the CUDA Execution Provider with specific optimisations:
  - `cudnn_conv_algo_search: DEFAULT` (not EXHAUSTIVE — the latter re-benchmarks every new input length, adding ~50 s per new shape).
  - `cudnn_conv1d_pad_to_nc1d: 1` (prevents cuDNN from selecting a very slow depthwise-conv kernel).
- **Critical bug fix (D25):** `import speechbrain` (pulled in by pyannote / SepFormer) globally disables TorchScript's profiling executor. The legacy executor then fuses the STFT's complex-valued ops into a runtime-compiled CUDA kernel that crashes on Windows (`c10::complex` not found). Fixed by disabling the TorchScript fuser specifically during the preprocessor call (`_no_jit_fusion` context manager). Features are identical; only speed is affected.

**Code reference:** [`asr_backends.py:IndicConformerASR`](../src/whospoke/asr_backends.py), lines 141–298

#### IndicWav2Vec-Hindi (AI4Bharat)

- **Architecture:** wav2vec 2.0 — self-supervised pre-training on raw audio from 40 Indian languages, fine-tuned for Hindi with CTC.
- **Decoding:** Greedy CTC (no language model).
- **Loading workaround:** The repo only ships `pytorch_model.bin`. Transformers ≥ 4.50 refuses `torch.load` without `weights_only=True` (CVE-2025-32434). The code loads the state dict safely and passes it to `from_pretrained`.

**Code reference:** [`asr_backends.py:IndicWav2VecASR`](../src/whospoke/asr_backends.py), lines 300–354

### Shared Preprocessing

Both backends share the `_ASRBackend` base class which handles:

1. **Resampling** to 16 kHz if needed.
2. **Long-clip splitting** (`_split_long`): Clips longer than `max_chunk_s` (30 s for IndicConformer, 20 s for IndicWav2Vec) are cut at the quietest 25 ms frame within the last 5 seconds of the allowed window — avoiding word-boundary cuts.
3. **Length-sorted batching:** Utterances are sorted longest-first so each batch has similar lengths (less padding), and the first allocation (the largest) is reused by later, smaller ones.
4. **Minimum duration:** Clips shorter than 0.1 s are skipped entirely.

**Code reference:** [`asr_backends.py:_ASRBackend`](../src/whospoke/asr_backends.py), lines 101–137

### Model Selection (D9, D19, D24)

The ASR model was **chosen on Vaani**, not IndicVoices, because IndicConformer was trained on IndicVoices (D19 — it has a "home advantage" there). Vaani recordings are real-world phone conversations from across India that neither model saw in training.

**400 random single-speaker utterances per dataset:**

| Model | Vaani WER | Vaani CER | English words recognised | IndicVoices WER | Speed | GPU Peak |
|---|---|---|---|---|---|---|
| **IndicConformer 600M** (RNNT) | **21.0%** | **10.0%** | **78%** of 775 | 13.9% | 18× real-time | 3.7 GB |
| IndicWav2Vec Hindi (CTC) | 38.0% | 16.8% | 48% of 775 | 36.1% | 135× real-time | 5.0 GB |

**Verdict:** IndicConformer wins on independent audio by 17 WER points. It is 7× slower but still 18× real-time, which is plenty fast for the pipeline. IndicWav2Vec remains available via `--asr indicwav2vec`.

### Hinglish Romanisation

The proposal asks for "standardized Latin representations for code-switched text." The ASR writes everything — including English words — in Devanagari. The romaniser (`hinglish.py`) converts it back.

#### Step 1: English Loanword Lookup

A **2,600-entry TSV lexicon** (`resources/loanwords.tsv`) maps Devanagari forms of English words to their correct English spelling:

```
ऑफिस    office
मोबाइल   mobile
मीटिंग    meeting
होमवर्क    homework
```

**How the lexicon was built (D20):**
1. **456 hand-curated entries** from the 2,500 most frequent words in IndicVoices.
2. **+2,147 mined entries** from Vaani *train* transcripts. Vaani annotators tag every English word spoken inside Hindi with curly braces (e.g., `बिल्डिंग {building}`). Rules for inclusion:
   - Seen ≥ 2 times.
   - ≥ 70% agreement on English spelling.
   - Tagged as English in ≥ 50% of appearances.
   - Not in the hand-curated "Hindi" list (blocks false positives like `बस` "enough" → "bus").
3. Only *train* shards were used. The *test* shards (8,630 English words) were never touched during lexicon construction.

**Result:** 82.0% of English words spoken inside Hindi are spelled correctly in English.

#### Step 2: Rule-Based Hindi Romanisation

Words not in the lexicon (and not matching common-word overrides) are romanised by linguistic rules:

1. **Devanagari parsing (`_units`):** Each word is decomposed into `[consonant, vowel, nasal]` units. Nukta consonants (e.g., `ज़ = z`, `फ़ = f`) are handled. The inherent schwa is represented as lowercase `a`; long vowels as uppercase (`A` = ā, `I` = ī, `U` = ū).

2. **Schwa deletion (`_delete_schwas`):** Hindi's "silent a" rule:
   - **Word-final:** The inherent schwa of the last consonant is dropped (unless the word is monosyllabic). E.g., `नाम → naam`, not `naama`.
   - **Medial V-C-_-C-V pattern:** Scanning right to left, a schwa is deleted if the preceding unit has a vowel and the following unit is a consonant+vowel. E.g., `कमरा → kamra` (the `m`'s schwa is deleted), `समझना → samajhna`.

3. **Chat-style vowel spelling:** Long vowels (ā, ī, ū) are doubled **only** in closed monosyllables (`baat`, `teen`, `phool`) or word-initial independent `आ` (`aap`). Elsewhere, a single letter is used (`tha`, `hamara`, `nahi`). This matches how Hindi speakers actually type in WhatsApp and SMS.

4. **Common-word overrides:** ~40 very frequent words have conventional spellings that rules wouldn't produce: `में → mein`, `है → hai`, `नहीं → nahi`, `वो → wo`, etc.

5. **Anusvara (ं) contextual nasal:** The nasal marker is romanised as `m` before labial consonants (p, b, m) and `n` otherwise: `संभव → sambhav`, `हिंदी → hindi`.

**Code reference:** [`hinglish.py`](../src/whospoke/hinglish.py)

### Which Audio Each Turn is Transcribed From

This depends on the pipeline order and separation policy (D26):

**Order B (splice, 0.5 s context — default):**
- For turns that don't overlap with anyone: transcribed from the **original mixture**.
- For turns that overlap: the overlap region is separated. Inside that region only, the original audio is swapped for the separated voice that sounds most like this speaker (highest cosine similarity to their centroid embedding, lowest to the other speaker's). The swap uses **20 ms cross-fades**. The rest of the turn keeps the original audio.
- Turns longer than 25 s are split at the quietest 50 ms frame near the cut.

**Order A:**
- Every turn is transcribed from the **separated track** it was found on.

**Code reference:** [`pipeline.py:Pipeline._targeted_separation()`](../src/whospoke/pipeline.py), lines 237–302

### Key Files

| File | Role |
|---|---|
| [`src/whospoke/asr_backends.py`](../src/whospoke/asr_backends.py) | IndicConformer and IndicWav2Vec wrappers with batching, chunking, and ONNX/CUDA optimisations |
| [`src/whospoke/hinglish.py`](../src/whospoke/hinglish.py) | Devanagari → Hinglish romaniser (rules + lexicon) |
| [`src/whospoke/resources/loanwords.tsv`](../src/whospoke/resources/loanwords.tsv) | 2,600-entry English loanword dictionary |
| [`scripts/eval_asr.py`](../scripts/eval_asr.py) | Head-to-head ASR model comparison on Vaani |
| [`scripts/mine_loanwords.py`](../scripts/mine_loanwords.py) | Mines additional loanword pairs from Vaani train |
| [`results/asr_comparison.csv`](../results/asr_comparison.csv) | Raw ASR comparison numbers |

### Key Decisions

- **D9:** Both ASR models evaluated head-to-head (the proposal says "or").
- **D19:** Model chosen on Vaani (independent audio), not IndicVoices (where IndicConformer has a home advantage).
- **D20:** Lexicon built from Vaani *train* only; test words never used during construction.
- **D24:** IndicConformer chosen — wins by 17 WER points on independent audio.
- **D25:** TorchScript fuser crash fixed with a targeted context manager.
