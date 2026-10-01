# Index — where everything is

Start with [README.md](README.md) (what this is, headline results, how to run it). This page is the map.

## Read in this order

| # | File | What it answers |
|---|---|---|
| 1 | [README.md](README.md) | What the project does, the headline numbers, how to install and run it |
| 2 | [docs/RESULTS.md](docs/RESULTS.md) | Every result, explained in plain English, with the evidence for each design choice |
| 3 | [DEVIATIONS.md](DEVIATIONS.md) | Every place we differ from the professor's proposal, and why |
| 4 | [DECISIONS.md](DECISIONS.md) | Every judgement call (D1–D26), recorded as it was made |
| 5 | [docs/PIPELINE.md](docs/PIPELINE.md) | Technical reference: how each stage works, stage by stage |
| [docs/MILESTONE4.md](docs/MILESTONE4.md) | Stage-4 prompt, guardrail and execution reference |
| 6 | [docs/RESULTS_TABLES.md](docs/RESULTS_TABLES.md) | Raw result tables with 95 % confidence intervals (generated, not hand-written) |
| 7 | [Audio_Engineering_AI_Project_Proposal.docx](Audio_Engineering_AI_Project_Proposal.docx) | The original brief |

## For the team

| File | Purpose |
|---|---|
| [docs/team/WHATSAPP_MESSAGE.md](docs/team/WHATSAPP_MESSAGE.md) | Who does what before the 6 October review |
| [docs/team/RED_TEAM_AUDIT_PROMPT.md](docs/team/RED_TEAM_AUDIT_PROMPT.md) | Prompt for the independent audit (output → `audit/RED_TEAM_AUDIT.md`) |
| [docs/team/EMAIL_TO_PROFESSOR.md](docs/team/EMAIL_TO_PROFESSOR.md) | Draft email flagging the two design decisions before the review |
| [notebooks/02_walkthrough.ipynb](notebooks/02_walkthrough.ipynb) | Run the whole pipeline on one conversation, in Colab or locally |
| [notebooks/01_benchmark.ipynb](notebooks/01_benchmark.ipynb) | Speed, memory and model-size benchmark (the proposal's "benchmarking notebook") |

## Code — `src/whospoke/`

| Module | Stage | What it does |
|---|---|---|
| `pipeline.py` | all | Runs the stages in Order A or Order B and writes the outputs |
| `__main__.py` | all | Command line: `python -m whospoke run audio.wav` |
| `separation.py` | 1 | Conv-TasNet / SepFormer separation with windowing, gain fit and stitching |
| `vad.py` | 2 | Where is anyone speaking? Where are two people speaking? (pyannote segmentation) |
| `diarization.py` | 2 | Voice fingerprints → clusters → a who-spoke-when timeline (both orders) |
| `clustering.py` | 2 | Spectral clustering and GMM, written from scratch, plus cluster clean-up |
| `asr_backends.py` | 3 | IndicConformer and IndicWav2Vec speech recognisers |
| `hinglish.py` | 3 | Devanagari → Hinglish (Latin) romaniser with an English-loanword lexicon |
| `llm_postprocess.py` | 4 | Guarded LLM clean-up, translation, summary, keywords and action extraction |
| `resources/loanwords.tsv` | 3 | 2,600 Devanagari → English spellings (hand-made + mined from Vaani *train*) |
| `metrics.py` | eval | SI-SDR, DER, JER, WER/CER, cpWER (who-said-what error) |
| `synth.py` | eval | Builds test conversations from real speech with exact labels |
| `noise.py` | eval | Village and market background noise (DEMAND + ESC-50) |
| `corpora.py` | data | Downloads and reads IndicVoices, Vaani, DEMAND, ESC-50 |
| `audio.py`, `paths.py` | — | Audio helpers; where the data folder is |

## Scripts — `scripts/` (run in this order to reproduce everything)

| Script | Produces |
|---|---|
| `fetch_data.py` | DEMAND, ESC-50 and Vaani downloads (IndicVoices comes with the next step) |
| `build_dataset.py` | Dev and test conversations (`<data>/synth/{dev,test}/`) |
| `mine_loanwords.py` | The mined part of `loanwords.tsv` |
| `eval_separation.py` | `results/separation_{split}.csv` — Stage 1 |
| `tune_diarization.py` | `results/tuned_params.json` — Stage 2 settings, chosen on **dev** only |
| `tune_separation_policy.py` | `results/separation_policy_dev.csv` — how Order B feeds separated audio to the ASR (**dev** only) |
| `eval_asr.py` | `results/asr_comparison.*` — Stage 3 model choice (on Vaani) |
| `evaluate.py` | `results/eval_test_*.csv` — every system end to end on the **test** set |
| `make_report.py` | `docs/RESULTS_TABLES.md` + `results/figures/*.png` |
| `build_notebooks.py` | `notebooks/*.ipynb` (executed) |
| `bench_asr_backends.py` | `results/asr_backend_bench.json` — ASR decoding-mode benchmark |
| `postprocess.py` | Stage-3 `transcript.json` → Stage-4 `report.json` + `report.md` |

## Tests — `tests/`

`pytest` runs the fast suite (metrics, simulation, clustering, timeline logic, separation stitching, romaniser);
`pytest -m slow` also runs the real models end to end.

## Results — `results/`

| File | Contents |
|---|---|
| `separation_*.csv` | SI-SDR per conversation, per separator |
| `tuning_diarization_dev.csv`, `tuned_params.json` | Full dev grid search and the frozen settings |
| `asr_comparison.{csv,json}`, `asr_comparison_utterances.csv` | ASR model comparison, per utterance |
| `eval_test_indicconformer.csv` | Every system × every test conversation |
| `eval_test_indicconformer/*.json` | Every system's full transcript for every test conversation |
| `figures/` | Figures used in RESULTS.md and the slides |
| `demo/` | Example output of one full run (timeline, transcripts, subtitles) |
| `runs/<name>/report.{json,md}` | Stage-4 semantic report when `--postprocess` is enabled |
