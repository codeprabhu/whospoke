"""Stage 3 cross-check on Nirantar (AI4Bharat) Hindi — the third real-speech set from the proposal.

Nirantar ships as one ~198 GB archive with all languages shuffled (D7), so a teammate streamed it on
Colab and kept only Hindi; we score a random 800-clip sample of that. Its Hindi comes from the same
collection as IndicVoices, so before scoring we drop every speaker in the IndicVoices Hindi valid
split (our test speakers) and report how many clips/speakers also appear in the local IndicVoices files.

    python scripts/eval_asr_nirantar.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whospoke.audio import load  # noqa: E402
from whospoke.corpora import clean_transcript  # noqa: E402
from whospoke.metrics import corpus_rate  # noqa: E402
from whospoke.paths import RAW  # noqa: E402
from whospoke.pipeline import make_asr  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
NIR = RAW / "nirantar"


def indicvoices_ids(split_glob: str) -> tuple[set[str], set[str]]:
    speakers, files = set(), set()
    for f in sorted((RAW / "indicvoices/hindi").glob(split_glob)):
        t = pq.read_table(f, columns=["speaker_id", "audio_filepath"]).to_pandas()
        speakers |= set(t.speaker_id)
        files |= {Path(a["path"]).name for a in t.audio_filepath}
    return speakers, files


def main() -> None:
    full = pd.read_json(NIR / "hindi_metadata.jsonl", lines=True)
    sample = pd.read_json(NIR / "sample_800/metadata.jsonl", lines=True)
    full["file"] = full.audio_filepath.map(lambda p: Path(p).name)
    sample["file"] = sample.audio_filepath.map(lambda p: Path(p).name)

    overlap = {}
    for split, pattern in [("valid", "valid-*.parquet"), ("train_local_shards", "train-*.parquet")]:
        spk, files = indicvoices_ids(pattern)
        overlap[split] = {
            "indicvoices_speakers": len(spk),
            "shared_speakers": len(spk & set(full.speaker_id)),
            "nirantar_clips_with_identical_filename": int(full.file.isin(files).sum()),
        }
        if split == "valid":
            test_speakers = spk
    print(json.dumps(overlap, indent=1))

    kept = sample[~sample.speaker_id.isin(test_speakers)]
    kept = kept.assign(text=kept.text.fillna("").map(clean_transcript))
    kept = kept[(kept.text.str.len() > 0) & kept.duration.between(1, 20)].reset_index(drop=True)
    print(f"sample: {len(sample)} clips -> {len(kept)} after dropping test speakers / empty / >20 s")
    wavs = [load((NIR / "sample_800" / p).read_bytes()) for p in kept.audio_filepath]

    rows = []
    for model_name in ["indicconformer", "indicwav2vec"]:
        asr = make_asr(model_name)
        hyps = []
        for i in range(0, len(wavs), 8):
            hyps += asr.transcribe(wavs[i: i + 8])
        pairs = list(zip(kept.text, hyps))
        rows.append({"model": model_name, "set": "nirantar-hi-sample", "n": len(pairs),
                     "hours": round(kept.duration.sum() / 3600, 3),
                     "wer": corpus_rate(pairs, "word"), "cer": corpus_rate(pairs, "char")})
        print(rows[-1], flush=True)
        del asr
        torch.cuda.empty_cache()

    out = PROJECT / "results"
    pd.DataFrame(rows).to_csv(out / "asr_nirantar.csv", index=False)
    summary = {"sample_clips": len(sample), "scored_clips": len(kept),
               "scored_speakers": int(kept.speaker_id.nunique()),
               "full_hindi": {"clips": len(full), "hours": round(full.duration.sum() / 3600, 1),
                              "speakers": int(full.speaker_id.nunique())},
               "overlap_with_indicvoices": overlap, "table": rows}
    (out / "asr_nirantar.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
