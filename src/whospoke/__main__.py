"""Command line for the complete WhoSpoke pipeline.

Examples::

    python -m whospoke run recording.wav
    python -m whospoke run recording.wav --postprocess
    python -m whospoke postprocess results/demo/transcript.json

Milestone 4 uses a local OpenAI-compatible LLM endpoint by default.  AI4Bharat's Airavata can
be served locally with llama.cpp; no hosted API key is required by this client.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .audio import load
from .diarization import Diarizer
from .llm_postprocess import DEFAULT_BASE_URL, DEFAULT_MODEL, PostProcessor

TUNED = Path(__file__).resolve().parents[2] / "results" / "tuned_params.json"


def tuned_diarizer(order: str, clustering: str) -> Diarizer:
    """Diarizer with the settings tuned on the dev set (falls back to defaults if not tuned yet)."""
    if TUNED.exists():
        p = json.loads(TUNED.read_text(encoding="utf-8")).get(f"{order}-{clustering}")
        if p:
            return Diarizer(p["clustering"], merge_sim=p["merge_sim"], min_cluster_frac=p["min_cluster_frac"],
                            overlap_aware=p["overlap_aware"], cluster_kwargs=p["cluster_kwargs"])
    return Diarizer(clustering)


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-url", default=DEFAULT_BASE_URL,
                        help=f"OpenAI-compatible LLM base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL,
                        help=f"LLM model identifier (default: {DEFAULT_MODEL})")
    parser.add_argument("--llm-temperature", type=float, default=float(os.getenv("WHOSPOKE_LLM_TEMPERATURE", "0.1")))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.getenv("WHOSPOKE_LLM_MAX_TOKENS", "4096")))
    parser.add_argument("--llm-chunk-chars", type=int, default=18000,
                        help="maximum source JSON size per Stage-4 cleaning call")
    parser.add_argument("--llm-timeout", type=int, default=180,
                        help="LLM HTTP timeout in seconds")


def build_postprocessor(args: argparse.Namespace) -> PostProcessor:
    return PostProcessor(model=args.llm_model, base_url=args.llm_url,
                          temperature=args.llm_temperature, max_tokens=args.llm_max_tokens,
                          chunk_chars=args.llm_chunk_chars, timeout_s=args.llm_timeout)


def run_audio(args: argparse.Namespace) -> None:
    from .pipeline import Pipeline

    post = build_postprocessor(args) if args.postprocess else None
    pipe = Pipeline(args.order, diarizer=tuned_diarizer(args.order, args.clustering), asr=args.asr,
                    postprocessor=post)
    result = pipe.run(load(args.audio), n_speakers=args.speakers, postprocess=args.postprocess)
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "results" / "runs" / Path(args.audio).stem
    result.save(out)
    print(result.transcript(hinglish=True))
    if result.report is not None:
        print(f"\nMilestone-4 report: {out / 'report.md'}")
    print(f"\nsaved to {out}  |  timings (s): {result.timings}")


def postprocess_transcript(args: argparse.Namespace) -> None:
    transcript = Path(args.transcript)
    out = Path(args.out) if args.out else transcript.parent / "milestone4"
    processor = build_postprocessor(args)
    report = processor.process_file(transcript)
    report.save(out)
    print(report.markdown())
    print(f"\nSaved Milestone-4 report to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="whospoke", description="Who spoke what and when — Hindi/Hinglish audio.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="process one audio file")
    run.add_argument("audio")
    run.add_argument("--out", default=None, help="output folder (default: results/runs/<file name>)")
    run.add_argument("--order", choices=["A", "B"], default="B")
    run.add_argument("--clustering", choices=["spectral", "gmm"], default="spectral")
    run.add_argument("--asr", choices=["indicconformer", "indicwav2vec"], default="indicconformer")
    run.add_argument("--speakers", type=int, default=None, help="number of speakers, if known")
    run.add_argument("--postprocess", action="store_true", help="run Milestone 4 after Stage 3")
    add_llm_args(run)

    post = sub.add_parser("postprocess", help="run Milestone 4 on an existing transcript.json")
    post.add_argument("transcript")
    post.add_argument("--out", default=None, help="output folder (default: <transcript-dir>/milestone4)")
    add_llm_args(post)

    args = ap.parse_args()
    if args.cmd == "run":
        run_audio(args)
    elif args.cmd == "postprocess":
        postprocess_transcript(args)


if __name__ == "__main__":
    main()
