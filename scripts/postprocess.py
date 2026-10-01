#!/usr/bin/env python3
"""Run WhoSpoke Milestone 4 on an existing Stage-3 transcript.json.

Example:
    python scripts/postprocess.py results/demo/transcript.json --out results/demo_m4

Start a local Airavata server first, for example with llama.cpp:
    llama serve -hf ai4bharat/Airavata
"""
from __future__ import annotations

import argparse
from pathlib import Path

from whospoke.llm_postprocess import DEFAULT_BASE_URL, DEFAULT_MODEL, PostProcessor


def main() -> None:
    ap = argparse.ArgumentParser(description="WhoSpoke Milestone 4 — LLM transcript clean-up and structuring")
    ap.add_argument("transcript", help="Stage-3 transcript.json produced by whospoke run")
    ap.add_argument("--out", default=None, help="output directory (default: <transcript-dir>/milestone4)")
    ap.add_argument("--url", default=DEFAULT_BASE_URL, help=f"LLM OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model identifier (default: {DEFAULT_MODEL})")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chunk-chars", type=int, default=18000)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    transcript = Path(args.transcript)
    out = Path(args.out) if args.out else transcript.parent / "milestone4"
    processor = PostProcessor(
        model=args.model,
        base_url=args.url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        chunk_chars=args.chunk_chars,
        timeout_s=args.timeout,
    )
    report = processor.process_file(transcript)
    report.save(out)
    print(report.markdown())
    print(f"\nSaved Milestone-4 report to {out}")


if __name__ == "__main__":
    main()
