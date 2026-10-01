# Milestone 4 — Semantic Post-Processing & Structuring

Milestone 4 consumes the Stage-3 `transcript.json` and turns it into a polished, human-readable report.

## Pipeline

```text
Stage 3 transcript.json
        │
        ▼
 bounded transcript chunks
        │
        ▼
 Airavata / local LLM
        │
        ├── cleaned Devanagari
        ├── cleaned Hinglish
        ├── faithful English translation
        ├── chunk summary
        ├── grounded keywords
        └── explicit action items + evidence line IDs
        │
        ▼
 validated + immutable timeline
        │
        ▼
 report.json + report.md
```

## Required deliverable

The final report contains:

- concise executive summary
- central topic
- factual key points
- keyword tags
- explicit action items, each linked to source line IDs
- speaker-separated cleaned dialogue
- faithful English translation for each dialogue line
- uncertainty notes for lines the LLM could not safely repair
- provenance including the source transcript SHA-256 and model name

## Guardrails

The LLM is not allowed to change speaker labels or timestamps. Every Stage-3 line must appear exactly once.
Hallucinated line IDs are dropped. Missing lines are restored from Stage 3 and marked uncertain. Action items are kept
only when their owner is an existing speaker and their evidence IDs point to real transcript lines.

The prompt explicitly forbids inventing people, places, dates, numbers, events, commitments, intentions or quotations.
The English translation must remain faithful to the source.

## Local Airavata

The default backend is an OpenAI-compatible local HTTP server configured as `ai4bharat/Airavata`. AI4Bharat's
model card documents Airavata as a 7B Hindi instruction-tuned model and documents local `llama.cpp` serving.

```bash
llama serve -hf ai4bharat/Airavata
```

Then:

```bash
python -m whospoke postprocess results/demo/transcript.json
```

or run all four stages directly:

```bash
python -m whospoke run my_recording.wav --postprocess
```

Default endpoint: `http://127.0.0.1:8080/v1`. Change it with `--llm-url`.

## Source code

- `src/whospoke/llm_postprocess.py` — backend interface, prompt framework, JSON parsing, guardrails and report renderer.
- `scripts/postprocess.py` — standalone Stage-4 command.
- `src/whospoke/pipeline.py` — optional Stage-4 integration after ASR.
- `src/whospoke/__main__.py` — `run --postprocess` and `postprocess` CLI commands.
- `tests/test_llm_postprocess.py` — deterministic guardrail tests.
