"""Milestone 4 — LLM semantic post-processing and transcript structuring.

The Stage-3 transcript is deliberately treated as *evidence*, not as free-form text for the
LLM to rewrite however it wants.  The post-processor therefore uses three guardrails:

1. Every source line receives an immutable ``line_id`` and immutable speaker/timestamps.
2. The model is explicitly forbidden from inventing facts, dialogue, speakers, numbers, names,
   dates or actions not supported by the source transcript.
3. The generated JSON is validated before it is accepted. Missing, duplicated or reordered
   source lines are repaired with the original Stage-3 line rather than silently dropped.

The default backend is an OpenAI-compatible local HTTP server (for example llama.cpp running
AI4Bharat/Airavata).  No OpenAI Python dependency is required; the client uses urllib from the
standard library.  A deterministic ``StaticLLM`` backend is included for tests and notebooks.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


DEFAULT_MODEL = os.getenv("WHOSPOKE_LLM_MODEL", "ai4bharat/Airavata")
DEFAULT_BASE_URL = os.getenv("WHOSPOKE_LLM_URL", "http://127.0.0.1:8080/v1")
DEFAULT_TEMPERATURE = float(os.getenv("WHOSPOKE_LLM_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("WHOSPOKE_LLM_MAX_TOKENS", "4096"))

SYSTEM_PROMPT = """You are Milestone 4 of a speech pipeline called WhoSpoke.

You receive a raw speaker-attributed Hindi/English/Hinglish transcript produced by ASR.
Your job is conservative transcript post-processing, not creative writing.

NON-NEGOTIABLE RULES
1. Treat the source transcript as the only source of truth.
2. Never invent facts, people, organizations, dates, times, numbers, places, events, actions,
   intentions, diagnoses, quotations, or commitments.
3. Preserve every source line exactly once. Do not merge lines, split lines, reorder lines, add
   new lines, or change speaker labels or timestamps.
4. You may repair an obvious ASR artifact, duplicated word, punctuation issue, or small grammatical
   fragment ONLY when the intended wording is strongly supported by the surrounding transcript.
   If uncertain, keep the original wording and flag the line as uncertain.
5. Preserve proper nouns and technical terms when uncertain instead of guessing.
6. Keep code-switching natural. Do not translate a term merely because it is English in a Hindi
   sentence.
7. English translations must be faithful paraphrases of the source line. Never add information.
8. Action items may only be extracted when the transcript contains an explicit request, instruction,
   commitment, or clearly stated next step. Otherwise return an empty list.
9. Keywords must be grounded in words/concepts that occur in the transcript. Do not add generic
   filler keywords.
10. Return ONLY valid JSON. No markdown fences, no commentary before or after the JSON.
"""

CHUNK_PROMPT = """Process the following source transcript lines.

Return JSON with this exact top-level shape:
{
  "cleaned_dialogue": [
    {
      "line_id": 1,
      "speaker": "Speaker_A",
      "start": 0.0,
      "end": 1.0,
      "text": "cleaned Devanagari/ASR text",
      "hinglish": "cleaned Hinglish text",
      "translation": "faithful English translation",
      "uncertain": false,
      "note": ""
    }
  ],
  "chunk_summary": "One or two factual sentences describing this chunk.",
  "keywords": ["keyword1", "keyword2"],
  "action_items": [
    {
      "owner": "Speaker_A",
      "action": "Explicit action supported by the transcript",
      "evidence_line_ids": [3, 4]
    }
  ]
}

The ``cleaned_dialogue`` array MUST contain every supplied line_id exactly once, in the same order.
Keep speaker/start/end exactly as supplied. When uncertain, preserve the supplied text and set
``uncertain`` to true with a short reason in ``note``.
"""

SYNTHESIS_PROMPT = """Using the already-validated chunk results below, produce a final structured report.

Return JSON with exactly this shape:
{
  "title": "A concise factual title",
  "topic": "The central topic, only if supported by the transcript",
  "executive_summary": "A concise factual summary in English. Do not invent anything.",
  "key_points": ["factual point", "factual point"],
  "keywords": ["keyword1", "keyword2"],
  "action_items": [
    {
      "owner": "Speaker_A",
      "action": "Explicit action supported by the transcript",
      "evidence_line_ids": [3, 4]
    }
  ]
}

Rules:
- Never add unsupported details.
- If no explicit action exists, use an empty action_items array.
- Keywords and key points must be grounded in the supplied chunk evidence.
- Keep the summary concise and useful for a human reader.
"""


class LLMBackend(Protocol):
    """Small provider interface so the report logic is independent of the model runner."""

    model_name: str

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        ...


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """Call a local OpenAI-compatible ``/v1/chat/completions`` endpoint.

    This works with llama.cpp, vLLM and other local servers implementing the common Chat
    Completions response shape.  The default target is a local Airavata server.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout_s: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout_s = timeout_s

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise LLMError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Could not reach LLM server at {self.base_url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("LLM server returned invalid JSON") from exc

        try:
            choice = body["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            if content is None:
                content = choice.get("text")
            if not isinstance(content, str):
                raise TypeError
            return content.strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {body!r}") from exc


class StaticLLM:
    """Deterministic backend used for unit tests and offline validation."""

    def __init__(self, responses: list[str] | str, model_name: str = "static-test"):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.model_name = model_name
        self.calls = 0

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        if not self.responses:
            raise LLMError("StaticLLM has no remaining responses")
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]


@dataclass
class CleanedLine:
    line_id: int
    speaker: str
    start: float
    end: float
    text: str
    hinglish: str
    translation: str
    uncertain: bool = False
    note: str = ""


@dataclass
class ActionItem:
    owner: str
    action: str
    evidence_line_ids: list[int] = field(default_factory=list)


@dataclass
class PostProcessedReport:
    title: str
    topic: str
    executive_summary: str
    key_points: list[str]
    keywords: list[str]
    action_items: list[ActionItem]
    cleaned_dialogue: list[CleanedLine]
    uncertain_lines: list[int]
    source_sha256: str
    source_line_count: int
    llm_model: str
    generated_at_utc: str
    pipeline_order: str | None = None
    source_audio_duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "topic": self.topic,
            "executive_summary": self.executive_summary,
            "key_points": self.key_points,
            "keywords": self.keywords,
            "action_items": [asdict(x) for x in self.action_items],
            "cleaned_dialogue": [asdict(x) for x in self.cleaned_dialogue],
            "uncertain_lines": self.uncertain_lines,
            "provenance": {
                "source_sha256": self.source_sha256,
                "source_line_count": self.source_line_count,
                "llm_model": self.llm_model,
                "generated_at_utc": self.generated_at_utc,
                "pipeline_order": self.pipeline_order,
                "source_audio_duration_s": self.source_audio_duration_s,
            },
        }

    def markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            "## Executive Summary",
            self.executive_summary or "No summary was generated.",
            "",
            "## Topic",
            self.topic or "Not confidently identified from the transcript.",
            "",
            "## Key Points",
        ]
        if self.key_points:
            lines.extend(f"- {p}" for p in self.key_points)
        else:
            lines.append("- None extracted.")
        lines += ["", "## Keywords"]
        lines.append(", ".join(self.keywords) if self.keywords else "None extracted.")
        lines += ["", "## Action Items"]
        if self.action_items:
            for item in self.action_items:
                ev = ", ".join(str(i) for i in item.evidence_line_ids)
                lines.append(f"- **{item.owner}**: {item.action} (evidence: line {ev})")
        else:
            lines.append("- None explicitly stated in the transcript.")
        lines += ["", "## Cleaned Speaker Dialogue", "",
                  "| Time | Speaker | Cleaned transcript | Hinglish | English |", 
                  "|---|---|---|---|---|"]
        for line in self.cleaned_dialogue:
            flag = " ⚠️" if line.uncertain else ""
            time_range = f"{_fmt_time(line.start)}–{_fmt_time(line.end)}"
            lines.append(
                f"| {time_range} | {line.speaker} | {_md(line.text)}{flag} | {_md(line.hinglish)} | {_md(line.translation)} |"
            )
        lines += ["", "## ASR / Post-processing Uncertainty"]
        uncertain = [x for x in self.cleaned_dialogue if x.uncertain]
        if uncertain:
            for x in uncertain:
                lines.append(f"- Line {x.line_id}: {x.note or 'uncertain wording retained'}")
        else:
            lines.append("- No lines were flagged by the post-processor.")
        lines += ["", "## Provenance", "",
                  f"- Source transcript SHA-256: `{self.source_sha256}`",
                  f"- Source lines: {self.source_line_count}",
                  f"- LLM: `{self.llm_model}`",
                  f"- Pipeline order: `{self.pipeline_order or 'unknown'}`",
                  f"- Generated: `{self.generated_at_utc}`",
                  "- Speaker labels and timestamps above are inherited from Stage 3 and are not altered by Stage 4.",
                  ""]
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "report.md").write_text(self.markdown(), encoding="utf-8")
        return out


def _fmt_time(t: float) -> str:
    total = int(round(t))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _md(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON while tolerating a model wrapping the object in a few stray words."""
    clean = _strip_code_fences(text)
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("LLM did not return a JSON object")
        try:
            obj = json.loads(clean[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned malformed JSON: {clean[:500]!r}") from exc
    if not isinstance(obj, dict):
        raise LLMError("LLM JSON root must be an object")
    return obj


def _source_fingerprint(source: dict[str, Any]) -> str:
    canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _line_records(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    raw = transcript.get("lines", [])
    if not isinstance(raw, list):
        raise ValueError("transcript JSON 'lines' must be a list")
    records = []
    for idx, line in enumerate(raw, 1):
        if not isinstance(line, dict):
            raise ValueError(f"line {idx} is not an object")
        text = str(line.get("text", ""))
        records.append({
            "line_id": idx,
            "speaker": str(line.get("speaker", "Speaker_Unknown")),
            "start": float(line.get("start", 0.0)),
            "end": float(line.get("end", 0.0)),
            "text": text,
            "hinglish": str(line.get("hinglish", "")),
            "audio": str(line.get("audio", "mixture")),
        })
    return records


def _chunk_lines(lines: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    if not lines:
        return []
    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    chars = 0
    for line in lines:
        line_chars = len(json.dumps(line, ensure_ascii=False))
        if cur and chars + line_chars > max_chars:
            out.append(cur)
            cur = []
            chars = 0
        cur.append(line)
        chars += line_chars
    if cur:
        out.append(cur)
    return out


class PostProcessor:
    """Conservative, validated Stage-4 transcript post-processor."""

    def __init__(self, backend: LLMBackend | None = None, *, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS, chunk_chars: int = 18000, timeout_s: int = 180):
        self.backend = backend or OpenAICompatibleLLM(base_url=base_url, model=model, timeout_s=timeout_s)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chunk_chars = chunk_chars

    def process_transcript(self, transcript: dict[str, Any]) -> PostProcessedReport:
        source_lines = _line_records(transcript)
        chunks = _chunk_lines(source_lines, self.chunk_chars)
        if not chunks:
            return PostProcessedReport(
                title="Empty WhoSpoke Transcript", topic="", executive_summary="No speech was transcribed.",
                key_points=[], keywords=[], action_items=[], cleaned_dialogue=[], uncertain_lines=[],
                source_sha256=_source_fingerprint(transcript), source_line_count=0,
                llm_model=self.backend.model_name, generated_at_utc=_utc_now(),
                pipeline_order=transcript.get("order"), source_audio_duration_s=transcript.get("duration_s"),
            )

        chunk_results: list[dict[str, Any]] = []
        cleaned_all: list[CleanedLine] = []
        for chunk in chunks:
            user = CHUNK_PROMPT + "\n\nSOURCE TRANSCRIPT LINES:\n" + json.dumps(chunk, ensure_ascii=False, indent=2)
            raw = self.backend.generate(SYSTEM_PROMPT, user, temperature=self.temperature, max_tokens=self.max_tokens)
            result = parse_json_object(raw)
            cleaned = _validate_chunk(result, chunk)
            cleaned_all.extend(cleaned)
            chunk_results.append({
                "chunk_summary": str(result.get("chunk_summary", "")).strip(),
                "keywords": _clean_string_list(result.get("keywords", [])),
                "action_items": [asdict(x) for x in _clean_actions(
                    result.get("action_items", []), {x.line_id for x in cleaned},
                    {x.speaker for x in cleaned},
                )],
            })

        synthesis_input = {
            "chunks": chunk_results,
            "dialogue_evidence": [
                {
                    "line_id": x.line_id,
                    "speaker": x.speaker,
                    "text": x.text,
                    "translation": x.translation,
                }
                for x in cleaned_all
            ],
        }
        synth_user = SYNTHESIS_PROMPT + "\n\nEVIDENCE:\n" + json.dumps(synthesis_input, ensure_ascii=False, indent=2)
        raw = self.backend.generate(SYSTEM_PROMPT, synth_user, temperature=self.temperature, max_tokens=self.max_tokens)
        synth = parse_json_object(raw)

        actions = _clean_actions(synth.get("action_items", []), {x.line_id for x in cleaned_all},
                                 {x.speaker for x in cleaned_all})
        action_source = actions or _flatten_actions(chunk_results)
        keywords = _dedupe(_clean_string_list(synth.get("keywords", [])))
        if not keywords:
            keywords = _dedupe(k for c in chunk_results for k in c["keywords"])
        # Last safety pass: exact source line coverage and stable speaker/time fields.
        cleaned_all = _finalize_lines(cleaned_all, source_lines)
        uncertain = [x.line_id for x in cleaned_all if x.uncertain]
        return PostProcessedReport(
            title=_safe_title(str(synth.get("title", "WhoSpoke Transcript Report"))),
            topic=str(synth.get("topic", "")).strip(),
            executive_summary=str(synth.get("executive_summary", "")).strip(),
            key_points=_clean_string_list(synth.get("key_points", []))[:8],
            keywords=keywords[:12],
            action_items=action_source[:12],
            cleaned_dialogue=cleaned_all,
            uncertain_lines=uncertain,
            source_sha256=_source_fingerprint(transcript),
            source_line_count=len(source_lines),
            llm_model=self.backend.model_name,
            generated_at_utc=_utc_now(),
            pipeline_order=transcript.get("order"),
            source_audio_duration_s=transcript.get("duration_s"),
        )

    def process_file(self, path: str | Path) -> PostProcessedReport:
        p = Path(path)
        transcript = json.loads(p.read_text(encoding="utf-8"))
        report = self.process_transcript(transcript)
        return report


def _validate_chunk(result: dict[str, Any], source: list[dict[str, Any]]) -> list[CleanedLine]:
    by_id = {x["line_id"]: x for x in source}
    raw_lines = result.get("cleaned_dialogue")
    if not isinstance(raw_lines, list):
        raw_lines = []
    parsed: dict[int, CleanedLine] = {}
    for item in raw_lines:
        if not isinstance(item, dict):
            continue
        try:
            line_id = int(item["line_id"])
        except (KeyError, TypeError, ValueError):
            continue
        src = by_id.get(line_id)
        if src is None:
            # Hallucinated line id: ignore it entirely.
            continue
        # Speaker and timing are immutable Stage-3 evidence. Ignore any LLM changes.
        parsed[line_id] = CleanedLine(
            line_id=line_id,
            speaker=src["speaker"],
            start=src["start"],
            end=src["end"],
            text=str(item.get("text", src["text"])).strip() or src["text"],
            hinglish=str(item.get("hinglish", src["hinglish"])).strip() or src["hinglish"],
            translation=str(item.get("translation", "")).strip(),
            uncertain=bool(item.get("uncertain", False)),
            note=str(item.get("note", "")).strip(),
        )
    # Missing or duplicated source lines are repaired deterministically from Stage 3.
    out = []
    for src in source:
        if src["line_id"] in parsed:
            line = parsed[src["line_id"]]
            if not line.translation:
                line.translation = src["hinglish"] or src["text"]
                line.uncertain = True
                line.note = line.note or "The LLM did not supply a translation; source wording retained."
            out.append(line)
        else:
            out.append(CleanedLine(
                line_id=src["line_id"], speaker=src["speaker"], start=src["start"], end=src["end"],
                text=src["text"], hinglish=src["hinglish"], translation=src["hinglish"] or src["text"],
                uncertain=True, note="LLM omitted this line; original Stage-3 wording retained.",
            ))
    return out


def _finalize_lines(cleaned: list[CleanedLine], source: list[dict[str, Any]]) -> list[CleanedLine]:
    if len(cleaned) != len(source):
        return _validate_chunk({"cleaned_dialogue": [asdict(x) for x in cleaned]}, source)
    expected = [x["line_id"] for x in source]
    actual = [x.line_id for x in cleaned]
    if actual != expected or len(set(actual)) != len(actual):
        return _validate_chunk({"cleaned_dialogue": [asdict(x) for x in cleaned]}, source)
    return cleaned


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _clean_actions(value: Any, valid_ids: set[int], valid_speakers: set[str] | None = None) -> list[ActionItem]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if not isinstance(item, dict):
            continue
        owner = str(item.get("owner", "")).strip()
        action = str(item.get("action", "")).strip()
        ids = []
        for x in item.get("evidence_line_ids", []):
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i in valid_ids and i not in ids:
                ids.append(i)
        if valid_speakers is not None and owner not in valid_speakers:
            continue
        if owner and action and ids:
            out.append(ActionItem(owner, action, ids))
    return out


def _flatten_actions(chunk_results: list[dict[str, Any]]) -> list[ActionItem]:
    out: list[ActionItem] = []
    seen = set()
    for chunk in chunk_results:
        for item in chunk["action_items"]:
            if isinstance(item, ActionItem):
                action = item
            else:
                cleaned = _clean_actions([item], set(range(1, 10**9)))
                if not cleaned:
                    continue
                action = cleaned[0]
            key = (action.owner, action.action, tuple(action.evidence_line_ids))
            if key not in seen:
                seen.add(key)
                out.append(action)
    return out


def _dedupe(items: Any) -> list[str]:
    if isinstance(items, str):
        items = [items]
    out = []
    for item in items or []:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _safe_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120] or "WhoSpoke Transcript Report"


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_MODEL", "DEFAULT_BASE_URL", "LLMError", "LLMBackend",
    "OpenAICompatibleLLM", "StaticLLM", "CleanedLine", "ActionItem",
    "PostProcessedReport", "PostProcessor", "parse_json_object",
]
