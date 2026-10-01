"""The full "who spoke what and when" pipeline, in both orders.

Order A (as written in the proposal)
    mixture ──separate──▶ 2 tracks ──diarize tracks jointly──▶ turns (+track) ──ASR on the track──▶ transcript

Order B (overlap-targeted separation)
    mixture ──diarize + detect overlaps──▶ turns ──separate ONLY the overlapping stretches──▶ ASR ──▶ transcript
            (everything outside overlaps is transcribed straight from the mixture)

Each stage is timed and its peak GPU memory recorded, for the benchmark notebook.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .audio import SR, fmt_ts
from .diarization import Diarization, Diarizer, PyannoteDiarizer, Turn, embed
from .hinglish import romanise
from .separation import Separator


@dataclass
class Line:
    speaker: str
    start: float
    end: float
    text: str                 # Devanagari, as recognised
    hinglish: str             # romanised (Latin) version
    audio: str = "mixture"    # which audio was transcribed: mixture | separated | track-<k>

    def render(self, hinglish: bool = False) -> str:
        return f"[{fmt_ts(self.start)} - {fmt_ts(self.end)}] {self.speaker}: {self.hinglish if hinglish else self.text}"


@dataclass
class Result:
    order: str
    diarization: Diarization
    lines: list[Line]
    timings: dict[str, float] = field(default_factory=dict)
    gpu_peak_mb: dict[str, float] = field(default_factory=dict)
    duration_s: float = 0.0
    report: dict | None = None          # Milestone 4 LLM report (optional; disabled by default)
    report_markdown: str | None = None

    def text_by_speaker(self) -> dict[str, str]:
        out: dict[str, list[str]] = {}
        for ln in sorted(self.lines, key=lambda l: l.start):
            out.setdefault(ln.speaker, []).append(ln.text)
        return {k: " ".join(v) for k, v in out.items()}

    def transcript(self, hinglish: bool = False) -> str:
        return "\n".join(ln.render(hinglish) for ln in sorted(self.lines, key=lambda l: l.start) if ln.text)

    def srt(self, hinglish: bool = True) -> str:
        def ts(t: float) -> str:
            ms = int(round(t * 1000))
            h, ms = divmod(ms, 3_600_000)
            m, ms = divmod(ms, 60_000)
            s, ms = divmod(ms, 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        blocks = [f"{i}\n{ts(l.start)} --> {ts(l.end)}\n{l.speaker}: {l.hinglish if hinglish else l.text}"
                  for i, l in enumerate(sorted(self.lines, key=lambda l: l.start), 1) if l.text]
        return "\n\n".join(blocks) + "\n"

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.diarization.save(out / "timeline.json")
        (out / "transcript.txt").write_text(self.transcript() + "\n", encoding="utf-8")
        (out / "transcript_hinglish.txt").write_text(self.transcript(hinglish=True) + "\n", encoding="utf-8")
        (out / "transcript.srt").write_text(self.srt(), encoding="utf-8")
        (out / "transcript.json").write_text(json.dumps({
            "order": self.order, "duration_s": self.duration_s, "timings_s": self.timings,
            "gpu_peak_mb": self.gpu_peak_mb, "speakers": self.diarization.speakers,
            "lines": [asdict(l) for l in sorted(self.lines, key=lambda l: l.start)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        if self.report is not None:
            (out / "report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.report_markdown:
                (out / "report.md").write_text(self.report_markdown, encoding="utf-8")
        return out


def make_asr(name: str = "indicconformer", device: str | None = None, decoding: str = "rnnt"):
    """IndicConformer uses RNNT decoding by default (≈2 WER points better than CTC, still ~27× real time on GPU)."""
    from . import asr_backends

    if name == "indicconformer":
        return asr_backends.IndicConformerASR(device=device, decoding=decoding)
    if name == "indicwav2vec":
        return asr_backends.IndicWav2VecASR(device=device)
    raise ValueError(f"unknown ASR backend {name!r}")


class _Stage:
    """Context manager: wall time + peak GPU memory of one stage.

    PyTorch's own counter misses memory held by ONNX Runtime (the IndicConformer ASR), so a background
    thread also samples the whole device's used memory every 10 ms. ``gpu_peak_mb`` is that device-wide
    peak (everything loaded on the GPU at that moment — what decides whether the pipeline fits on a card).
    """

    def __init__(self, result: Result, name: str):
        self.result, self.name = result, name

    def _poll(self) -> None:
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            self._peak = max(self._peak, total - free)
            self._stop.wait(0.01)

    def __enter__(self):
        import threading

        self._peak = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self._stop.set()
            self._thread.join()
            self.result.gpu_peak_mb[self.name] = round(self._peak / 2**20, 1)
        self.result.timings[self.name] = round(time.perf_counter() - self.t, 3)


def overlap_intervals(turns: list[Turn]) -> list[tuple[float, float]]:
    """Time spans where two or more (different-speaker) turns are active."""
    ev = sorted([(t.start, 1) for t in turns] + [(t.end, -1) for t in turns])
    out, active, start = [], 0, 0.0
    for time_, d in ev:
        if active < 2 <= active + d:
            start = time_
        elif active >= 2 > active + d and time_ > start:
            out.append((start, time_))
        active += d
    return out


def split_long(start: float, end: float, wav: np.ndarray, max_s: float) -> list[tuple[float, float]]:
    """Split [start, end] into pieces ≤ ``max_s`` at the quietest 50 ms frame near each cut."""
    pieces, a = [], start
    while end - a > max_s:
        lo, hi = a + 0.6 * max_s, a + max_s
        seg = wav[int(lo * SR): int(hi * SR)]
        hop = int(0.05 * SR)
        n = max(1, len(seg) // hop)
        e = np.mean(seg[: n * hop].reshape(n, hop) ** 2, axis=1) if n * hop <= len(seg) and n > 0 else np.zeros(1)
        cut = lo + float(np.argmin(e)) * 0.05
        pieces.append((a, cut))
        a = cut
    pieces.append((a, end))
    return pieces


class Pipeline:
    def __init__(self, order: str = "B", separator: str | None = "convtasnet",
                 diarizer: Diarizer | PyannoteDiarizer | None = None, asr: str = "indicconformer",
                 max_segment_s: float = 25.0, device: str | None = None, asr_model=None,
                 sep_mode: str = "splice", sep_context_s: float = 0.5, postprocessor=None):
        if order not in ("A", "B"):
            raise ValueError("order must be 'A' or 'B'")
        if order == "A" and separator is None:
            raise ValueError("Order A needs a separator")
        self.order = order
        self.device = device
        self.separator = Separator(separator, device=device) if separator else None
        self.diarizer = diarizer or Diarizer(device=device)
        self.asr = asr_model or make_asr(asr, device)
        self.max_segment_s = max_segment_s
        if sep_mode not in ("splice", "turn"):
            raise ValueError("sep_mode must be 'splice' or 'turn'")
        self.sep_mode = sep_mode
        self.sep_context_s = sep_context_s
        self.postprocessor = postprocessor

    @property
    def name(self) -> str:
        sep = self.separator.name if self.separator else "nosep"
        return f"{self.order}|{sep}|{self.diarizer.name}|{self.asr.name}"

    def warmup(self) -> None:
        """Load every model once so that timings measure processing, not loading."""
        x = (np.random.default_rng(0).standard_normal(3 * SR) * 0.01).astype(np.float32)
        self.run(x, postprocess=False)

    # ------------------------------------------------------------------ main entry point
    def run(self, wav: np.ndarray, n_speakers: int | None = None, diarization: Diarization | None = None,
            postprocess: bool | None = None) -> Result:
        """Process a 16 kHz mono recording. ``diarization`` may be supplied (oracle experiments).

        ``postprocess`` controls Milestone 4. It defaults to whether a postprocessor was supplied
        to the constructor, preserving the Stage-1–3 behaviour for existing callers.
        """
        wav = np.asarray(wav, np.float32)
        res = Result(self.order, Diarization([]), [], duration_s=round(len(wav) / SR, 3))
        t0 = time.perf_counter()
        if self.order == "A":
            with _Stage(res, "separation"):
                streams = self.separator.separate(wav)
            with _Stage(res, "diarization"):
                res.diarization = diarization or self._diarize_streams(streams, n_speakers)
            with _Stage(res, "asr"):
                units = [(t, streams[t.stream if t.stream is not None else 0], f"track-{t.stream}")
                         for t in res.diarization.turns]
                res.lines = self._transcribe(units)
        else:
            with _Stage(res, "diarization"):
                res.diarization = diarization or self.diarizer(wav, n_speakers)
            with _Stage(res, "separation"):
                units = self._targeted_separation(wav, res.diarization)
            with _Stage(res, "asr"):
                res.lines = self._transcribe(units)
        do_postprocess = self.postprocessor is not None if postprocess is None else postprocess
        if do_postprocess:
            if self.postprocessor is None:
                raise ValueError("postprocess=True requires a Stage-4 postprocessor")
            with _Stage(res, "llm_postprocess"):
                transcript = {
                    "order": res.order, "duration_s": res.duration_s, "timings_s": res.timings,
                    "gpu_peak_mb": res.gpu_peak_mb, "speakers": res.diarization.speakers,
                    "lines": [asdict(l) for l in sorted(res.lines, key=lambda l: l.start)],
                }
                report = self.postprocessor.process_transcript(transcript)
                res.report = report.to_dict()
                res.report_markdown = report.markdown()
        res.timings["total"] = round(time.perf_counter() - t0, 3)
        return res

    def _diarize_streams(self, streams: np.ndarray, n_speakers: int | None) -> Diarization:
        if not isinstance(self.diarizer, Diarizer):
            raise TypeError("Order A needs our Diarizer (it clusters the separated tracks jointly)")
        return self.diarizer.diarize_streams(streams, n_speakers)

    # ------------------------------------------------------------------ Order B separation
    def _overlap_regions(self, diar: Diarization) -> list[tuple[float, float]]:
        """Where to separate: overlaps found by the overlap detector ∪ places where two turns overlap."""
        spans = sorted(list(diar.info.get("overlap", [])) + overlap_intervals(diar.turns))
        merged: list[list[float]] = []
        for a, b in spans:
            if b - a < 0.1:
                continue
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    def _targeted_separation(self, wav: np.ndarray, diar: Diarization) -> list[tuple[Turn, np.ndarray, str]]:
        """Per turn, choose the audio to transcribe.

        ``sep_mode="splice"`` (default): each overlap region is separated once (with ``sep_context_s`` of context
        on both sides), and inside the region the turn's speaker is replaced by the separated voice that sounds
        most like them; outside overlap regions the turn keeps the original mixture.
        ``sep_mode="turn"``: the whole turn is replaced by its separated voice (earlier design, kept for comparison).
        """
        turns = diar.turns
        regions = self._overlap_regions(diar)
        if self.separator is None or not regions or not turns:
            return [(t, wav, "mixture") for t in turns]
        centroids = self._centroids(wav, turns, regions, diar)
        n = len(wav)

        def pick(est: np.ndarray, span_s: float, t: Turn, others: list[Turn]) -> np.ndarray:
            # Keep the separated voice that sounds most like this speaker (and least like the other one).
            fp = np.concatenate([embed(est[k], [(0.0, span_s)], self.device) for k in range(2)])
            fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9
            score = fp @ centroids[t.speaker]
            other = max(others, key=lambda o: min(o.end, t.end) - max(o.start, t.start)).speaker if others else None
            if other in centroids:
                score = score - fp @ centroids[other]
            return est[int(np.argmax(score))]

        units = []
        if self.sep_mode == "turn":
            for t in turns:
                if t.speaker not in centroids or not any(a < t.end and b > t.start for a, b in regions):
                    units.append((t, wav, "mixture"))
                    continue
                a, b = int(t.start * SR), int(t.end * SR)
                est = self.separator.separate(wav[a:b])
                others = [o for o in turns if o.speaker != t.speaker and o.start < t.end and o.end > t.start]
                track = np.zeros_like(wav)
                track[a:b] = pick(est, (b - a) / SR, t, others)
                units.append((t, track, "separated"))
            return units

        # splice mode: separate each region once, with context
        ctx = int(self.sep_context_s * SR)
        separated = {}
        for r0, r1 in regions:
            a, b = max(0, int(r0 * SR) - ctx), min(n, int(r1 * SR) + ctx)
            separated[(r0, r1)] = (a, b, self.separator.separate(wav[a:b]))
        fade = int(0.02 * SR)
        for t in turns:
            hits = [(r0, r1) for r0, r1 in regions if r0 < t.end and r1 > t.start]
            if not hits or t.speaker not in centroids:
                units.append((t, wav, "mixture"))
                continue
            track = wav.copy()
            for r0, r1 in hits:
                a, b, est = separated[(r0, r1)]
                others = [o for o in turns if o.speaker != t.speaker and o.start < r1 and o.end > r0]
                voice = pick(est, (b - a) / SR, t, others)
                s0, s1 = int(r0 * SR), int(r1 * SR)           # replace only the overlapped stretch ...
                seg = voice[s0 - a: s1 - a]
                w = np.ones(len(seg), np.float32)             # ... with 20 ms cross-fades at both edges
                k = min(fade, len(seg) // 2)
                if k:
                    w[:k] = np.linspace(0, 1, k, dtype=np.float32)
                    w[-k:] = np.linspace(1, 0, k, dtype=np.float32)
                track[s0:s1] = w * seg + (1 - w) * track[s0:s1]
            units.append((t, track, "separated"))
        return units

    def _centroids(self, wav, turns, overlaps, diar) -> dict[str, np.ndarray]:
        """Mean voice fingerprint per speaker from their *non-overlapped* speech."""
        if diar.centroids:
            return {k: v / (np.linalg.norm(v) + 1e-9) for k, v in diar.centroids.items()}
        out = {}
        for spk in {t.speaker for t in turns}:
            clean = []
            for t in turns:
                if t.speaker != spk:
                    continue
                cur = [(t.start, t.end)]
                for s, e in overlaps:
                    cur = [piece for a, b in cur for piece in ((a, min(b, s)), (max(a, e), b)) if piece[1] - piece[0] > 0.5]
                clean += cur
            clean = sorted(clean, key=lambda p: p[0] - p[1])[:10]
            if clean:
                v = embed(wav, clean, self.device).mean(0)
                out[spk] = v / (np.linalg.norm(v) + 1e-9)
        return out

    # ------------------------------------------------------------------ ASR
    def _transcribe(self, units: list[tuple[Turn, np.ndarray, str]]) -> list[Line]:
        pieces, owners = [], []
        for ui, (t, audio, _) in enumerate(units):
            for s, e in split_long(t.start, t.end, audio, self.max_segment_s):
                pieces.append(audio[int(s * SR): int(e * SR)])
                owners.append(ui)
        texts = self.asr.transcribe(pieces) if pieces else []
        joined: dict[int, list[str]] = {}
        for ui, txt in zip(owners, texts):
            joined.setdefault(ui, []).append(txt.strip())
        lines = []
        for ui, (t, _, src) in enumerate(units):
            text = " ".join(x for x in joined.get(ui, []) if x)
            lines.append(Line(t.speaker, round(t.start, 2), round(t.end, 2), text, romanise(text), src))
        return lines
