"""
Turning per-case scores into something a person can act on.

Three rules the report follows, all of them from the brief and all of them
easier to state than to keep:

1. **Every number carries its n.** A `Ratio` cannot render without one, which
   is why the metrics are ratios rather than floats.
2. **A change smaller than the run-to-run spread is not an improvement**, and
   the report says that in those words rather than printing two numbers and
   leaving the reader to squint.
3. **A metric that cannot be computed says so.** No metric defaults to zero to
   fill a column. "n/a (n=0)" is a finding — usually that the dataset has no
   case exercising it.

The blind-labelled subset is summarised separately throughout. Pre-labelled
cases and blind cases can disagree, and averaging them together would hide
exactly the thing the blind subset exists to detect.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from evals.config import RUNS_DIR, RunMeta, load_thresholds
from evals.cost import Usage, usd
from evals.metrics import PRF, Ratio, Spread, percentile
from evals.scorers.extraction import CaseScore, FieldTally


@dataclass
class Gate:
    """One floor from thresholds.yaml, and whether this run cleared it."""

    name: str
    value: float | None
    floor: float
    higher_is_better: bool
    n: int

    @property
    def passed(self) -> bool | None:
        """None when the metric could not be computed — neither pass nor fail."""
        if self.value is None:
            return None
        return self.value >= self.floor if self.higher_is_better else self.value <= self.floor

    def render(self) -> str:
        verdict = {True: "pass", False: "FAIL", None: "not measured"}[self.passed]
        shown = "n/a" if self.value is None else f"{self.value:.3f}"
        comparator = ">=" if self.higher_is_better else "<="
        return f"{self.name}: {shown} (n={self.n}) {comparator} {self.floor:.3f} — {verdict}"


@dataclass
class Slice:
    """Pooled metrics over a set of case scores."""

    label: str
    cases: int = 0

    per_kind: dict[str, PRF] = field(default_factory=dict)
    fields: dict[str, FieldTally] = field(default_factory=dict)

    schema_valid: Ratio = field(default_factory=Ratio)
    state_correct: Ratio = field(default_factory=Ratio)
    call_failures: int = 0

    wrongful_loss: Ratio = field(default_factory=Ratio)
    correct_loss: Ratio = field(default_factory=Ratio)
    near_miss: Ratio = field(default_factory=Ratio)
    punctuation_only: Ratio = field(default_factory=Ratio)
    unsupported_surviving: Ratio = field(default_factory=Ratio)
    cap_loss: Ratio = field(default_factory=Ratio)
    cap_dropped_by: dict[str, int] = field(default_factory=dict)
    rejected_totals: dict[str, int] = field(default_factory=dict)

    spurious_on_silence: Ratio = field(default_factory=Ratio)

    latencies_ms: list[float] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    @property
    def overall(self) -> PRF:
        total = PRF()
        for prf in self.per_kind.values():
            total = total + prf
        return total

    @property
    def costly(self) -> PRF:
        from evals.scorers.extraction import COSTLY_KINDS

        total = PRF()
        for kind, prf in self.per_kind.items():
            if kind in COSTLY_KINDS:
                total = total + prf
        return total

    @property
    def p50_ms(self) -> float | None:
        return percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float | None:
        return percentile(self.latencies_ms, 95)

    def due_on_accuracy(self) -> Ratio:
        tally = self.fields.get("due_on")
        return tally.accuracy if tally else Ratio()


def _pool(label: str, scores: Sequence[CaseScore]) -> Slice:
    """Sum every per-case measurement into one slice."""
    out = Slice(label=label, cases=len(scores))

    per_kind: dict[str, PRF] = defaultdict(PRF)
    fields: dict[str, FieldTally] = {}
    rejected: dict[str, int] = defaultdict(int)
    cap_by: dict[str, int] = defaultdict(int)

    for score in scores:
        out.schema_valid = out.schema_valid + Ratio(int(score.schema_valid), 1)
        if score.call_failed:
            out.call_failures += 1
        if score.state_correct is not None:
            out.state_correct = out.state_correct + Ratio(int(score.state_correct), 1)
        if score.latency_ms:
            out.latencies_ms.append(score.latency_ms)
        out.usage = out.usage + Usage(
            score.input_tokens, score.output_tokens, score.cache_read_tokens
        )

        for kind, prf in score.per_kind.items():
            per_kind[kind] = per_kind[kind] + prf

        for name, tally in score.fields.items():
            merged = fields.setdefault(name, FieldTally())
            merged.correct += tally.correct
            merged.total += tally.total
            merged.verdicts.update(tally.verdicts)
            merged.offsets.extend(tally.offsets)

        if score.gold_is_silent:
            out.spurious_on_silence = out.spurious_on_silence + Ratio(
                int(score.spurious_on_silence > 0), 1
            )

        report = score.verification
        if report is None:
            continue
        out.wrongful_loss = out.wrongful_loss + report.wrongful_loss
        out.correct_loss = out.correct_loss + report.correct_loss
        out.near_miss = out.near_miss + report.near_miss
        out.punctuation_only = out.punctuation_only + report.punctuation_only_loss
        out.unsupported_surviving = out.unsupported_surviving + report.unsupported_surviving
        out.cap_loss = out.cap_loss + report.cap.rate
        for reason, count in report.rejected.items():
            rejected[reason] += count
        for (kind, confidence), count in report.cap.by_kind_confidence.items():
            cap_by[f"{kind}/{confidence}"] += count

    out.per_kind = dict(per_kind)
    out.fields = fields
    out.rejected_totals = dict(rejected)
    out.cap_dropped_by = dict(cap_by)
    return out


@dataclass
class Summary:
    """A whole run, pooled and per repeat."""

    meta: RunMeta
    overall: Slice
    blind: Slice | None
    by_stratum: dict[str, Slice]
    spreads: dict[str, Spread]
    gates: list[Gate]
    total_usd: float

    @property
    def failed_gates(self) -> list[Gate]:
        return [g for g in self.gates if g.passed is False]


# Which pooled metric each floor reads, and which direction is good.
_GATE_SPEC: list[tuple[str, str, bool]] = [
    ("unsupported_surviving_max", "unsupported_surviving", False),
    ("schema_valid_min", "schema_valid", True),
    ("wrongful_loss_max", "wrongful_loss", False),
    ("near_miss_max", "near_miss", False),
    ("due_on_accuracy_min", "due_on_accuracy", True),
    ("costly_recall_min", "costly_recall", True),
    ("overall_recall_min", "overall_recall", True),
    ("overall_precision_min", "overall_precision", True),
    ("spurious_on_silence_max", "spurious_on_silence", False),
]


def _gate_value(slice_: Slice, key: str) -> Ratio:
    if key == "due_on_accuracy":
        return slice_.due_on_accuracy()
    if key == "costly_recall":
        return slice_.costly.recall
    if key == "overall_recall":
        return slice_.overall.recall
    if key == "overall_precision":
        return slice_.overall.precision
    value = getattr(slice_, key, None)
    return value if isinstance(value, Ratio) else Ratio()


def summarise(
    meta: RunMeta,
    scores: Sequence[CaseScore],
    blind_case_ids: Iterable[str] = (),
    thresholds: dict[str, dict[str, float]] | None = None,
) -> Summary:
    """Pool a run's scores, measure its spread, and check it against the floors."""
    thresholds = thresholds or load_thresholds()
    floors = thresholds.get("extraction", {})

    overall = _pool("all cases", scores)

    blind_ids = set(blind_case_ids)
    blind = (
        _pool("blind-labelled", [s for s in scores if s.case_id in blind_ids])
        if blind_ids
        else None
    )

    by_stratum: dict[str, Slice] = {}
    for stratum in sorted({s.stratum for s in scores if s.stratum}):
        by_stratum[stratum] = _pool(stratum, [s for s in scores if s.stratum == stratum])

    # Spread across repeats. Each repeat is scored independently and the
    # headline metrics are recomputed per repeat, so the variation reported is
    # the model's and not an artefact of pooling.
    spreads: dict[str, Spread] = defaultdict(Spread)
    for repeat_scores in _group_by_repeat(scores, meta.repeats).values():
        pooled = _pool("repeat", repeat_scores)
        spreads["wrongful_loss"].add(pooled.wrongful_loss.value)
        spreads["near_miss"].add(pooled.near_miss.value)
        spreads["overall_recall"].add(pooled.overall.recall.value)
        spreads["overall_precision"].add(pooled.overall.precision.value)
        spreads["costly_recall"].add(pooled.costly.recall.value)
        spreads["due_on_accuracy"].add(pooled.due_on_accuracy().value)
        spreads["cap_loss"].add(pooled.cap_loss.value)
        spreads["schema_valid"].add(pooled.schema_valid.value)

    gates = []
    for floor_name, metric_key, higher_better in _GATE_SPEC:
        if floor_name not in floors:
            continue
        ratio = _gate_value(overall, metric_key)
        gates.append(
            Gate(
                name=metric_key,
                value=ratio.value,
                floor=floors[floor_name],
                higher_is_better=higher_better,
                n=ratio.n,
            )
        )

    total = usd(overall.usage, meta.model, batch=meta.batch)
    return Summary(
        meta=meta,
        overall=overall,
        blind=blind,
        by_stratum=by_stratum,
        spreads=dict(spreads),
        gates=gates,
        total_usd=total,
    )


def _group_by_repeat(scores: Sequence[CaseScore], repeats: int) -> dict[int, list[CaseScore]]:
    """
    Split pooled scores back into their repeats.

    Cases appear once per repeat in dispatch order, so the nth occurrence of a
    case id belongs to the nth repeat. Counting occurrences is robust to the
    out-of-order completion that bounded concurrency produces, which a simple
    slice would not be.
    """
    seen: dict[str, int] = defaultdict(int)
    grouped: dict[int, list[CaseScore]] = defaultdict(list)
    for score in scores:
        index = seen[score.case_id]
        seen[score.case_id] += 1
        grouped[min(index, max(0, repeats - 1))].append(score)
    return grouped


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _ratio_row(name: str, ratio: Ratio, spread: Spread | None = None) -> str:
    cell = ratio.render()
    if spread is not None and spread.n > 1:
        cell = f"{cell} · {spread.render()}"
    return f"| {name} | {cell} |"


def render_markdown(summary: Summary) -> str:
    meta = summary.meta
    o = summary.overall
    lines: list[str] = []

    lines.append(f"# Run `{meta.run_id}`")
    lines.append("")
    lines.append(f"**Target** {meta.target} · **label** {meta.label or '(none)'}")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{meta.model}` |")
    lines.append(f"| Effort | {meta.effort or 'not set (deliberate for extraction)'} |")
    lines.append(f"| Environment config | {meta.env_label} |")
    lines.append(f"| Mode | {'batch' if meta.batch else 'live'} |")
    lines.append(f"| Repeats | {meta.repeats} |")
    lines.append(f"| Dataset | `{meta.dataset_name}` @ `{meta.dataset_hash[:12]}` |")
    lines.append(f"| Cases | {meta.case_count} |")
    lines.append(f"| Prompt hash | `{meta.prompt_hash[:12]}` |")
    lines.append(f"| Commit | `{meta.git_sha[:12] or 'unknown'}` |")
    lines.append(f"| Started | {meta.started_at} |")
    lines.append(f"| Calls | {meta.calls} |")
    lines.append(
        f"| Spend | ${meta.spent_usd:.4f} of ${meta.spend_cap_usd:.2f} cap"
        f"{' — **CAPPED, run is truncated**' if meta.capped else ''} |"
    )
    lines.append("")

    if meta.notes:
        lines.append("## Notes on this run")
        lines.append("")
        for note in meta.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Headline — verification loss")
    lines.append("")
    lines.append(
        "Wrongful loss is the number to read: facts the model found, that the "
        "labels call real and quotable, which the verifier discarded anyway. "
        "Correct loss is the same guard working. Near-miss and punctuation-only "
        "need no labels at all."
    )
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(_ratio_row("Wrongful loss", o.wrongful_loss, summary.spreads.get("wrongful_loss")))
    lines.append(_ratio_row("Correct loss", o.correct_loss))
    lines.append(
        _ratio_row(
            "Near miss, of quotes rejected", o.near_miss, summary.spreads.get("near_miss")
        )
    )
    lines.append(_ratio_row("Punctuation-only, of quotes rejected", o.punctuation_only))
    lines.append(
        _ratio_row(
            "Cap loss, of facts certified", o.cap_loss, summary.spreads.get("cap_loss")
        )
    )
    lines.append(
        _ratio_row("Unsupported surviving (tripwire, expect 0)", o.unsupported_surviving)
    )
    lines.append("")

    if o.rejected_totals:
        lines.append("### Why the verifier discarded things")
        lines.append("")
        lines.append("| Reason | Facts |")
        lines.append("|---|---|")
        for reason, count in sorted(o.rejected_totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{reason}` | {count} |")
        lines.append("")

    if o.cap_dropped_by:
        lines.append("### What the twelve-fact ceiling dropped")
        lines.append("")
        lines.append("| Kind / confidence | Facts |")
        lines.append("|---|---|")
        for key, count in sorted(o.cap_dropped_by.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {key} | {count} |")
        lines.append("")

    lines.append("## Precision and recall, per fact kind")
    lines.append("")
    lines.append("On the validated output, which is what the agent actually sees.")
    lines.append("")
    lines.append("| Kind | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|")
    for kind in sorted(o.per_kind):
        prf = o.per_kind[kind]
        f1 = prf.f1
        lines.append(
            f"| `{kind}` | {prf.precision.render()} | {prf.recall.render()} | "
            f"{'n/a' if f1 is None else f'{f1:.3f}'} | {prf.true_positives} | "
            f"{prf.false_positives} | {prf.false_negatives} |"
        )
    total = o.overall
    f1 = total.f1
    lines.append(
        f"| **all kinds** | {total.precision.render()} | {total.recall.render()} | "
        f"{'n/a' if f1 is None else f'{f1:.3f}'} | {total.true_positives} | "
        f"{total.false_positives} | {total.false_negatives} |"
    )
    lines.append("")

    if o.fields:
        lines.append("## Field agreement")
        lines.append("")
        lines.append("| Field | Accuracy | Disagreement shapes |")
        lines.append("|---|---|---|")
        for name in sorted(o.fields):
            tally = o.fields[name]
            shapes = ", ".join(
                f"{k}×{v}" for k, v in sorted(tally.verdicts.items(), key=lambda kv: -kv[1])
            )
            row = f"| `{name}` | {tally.accuracy.render()} | {shapes} |"
            lines.append(row)
        lines.append("")
        offsets = o.fields.get("due_on")
        if offsets and offsets.offsets:
            counts: dict[int, int] = defaultdict(int)
            for value in offsets.offsets:
                counts[value] += 1
            spread = ", ".join(
                f"{'+' if d > 0 else ''}{d}d×{c}" for d, c in sorted(counts.items())
            )
            lines.append(
                f"Due-date offsets when wrong: {spread}. A one-sided "
                "distribution points at the wrong anchor date rather than at "
                "arithmetic."
            )
            lines.append("")

    lines.append("## Abstention, state and reliability")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(_ratio_row("Schema valid", o.schema_valid, summary.spreads.get("schema_valid")))
    lines.append(_ratio_row("Conversation state correct", o.state_correct))
    lines.append(
        _ratio_row("Produced facts where the gold is silent", o.spurious_on_silence)
    )
    lines.append(f"| Call failures | {o.call_failures} |")
    lines.append("")

    lines.append("## Cost and latency")
    lines.append("")
    if meta.batch:
        lines.append(
            "Batched run: the Message Batches results carry no usage, so token "
            "counts are absent rather than estimated, and latency is per batch "
            "rather than per case."
        )
        lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    p50, p95 = o.p50_ms, o.p95_ms
    lines.append(f"| Latency p50 | {'n/a' if p50 is None else f'{p50:.0f} ms'} |")
    lines.append(f"| Latency p95 | {'n/a' if p95 is None else f'{p95:.0f} ms'} |")
    lines.append(f"| Input tokens | {o.usage.input_tokens:,} |")
    lines.append(f"| Output tokens | {o.usage.output_tokens:,} |")
    # Zero here is the expected result, not a bug: extraction caching is off
    # because the system prompt is under the cacheable minimum. Saying so in the
    # report stops the next reader filing it as one.
    cache_note = (
        " — zero is expected: extraction caching is off by design"
        if o.usage.cache_read_tokens == 0
        else ""
    )
    lines.append(f"| Cache-read tokens | {o.usage.cache_read_tokens:,}{cache_note} |")
    cost_note = "" if meta.price_table_known else " (model not in price table — upper bound)"
    lines.append(f"| Cost | ${summary.total_usd:.4f}{cost_note} |")
    if o.cases:
        lines.append(f"| Cost per case | ${summary.total_usd / o.cases:.5f} |")
    lines.append("")

    if summary.blind is not None:
        lines.append("## Blind-labelled subset, separately")
        lines.append("")
        lines.append(
            "Labelled before any model annotation was seen. Reported apart from "
            "the rest so it is visible whether pre-labels pulled the judgement."
        )
        lines.append("")
        b = summary.blind
        lines.append("| Metric | Blind | All cases |")
        lines.append("|---|---|---|")
        lines.append(
            f"| Recall | {b.overall.recall.render()} | {o.overall.recall.render()} |"
        )
        lines.append(
            f"| Precision | {b.overall.precision.render()} | {o.overall.precision.render()} |"
        )
        lines.append(
            f"| Wrongful loss | {b.wrongful_loss.render()} | {o.wrongful_loss.render()} |"
        )
        lines.append("")

    if summary.by_stratum:
        lines.append("## By stratum")
        lines.append("")
        lines.append("| Stratum | Cases | Recall | Precision | Wrongful loss | Near miss |")
        lines.append("|---|---|---|---|---|---|")
        for name in sorted(summary.by_stratum):
            s = summary.by_stratum[name]
            lines.append(
                f"| {name} | {s.cases} | {s.overall.recall.render()} | "
                f"{s.overall.precision.render()} | {s.wrongful_loss.render()} | "
                f"{s.near_miss.render()} |"
            )
        lines.append("")

    lines.append("## Floors")
    lines.append("")
    if not summary.gates:
        lines.append("No floors configured.")
    else:
        for gate in summary.gates:
            lines.append(f"- {gate.render()}")
    lines.append("")
    if summary.failed_gates:
        lines.append(
            f"**{len(summary.failed_gates)} floor(s) not met.** In CI this fails the run."
        )
    lines.append("")

    if meta.repeats > 1:
        lines.append("## On reading these deltas")
        lines.append("")
        lines.append(
            f"Measured over {meta.repeats} repeats. Where a spread is shown, a "
            "difference smaller than it is not an improvement — see `compare` "
            "for an explicit verdict on any two runs."
        )
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Ratio):
        return {"numerator": value.numerator, "denominator": value.denominator}
    return value


def write_run(summary: Summary, scores: Sequence[CaseScore]) -> tuple[Path, Path]:
    """
    Persist a run: the summary as Markdown, the whole thing as JSON.

    `runs/` is gitignored — the per-case records hold the model's own
    output, and for a private dataset that output quotes the source material.
    Summaries are what gets committed, deliberately and by hand.
    """
    directory = RUNS_DIR / summary.meta.run_id
    directory.mkdir(parents=True, exist_ok=True)

    markdown = directory / "report.md"
    markdown.write_text(render_markdown(summary), encoding="utf-8", newline="\n")

    payload = {
        "meta": _jsonable(summary.meta),
        "gates": [_jsonable(g) | {"passed": g.passed} for g in summary.gates],
        "overall": _slice_json(summary.overall),
        "blind": _slice_json(summary.blind) if summary.blind else None,
        "by_stratum": {k: _slice_json(v) for k, v in summary.by_stratum.items()},
        "spreads": {
            k: {"mean": v.mean, "sd": v.sd, "lo": v.lo, "hi": v.hi, "n": v.n}
            for k, v in summary.spreads.items()
        },
        "cases": [_case_json(s) for s in scores],
    }
    data = directory / "run.json"
    data.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="\n"
    )
    return markdown, data


def _slice_json(slice_: Slice) -> dict[str, Any]:
    return {
        "label": slice_.label,
        "cases": slice_.cases,
        "per_kind": {
            k: {
                "tp": v.true_positives,
                "fp": v.false_positives,
                "fn": v.false_negatives,
                "precision": v.precision.value,
                "recall": v.recall.value,
                "f1": v.f1,
                "n_precision": v.precision.n,
                "n_recall": v.recall.n,
            }
            for k, v in slice_.per_kind.items()
        },
        "fields": {
            k: {
                "correct": v.correct,
                "total": v.total,
                "accuracy": v.accuracy.value,
                "verdicts": dict(v.verdicts),
                "offsets": v.offsets,
            }
            for k, v in slice_.fields.items()
        },
        "wrongful_loss": _jsonable(slice_.wrongful_loss),
        "correct_loss": _jsonable(slice_.correct_loss),
        "near_miss": _jsonable(slice_.near_miss),
        "punctuation_only": _jsonable(slice_.punctuation_only),
        "unsupported_surviving": _jsonable(slice_.unsupported_surviving),
        "cap_loss": _jsonable(slice_.cap_loss),
        "cap_dropped_by": slice_.cap_dropped_by,
        "rejected_totals": slice_.rejected_totals,
        "schema_valid": _jsonable(slice_.schema_valid),
        "state_correct": _jsonable(slice_.state_correct),
        "spurious_on_silence": _jsonable(slice_.spurious_on_silence),
        "call_failures": slice_.call_failures,
        "p50_ms": slice_.p50_ms,
        "p95_ms": slice_.p95_ms,
        "usage": _jsonable(slice_.usage),
    }


def _case_json(score: CaseScore) -> dict[str, Any]:
    report = score.verification
    return {
        "case_id": score.case_id,
        "stratum": score.stratum,
        "schema_valid": score.schema_valid,
        "call_failed": score.call_failed,
        "error": score.error,
        "state_correct": score.state_correct,
        "latency_ms": score.latency_ms,
        "input_tokens": score.input_tokens,
        "output_tokens": score.output_tokens,
        "cache_read_tokens": score.cache_read_tokens,
        "per_kind": {
            k: [v.true_positives, v.false_positives, v.false_negatives]
            for k, v in score.per_kind.items()
        },
        "verification": None
        if report is None
        else {
            "raw": report.raw_count,
            "kept": report.kept,
            "rejected": report.rejected,
            "cap_dropped": report.cap.dropped,
            "outcomes": [
                {
                    "index": o.index,
                    "kind": o.kind,
                    "confidence": o.confidence,
                    "source_label": o.source_label,
                    "passed": o.passed_verification,
                    "reason": o.reason,
                    "quote_presence": o.quote.presence.value,
                    "quote_similarity": round(o.quote.similarity, 4),
                    "gold_index": o.gold_index,
                    "wrongful": o.wrongful,
                }
                for o in report.outcomes
            ],
        },
    }
