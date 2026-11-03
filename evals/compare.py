"""
Comparing two runs, and refusing to when the comparison is meaningless.

Most of the value here is the refusal. A delta between two runs means something
only if the dataset, the model, the environment and the mode were the same; if
any of them moved, the number is a difference between two different experiments.
`RunMeta.comparable_with` lists every reason, and this prints all of them rather
than stopping at the first.

The second thing it does is label a delta honestly. With repeats recorded, a
movement smaller than the run-to-run spread is not an improvement, and saying
"0.04 better" about noise is the most common way an eval misleads the person who
commissioned it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.config import RUNS_DIR

# Metric, label, and whether up is good. Keyed to what `report._slice_json`
# writes, so adding a metric to the report surfaces it here too.
COMPARED: list[tuple[str, str, bool]] = [
    ("wrongful_loss", "Wrongful verification loss", False),
    ("near_miss", "Near-miss rejections", False),
    ("punctuation_only", "Punctuation-only rejections", False),
    ("cap_loss", "Cap loss", False),
    ("unsupported_surviving", "Unsupported surviving (tripwire)", False),
    ("schema_valid", "Schema valid", True),
    ("state_correct", "Conversation state correct", True),
    ("spurious_on_silence", "Facts produced on silence", False),
]


class ComparisonError(RuntimeError):
    """Two runs that cannot be compared, or one that cannot be read."""


@dataclass
class Delta:
    name: str
    before: float | None
    after: float | None
    n_before: int
    n_after: int
    higher_is_better: bool
    noise: float | None
    """The larger of the two runs' spreads, where both had repeats."""

    @property
    def change(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def verdict(self) -> str:
        change = self.change
        if change is None:
            return "not measured in both runs"
        if change == 0:
            return "unchanged"
        if self.noise is not None and abs(change) <= self.noise:
            return "inside the run-to-run spread — not an improvement either way"
        improved = (change > 0) == self.higher_is_better
        return "better" if improved else "WORSE"

    def render(self) -> str:
        before = "n/a" if self.before is None else f"{self.before:.3f}"
        after = "n/a" if self.after is None else f"{self.after:.3f}"
        change = self.change
        arrow = "" if change is None else f" ({change:+.3f})"
        return (
            f"| {self.name} | {before} (n={self.n_before}) | "
            f"{after} (n={self.n_after}){arrow} | {self.verdict} |"
        )


@dataclass
class CaseFlip:
    case_id: str
    before: str
    after: str
    direction: str


def load_run(reference: str | Path) -> dict[str, Any]:
    """
    Read a persisted run by id or by path.

    Accepts a run id so the CLI reads the way the reports are named, and a path
    so a run archived elsewhere can still be compared.
    """
    path = Path(reference)
    if path.is_dir():
        path = path / "run.json"
    elif not path.is_file():
        path = RUNS_DIR / str(reference) / "run.json"
    if not path.is_file():
        raise ComparisonError(f"No run found at {reference!r} (looked for {path})")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"{path}: not valid JSON — {exc}") from exc


def _ratio(payload: dict[str, Any], key: str) -> tuple[float | None, int]:
    entry = (payload.get("overall") or {}).get(key) or {}
    numerator = entry.get("numerator")
    denominator = entry.get("denominator") or 0
    if not denominator or numerator is None:
        return None, 0
    return float(numerator) / float(denominator), int(denominator)


def _noise(before: dict[str, Any], after: dict[str, Any], key: str) -> float | None:
    """
    The run-to-run spread to judge a delta against.

    The larger of the two runs' standard deviations. None when neither run had
    enough repeats to have one — in which case the comparison is reported
    without a noise verdict rather than with an invented one.
    """
    candidates = []
    for payload in (before, after):
        entry = (payload.get("spreads") or {}).get(key) or {}
        sd = entry.get("sd")
        if sd is not None:
            candidates.append(float(sd))
    return max(candidates) if candidates else None


def _case_signature(case: dict[str, Any]) -> str:
    """
    One case's outcome, compressed enough to compare across runs.

    Totals rather than per-kind: a case whose misses went from two to zero is
    the interesting event, and which kind they were is in the full report.
    """
    if case.get("call_failed"):
        return "call failed"
    verification = case.get("verification") or {}
    per_kind = case.get("per_kind") or {}
    tp = sum(v[0] for v in per_kind.values())
    fp = sum(v[1] for v in per_kind.values())
    fn = sum(v[2] for v in per_kind.values())
    wrongful = sum(
        1 for o in verification.get("outcomes", []) if o.get("wrongful")
    )
    return f"tp={tp} fp={fp} fn={fn} wrongful={wrongful} kept={verification.get('kept', 0)}"


def _score_of(signature: str) -> tuple[int, int, int]:
    """Parse a signature back into the parts a direction can be read from."""
    parts = dict(
        piece.split("=", 1) for piece in signature.split() if "=" in piece
    )
    return (
        int(parts.get("fn", 0)),
        int(parts.get("fp", 0)),
        int(parts.get("wrongful", 0)),
    )


def flipped_cases(before: dict[str, Any], after: dict[str, Any]) -> list[CaseFlip]:
    """
    Cases whose outcome changed, and which way.

    Where a case ran several repeats, the first occurrence is taken on both
    sides: comparing a case against itself across runs needs one representative
    outcome, and averaging signatures would produce a string that describes no
    actual run.
    """
    def first_by_case(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for case in payload.get("cases", []):
            case_id = case.get("case_id")
            if isinstance(case_id, str) and case_id not in out:
                out[case_id] = case
        return out

    left, right = first_by_case(before), first_by_case(after)
    flips: list[CaseFlip] = []

    for case_id in sorted(set(left) & set(right)):
        a, b = _case_signature(left[case_id]), _case_signature(right[case_id])
        if a == b:
            continue
        if "failed" in (a, b):
            direction = "worse" if "failed" in b else "better"
        else:
            fn_a, fp_a, wr_a = _score_of(a)
            fn_b, fp_b, wr_b = _score_of(b)
            total_a, total_b = fn_a + fp_a + wr_a, fn_b + fp_b + wr_b
            direction = (
                "better" if total_b < total_a else "worse" if total_b > total_a else "mixed"
            )
        flips.append(CaseFlip(case_id, a, b, direction))

    return flips


def compare(before_ref: str | Path, after_ref: str | Path) -> str:
    """The full comparison, as Markdown."""
    before, after = load_run(before_ref), load_run(after_ref)
    meta_a, meta_b = before.get("meta", {}), after.get("meta", {})

    lines: list[str] = []
    lines.append(f"# `{meta_a.get('run_id')}` → `{meta_b.get('run_id')}`")
    lines.append("")

    problems: list[str] = []
    for key, label in (
        ("target", "targets"),
        ("dataset_hash", "dataset contents"),
        ("model", "models"),
        ("env_label", "environments"),
        ("batch", "modes"),
    ):
        if meta_a.get(key) != meta_b.get(key):
            problems.append(f"different {label} ({meta_a.get(key)!r} vs {meta_b.get(key)!r})")

    if problems:
        lines.append("## These runs are not directly comparable")
        lines.append("")
        for problem in problems:
            lines.append(f"- {problem}")
        lines.append("")
        lines.append(
            "The deltas below are printed anyway, because seeing them is "
            "sometimes the point — but they are not evidence that a change "
            "helped. Only one thing may move between a baseline and its re-run."
        )
        lines.append("")
    else:
        lines.append(
            f"Same dataset (`{str(meta_a.get('dataset_hash', ''))[:12]}`), same "
            f"model (`{meta_a.get('model')}`), same mode. Prompt hash "
            f"`{str(meta_a.get('prompt_hash', ''))[:12]}` → "
            f"`{str(meta_b.get('prompt_hash', ''))[:12]}`."
        )
        lines.append("")

    lines.append("## Per-metric delta")
    lines.append("")
    lines.append("| Metric | Before | After | Verdict |")
    lines.append("|---|---|---|---|")

    deltas: list[Delta] = []
    for key, label, higher_better in COMPARED:
        value_a, n_a = _ratio(before, key)
        value_b, n_b = _ratio(after, key)
        delta = Delta(
            name=label,
            before=value_a,
            after=value_b,
            n_before=n_a,
            n_after=n_b,
            higher_is_better=higher_better,
            noise=_noise(before, after, key),
        )
        deltas.append(delta)
        lines.append(delta.render())

    for label, key, higher_better in (
        ("Recall, all kinds", "recall", True),
        ("Precision, all kinds", "precision", True),
    ):
        a = _pooled_prf(before, key)
        b = _pooled_prf(after, key)
        delta = Delta(
            name=label,
            before=a[0],
            after=b[0],
            n_before=a[1],
            n_after=b[1],
            higher_is_better=higher_better,
            noise=_noise(before, after, f"overall_{key}"),
        )
        deltas.append(delta)
        lines.append(delta.render())
    lines.append("")

    worse = [d for d in deltas if d.verdict == "WORSE"]
    if worse:
        lines.append("### Regressions")
        lines.append("")
        for delta in worse:
            lines.append(f"- **{delta.name}** moved the wrong way ({delta.change:+.3f}).")
        lines.append("")

    noisy = [d for d in deltas if "spread" in d.verdict]
    if noisy:
        lines.append(
            f"{len(noisy)} metric(s) moved by less than their own run-to-run "
            "spread. Those are not improvements, whatever direction they went."
        )
        lines.append("")

    flips = flipped_cases(before, after)
    lines.append("## Cases that changed")
    lines.append("")
    if not flips:
        lines.append("None. Every case produced the same outcome in both runs.")
    else:
        better = [f for f in flips if f.direction == "better"]
        got_worse = [f for f in flips if f.direction == "worse"]
        mixed = [f for f in flips if f.direction == "mixed"]
        lines.append(
            f"{len(better)} better, {len(got_worse)} worse, {len(mixed)} mixed, "
            f"of {len(flips)} changed."
        )
        lines.append("")
        lines.append("| Case | Direction | Before | After |")
        lines.append("|---|---|---|---|")
        for flip in flips:
            marker = {"better": "better", "worse": "**worse**", "mixed": "mixed"}[
                flip.direction
            ]
            lines.append(f"| `{flip.case_id}` | {marker} | {flip.before} | {flip.after} |")
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append(
        f"${float(meta_a.get('spent_usd') or 0):.4f} → "
        f"${float(meta_b.get('spent_usd') or 0):.4f} "
        f"({meta_a.get('calls')} → {meta_b.get('calls')} calls)"
    )
    lines.append("")

    return "\n".join(lines)


def _pooled_prf(payload: dict[str, Any], which: str) -> tuple[float | None, int]:
    """Recall or precision over every kind, from the persisted per-kind counts."""
    per_kind = (payload.get("overall") or {}).get("per_kind") or {}
    tp = sum(int(v.get("tp", 0)) for v in per_kind.values())
    fp = sum(int(v.get("fp", 0)) for v in per_kind.values())
    fn = sum(int(v.get("fn", 0)) for v in per_kind.values())
    denominator = tp + (fn if which == "recall" else fp)
    if denominator <= 0:
        return None, 0
    return tp / denominator, denominator
