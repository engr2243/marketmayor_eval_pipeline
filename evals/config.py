"""
What a run was: its inputs, and the facts recorded about it afterwards.

Reproducibility is the whole job of this module. A metric with no record of the
prompt, the model, the dataset and the commit behind it cannot be compared
against anything, and comparing it anyway is how a prompt change gets credited
with a dataset edit.

Two of these matter more than they look:

- **`env_label`.** Where a model differs between environments — a cheaper one
  in development, say — a run that does not record which configuration it used
  is not interpretable, and two runs from different environments are not
  comparable however similar they look.
- **`prompt_hash`.** The prompts are inline constants with no version field, so
  the only way to know whether a prompt moved between two runs is to hash it.
  Hashing the schema alongside it catches a change to what the model is allowed
  to return, which has the same effect on the output and would otherwise be
  invisible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from evals.env import REPO_ROOT

Target = Literal["extraction", "chat"]

RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
THRESHOLDS_PATH = REPO_ROOT / "thresholds.yaml"

# Above this, the operator is shown the estimate and asked first. From the
# brief; the pipeline never crosses it unattended.
CONFIRM_ABOVE_CALLS = 50


@dataclass
class RunConfig:
    """The knobs for one run."""

    target: Target
    dataset_path: Path

    model: str = ""
    """Resolved from the application's settings when left empty."""

    repeats: int = 1
    concurrency: int = 4
    spend_cap_usd: float = 15.0
    batch: bool = False
    """Route extraction through the Message Batches helpers. Half price."""

    judge_model: str = "claude-opus-5"
    use_judge: bool = False

    env_label: str = ""
    """Which environment's model configuration this used. Resolved if empty."""

    limit: int | None = None
    case_ids: tuple[str, ...] = ()
    label: str = ""
    """A short human name for the run, e.g. "baseline" or "folded-quotes"."""

    assume_yes: bool = False
    """Skip the over-fifty-calls confirmation. For CI, never for a real run."""

    def resolved_model(self) -> str:
        """
        The model this run will actually use.

        Read from the application's own constant rather than restated here, so
        a model change in the application is reflected without an edit. An
        explicit `--model` overrides it, which is how a model comparison is run.

        Against a real application this usually reads a settings object instead,
        and it matters that it does: if the two differ, every number is attached
        to the wrong model id.
        """
        if self.model:
            return self.model

        from demo_app import adapter as ai_adapter

        return ai_adapter.EXTRACT_MODEL

    def resolved_env_label(self) -> str:
        if self.env_label:
            return self.env_label
        return os.environ.get("APP_ENV") or os.environ.get("ENV") or "local"

    def effort(self) -> str | None:
        """
        The effort setting, where the target has one.

        None for extraction, which sets neither effort nor adaptive thinking on
        purpose: the task is mechanical and the model rejects both. Recorded on
        every run anyway, because a run that does not say what it was
        configured with cannot be compared against one that does.
        """
        return None


def prompt_hash(system: str, schema: dict[str, Any] | None = None) -> str:
    """
    A fingerprint for "the instructions and the shape of the answer".

    The schema is included because tightening an enum changes the output as
    surely as rewriting a rule does, and neither is versioned in the
    application.
    """
    digest = sha256()
    digest.update(system.encode("utf-8", "replace"))
    if schema is not None:
        digest.update(json.dumps(schema, sort_keys=True).encode("utf-8", "replace"))
    return digest.hexdigest()


@dataclass
class RunMeta:
    """What is recorded about a run so it can be compared later."""

    run_id: str
    target: Target
    label: str

    model: str
    effort: str | None
    env_label: str
    batch: bool
    repeats: int

    dataset_name: str
    dataset_hash: str
    case_count: int

    prompt_hash: str
    git_sha: str
    started_at: str
    finished_at: str = ""

    spend_cap_usd: float = 0.0
    spent_usd: float = 0.0
    calls: int = 0
    capped: bool = False
    """True when the run stopped early because the cap was reached."""

    price_table_known: bool = True
    """False when a model id was missing from the price table and the cost is a guess."""

    notes: list[str] = field(default_factory=list)

    @staticmethod
    def new_run_id(target: str, label: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"-{label}" if label else ""
        return f"{stamp}-{target}{suffix}"

    def comparable_with(self, other: RunMeta) -> tuple[bool, list[str]]:
        """
        Whether a delta between these two runs means anything.

        Returns the verdict and every reason it might not, rather than a bare
        boolean: a model change *and* a dataset change together are a different
        conversation from either alone, and `compare` prints all of them.
        """
        problems: list[str] = []
        if self.target != other.target:
            problems.append(
                f"different targets ({self.target} vs {other.target})"
            )
        if self.dataset_hash != other.dataset_hash:
            problems.append(
                f"different dataset contents ({self.dataset_hash[:12]} vs "
                f"{other.dataset_hash[:12]})"
            )
        if self.model != other.model:
            problems.append(f"different models ({self.model} vs {other.model})")
        if self.env_label != other.env_label:
            problems.append(
                f"different environments ({self.env_label} vs {other.env_label})"
            )
        if self.batch != other.batch:
            problems.append("one batched, one live")
        return (not problems), problems


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "extraction": {
        "unsupported_surviving_max": 0.0,
        "schema_valid_min": 0.98,
        "wrongful_loss_max": 0.10,
        "near_miss_max": 0.25,
        "due_on_accuracy_min": 0.85,
        "costly_recall_min": 0.70,
        "overall_recall_min": 0.60,
        "overall_precision_min": 0.80,
        "spurious_on_silence_max": 0.05,
    },
    "chat": {
        "tenant_isolation_min": 1.0,
        "tool_choice_min": 0.85,
        "tool_args_min": 0.80,
        "grounding_violation_max": 0.05,
        "ambiguity_handled_min": 0.80,
        "turn_consistency_min": 0.90,
    },
}


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    """
    Floors from `thresholds.yaml`, falling back to the defaults above.

    The defaults are provisional and say so in the plan: a floor invented
    before a baseline exists is a guess. The real values get written here from
    the measured baseline minus its run-to-run spread, so a floor can never sit
    inside the noise.

    Parsed with PyYAML when available and with a deliberately small
    `key: value` reader when it is not, because the application's runtime
    dependencies do not include PyYAML and an evaluation has no business
    adding one to them.
    """
    path = path or THRESHOLDS_PATH
    if not path.is_file():
        return {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}

    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text) or {}
    except ImportError:
        loaded = _read_flat_yaml(text)

    merged = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    for section, values in (loaded or {}).items():
        if not isinstance(values, dict):
            continue
        merged.setdefault(str(section), {})
        for key, value in values.items():
            try:
                merged[str(section)][str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return merged


def _read_flat_yaml(text: str) -> dict[str, dict[str, float]]:
    """
    Two-level `section:` / `  key: value` only.

    Enough for a thresholds file and nothing more. If this file ever needs
    anchors or lists, add PyYAML to the pipeline's own requirements rather than
    growing this.
    """
    out: dict[str, dict[str, float]] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            section = line.split(":", 1)[0].strip()
            out.setdefault(section, {})
            continue
        if ":" not in line or not section:
            continue
        key, _, value = line.strip().partition(":")
        try:
            out[section][key.strip()] = float(value.strip())
        except ValueError:
            continue
    return out
