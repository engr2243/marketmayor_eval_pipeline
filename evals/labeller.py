"""
A small interactive labeller.

The brief asks for the first fifteen cases labelled blind and the rest reviewed
against a pre-label, with the blind subset reported separately so it is visible
whether the pre-labels pulled the judgement. That shape is the whole design
here:

- `--blind` hides any existing labels and records the result as `blind_human`.
- Without it, existing labels are shown for accept-or-edit and the result is
  recorded as `model_assisted_human`.

Every saved case carries who labelled it, when, under which guide version, and
by which method. Those four fields are what make "how were the labels created"
answerable from the data rather than from memory.

Deliberately terminal-only and deliberately small. A labelling UI is not the
deliverable, and an afternoon spent on one is an afternoon not spent labelling.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from evals.datasets import load, write
from evals.models import GoldCase, GoldFact, LabelMethod, LabelProvenance

# Bumped whenever the labelling guide changes a rule. Every label records the
# version it was made under, so a guide change does not silently reinterpret
# labels made before it.
GUIDE_VERSION = "1.1"

_HELP = """
  a / enter   accept this fact as it stands
  e           edit it field by field
  d           drop it (not a real fact, or not quotable from one message)
  q           mark it unquotable but real (the verifier is right to drop it)
  n           add a new fact this case is missing
  s           skip the whole case, leaving its labels untouched
  x           save and stop
  ?           this help
"""


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\n  Stopped; nothing further saved.") from None
    return answer or default


def _show_case(case: GoldCase, transcript: str) -> None:
    name = " ".join(p for p in (case.contact.first_name, case.contact.last_name) if p)
    print()
    print("=" * 78)
    print(f"  {case.case_id}   [{case.stratum}]   today {case.today.isoformat()}")
    if case.intent:
        print(f"  intent: {case.intent}")
    print("-" * 78)
    print(f"  {name} · stage {case.contact.stage or 'unknown'}")
    if case.contact.tags:
        print(f"  tags: {', '.join(case.contact.tags)}")
    print("-" * 78)
    for line in transcript.splitlines():
        print(f"  {line}")
    print("=" * 78)


def _show_fact(index: int, total: int, fact: GoldFact) -> None:
    print()
    print(f"  fact {index + 1} of {total}: {fact.kind}  [{fact.source_label}]")
    print(f"    quote     {fact.quote!r}")
    for name in (
        "due_on",
        "event_on",
        "person_name",
        "relationship_kind",
        "engagement_kind",
        "detail_field",
        "detail_value",
        "intent",
    ):
        value = getattr(fact, name, None)
        if value is not None:
            print(f"    {name:<9} {value}")
    print(f"    quotable  {fact.quotable}   essential {fact.essential}")


def _edit_fact(fact: GoldFact) -> GoldFact | None:
    """
    Field-by-field edit. Blank keeps the current value; `-` clears it.

    Returns None if the edit produced something the schema rejects, after
    saying why — a bad label is worse than no label, so it is refused rather
    than coerced.
    """
    data = fact.model_dump()
    print("  (blank keeps the value, '-' clears it)")
    for name in (
        "kind",
        "source_label",
        "quote",
        "due_on",
        "event_on",
        "person_name",
        "relationship_kind",
        "engagement_kind",
        "detail_field",
        "detail_value",
        "intent",
    ):
        current = data.get(name)
        shown = "" if current is None else str(current)
        answer = _ask(name, shown)
        data[name] = None if answer == "-" else (answer or None)

    data["quotable"] = _ask("quotable (y/n)", "y" if fact.quotable else "n").lower() != "n"
    data["essential"] = _ask("essential (y/n)", "y" if fact.essential else "n").lower() != "n"

    try:
        return GoldFact.model_validate(data)
    except Exception as exc:
        print(f"  Not a valid fact, so not saved: {exc}")
        return None


def _new_fact(labels: Sequence[str]) -> GoldFact | None:
    print(f"  new fact. available message labels: {', '.join(labels)}")
    blank = GoldFact(kind="preference", source_label=labels[0] if labels else "m1", quote="")
    return _edit_fact(blank)


def run_labeller(
    *,
    dataset_path: Path,
    output_path: Path | None,
    labeller: str,
    blind: bool,
    limit: int | None = None,
    case_ids: tuple[str, ...] = (),
) -> int:
    """
    Walk a dataset case by case, saving as it goes.

    Saving per case rather than at the end: a labelling session that crashes an
    hour in must not lose an hour of judgement.
    """
    if not labeller:
        labeller = _ask("Your name, recorded on every label")
    if not labeller:
        print("A labeller name is required; every label records who made it.", file=sys.stderr)
        return 2

    dataset = load(dataset_path, GoldCase)
    cases = [c for c in dataset.cases if isinstance(c, GoldCase)]
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    if limit:
        cases = cases[:limit]

    destination = output_path or dataset_path
    method: LabelMethod = "blind_human" if blind else "model_assisted_human"

    print()
    print(f"  {len(cases)} case(s) from {dataset_path}")
    print(f"  method: {method}   guide: {GUIDE_VERSION}   labeller: {labeller}")
    if blind:
        print("  BLIND: existing labels are hidden and will be replaced.")
    print(_HELP)

    labelled: dict[str, GoldCase] = {}
    stopped = False

    for case in cases:
        if stopped:
            break
        transcript = asyncio.run(_transcript_for(case))
        _show_case(case, transcript)

        facts: list[GoldFact] = [] if blind else list(case.gold_facts)
        labels = sorted(_labels_in(transcript))

        if blind:
            print("  No pre-labels. Add each fact you would expect to be extracted.")
            while True:
                choice = _ask("n = new fact, a = done with this case, s = skip, x = stop", "a")
                if choice == "n":
                    made = _new_fact(labels)
                    if made:
                        facts.append(made)
                elif choice == "s":
                    facts = list(case.gold_facts)
                    break
                elif choice == "x":
                    stopped = True
                    break
                else:
                    break
        else:
            index = 0
            while index < len(facts):
                _show_fact(index, len(facts), facts[index])
                choice = _ask("a/e/d/q/n/s/x/?", "a")
                if choice == "?":
                    print(_HELP)
                    continue
                if choice == "e":
                    edited = _edit_fact(facts[index])
                    if edited:
                        facts[index] = edited
                        index += 1
                    continue
                if choice == "d":
                    facts.pop(index)
                    continue
                if choice == "q":
                    facts[index] = facts[index].model_copy(update={"quotable": False})
                    index += 1
                    continue
                if choice == "n":
                    made = _new_fact(labels)
                    if made:
                        facts.insert(index + 1, made)
                    continue
                if choice == "s":
                    facts = list(case.gold_facts)
                    break
                if choice == "x":
                    stopped = True
                    break
                index += 1

            if not stopped:
                while _ask("add another fact? (n = yes, anything else = no)", "") == "n":
                    made = _new_fact(labels)
                    if made:
                        facts.append(made)

        state = _ask("conversation_state", case.gold_state) or case.gold_state
        labelled[case.case_id] = case.model_copy(
            update={
                "gold_facts": facts,
                "gold_state": state,
                "provenance": LabelProvenance(
                    labeller=labeller,
                    labelled_at=datetime.now(UTC),
                    guide_version=GUIDE_VERSION,
                    method=method,
                ),
            }
        )
        print(f"  saved {case.case_id}: {len(facts)} fact(s), state {state}")

    if not labelled:
        print("  Nothing labelled; dataset left untouched.")
        return 0

    # Every case from the original file, with the labelled ones replaced, so a
    # partial session never truncates the dataset.
    merged = [labelled.get(c.case_id, c) for c in dataset.cases if isinstance(c, GoldCase)]
    write(
        destination,
        merged,
        header=(
            f"Extraction cases. {len(merged)} total.\n"
            f"Last labelled by {labeller} under guide {GUIDE_VERSION} "
            f"({len(labelled)} case(s) this session, method {method})."
        ),
    )
    print()
    print(f"  wrote {len(merged)} case(s) to {destination}")
    print(f"  {len(labelled)} labelled this session as {method}")
    return 0


async def _transcript_for(case: GoldCase) -> str:
    """
    The transcript exactly as the model will see it.

    Labelling against a tidied-up version of the input is how a labeller comes
    to expect a quote the model was never shown.
    """
    from evals.targets import extraction as target

    prepared = await target.prepare(case)
    return prepared.ctx.transcript or "(no readable transcript)"


def _labels_in(transcript: str) -> set[str]:
    labels: set[str] = set()
    for line in transcript.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            labels.add(stripped[1 : stripped.index("]")])
    return labels
