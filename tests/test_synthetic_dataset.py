"""
The committed dataset checked against its own labels.

A mistyped gold quote does not fail loudly — it scores the model down for
missing a fact that was never quotable in the first place, and the resulting
number looks plausible. That is the most dangerous kind of bug in an eval, so it
is checked mechanically here rather than trusted to proofreading.

What this suite asserts:

- Every fact marked `quotable: true` really is findable in the message it cites,
  under the production verifier's own normalisation.
- Every fact marked `quotable: false` really is NOT findable, so the flag is
  never a way to excuse a typo.
- Every `source_label` exists in the rendered transcript.
- Dates obey the guide: `due_on` is never before its own message, `event_on`
  never after it.
- The strata are present and the dataset is large enough to report on.
"""

from __future__ import annotations

import pytest

from evals.config import DATA_DIR
from evals.datasets import load
from evals.models import GoldCase
from evals.scorers.fields import check_quote, parse_iso_date
from evals.targets import extraction as target

DATASET = DATA_DIR / "synthetic" / "extraction.jsonl"

# Mirrors the plan's intended mix. A stratum quietly dropping to zero would make
# a whole class of failure invisible.
MINIMUM_PER_STRATUM = 5


@pytest.fixture(scope="module")
def cases() -> list[GoldCase]:
    dataset = load(DATASET, GoldCase)
    return [c for c in dataset.cases if isinstance(c, GoldCase)]


async def _context_for(case: GoldCase) -> object:
    prepared = await target.prepare(case)
    return prepared.ctx


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
def test_the_dataset_is_big_enough_to_report_on(cases: list[GoldCase]) -> None:
    assert len(cases) >= 50, f"only {len(cases)} cases; every metric's n comes from this"


def test_every_stratum_is_represented(cases: list[GoldCase]) -> None:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.stratum] = counts.get(case.stratum, 0) + 1
    for stratum in ("typical", "edge", "adversarial"):
        assert counts.get(stratum, 0) >= MINIMUM_PER_STRATUM, f"{stratum}: {counts}"


def test_every_case_says_what_it_is_for(cases: list[GoldCase]) -> None:
    # The intent line is quoted in error-analysis tables and read by the
    # labeller. A case with no stated purpose cannot be reviewed.
    missing = [c.case_id for c in cases if not c.intent.strip()]
    assert not missing, f"no intent: {missing}"


def test_every_case_records_its_label_provenance(cases: list[GoldCase]) -> None:
    for case in cases:
        assert case.provenance.labeller
        assert case.provenance.guide_version
        assert case.provenance.method in (
            "blind_human",
            "model_assisted_human",
            "model_only",
        )


def test_some_facts_are_marked_unquotable(cases: list[GoldCase]) -> None:
    # Wrongful loss is measured against quotable facts only. With none marked
    # unquotable, the dataset cannot distinguish a verifier that is wrong from
    # one that is right, which is the whole measurement.
    unquotable = [f for c in cases for f in c.gold_facts if not f.quotable]
    assert unquotable, "no unquotable gold facts; the headline metric has no control group"


# --------------------------------------------------------------------------
# The labels against their own source text
# --------------------------------------------------------------------------
async def test_every_quotable_gold_fact_really_is_quotable(cases: list[GoldCase]) -> None:
    failures: list[str] = []
    for case in cases:
        ctx = await _context_for(case)
        for index, fact in enumerate(case.gold_facts):
            if not fact.quotable:
                continue
            source = ctx.texts.get(fact.source_label, "")  # type: ignore[attr-defined]
            check = check_quote(fact.quote, source)
            if not check.survives_production:
                failures.append(
                    f"{case.case_id} fact[{index}] ({fact.kind}) cites "
                    f"[{fact.source_label}] but its quote is {check.presence.value} "
                    f"there (similarity {check.similarity:.2f}): {fact.quote!r}"
                )
    assert not failures, "\n".join(failures)


async def test_every_unquotable_gold_fact_really_is_not(cases: list[GoldCase]) -> None:
    # Otherwise `quotable: false` becomes a way to hide a typo, and a fact that
    # should have counted towards wrongful loss silently stops counting.
    wrong: list[str] = []
    for case in cases:
        ctx = await _context_for(case)
        for index, fact in enumerate(case.gold_facts):
            if fact.quotable:
                continue
            source = ctx.texts.get(fact.source_label, "")  # type: ignore[attr-defined]
            if check_quote(fact.quote, source).survives_production:
                wrong.append(
                    f"{case.case_id} fact[{index}] is marked unquotable but its "
                    f"quote IS verbatim in [{fact.source_label}]: {fact.quote!r}"
                )
    assert not wrong, "\n".join(wrong)


async def test_every_source_label_exists_in_the_transcript(cases: list[GoldCase]) -> None:
    missing: list[str] = []
    for case in cases:
        ctx = await _context_for(case)
        labels = set(ctx.labels)  # type: ignore[attr-defined]
        for index, fact in enumerate(case.gold_facts):
            if fact.source_label not in labels:
                missing.append(
                    f"{case.case_id} fact[{index}] cites [{fact.source_label}], "
                    f"transcript has {sorted(labels)}"
                )
    assert not missing, "\n".join(missing)


async def test_a_case_with_no_readable_transcript_has_no_gold_facts(
    cases: list[GoldCase],
) -> None:
    # Production never sends these, so a gold fact on one could never be found.
    wrong: list[str] = []
    for case in cases:
        prepared = await target.prepare(case)
        if not prepared.has_content and case.gold_facts:
            wrong.append(f"{case.case_id}: {len(case.gold_facts)} fact(s) but nothing to read")
    assert not wrong, "\n".join(wrong)


# --------------------------------------------------------------------------
# Dates, against the guide's anchoring rule
# --------------------------------------------------------------------------
def test_no_due_date_precedes_the_message_that_states_it(cases: list[GoldCase]) -> None:
    # The verifier rejects these, so a gold label that breaks the rule is asking
    # for a fact that could never survive.
    bad: list[str] = []
    for case in cases:
        by_label = {f"m{i}": seed.day for i, seed in enumerate(_textual(case), start=1)}
        for index, fact in enumerate(case.gold_facts):
            if fact.due_on is None:
                continue
            anchor = by_label.get(fact.source_label)
            if anchor and fact.due_on < anchor:
                bad.append(
                    f"{case.case_id} fact[{index}] due {fact.due_on} precedes its "
                    f"message dated {anchor}"
                )
    assert not bad, "\n".join(bad)


def test_no_past_event_postdates_the_message_that_reports_it(cases: list[GoldCase]) -> None:
    bad: list[str] = []
    for case in cases:
        by_label = {f"m{i}": seed.day for i, seed in enumerate(_textual(case), start=1)}
        for index, fact in enumerate(case.gold_facts):
            if fact.event_on is None:
                continue
            anchor = by_label.get(fact.source_label)
            if anchor and fact.event_on > anchor:
                bad.append(
                    f"{case.case_id} fact[{index}] event {fact.event_on} postdates its "
                    f"message dated {anchor}"
                )
    assert not bad, "\n".join(bad)


def test_every_date_detail_value_parses(cases: list[GoldCase]) -> None:
    bad: list[str] = []
    for case in cases:
        for index, fact in enumerate(case.gold_facts):
            if (
                fact.detail_field in ("birthday", "home_anniversary")
                and parse_iso_date(fact.detail_value) is None
            ):
                bad.append(
                    f"{case.case_id} fact[{index}] {fact.detail_field} "
                    f"is not an ISO date: {fact.detail_value!r}"
                )
    assert not bad, "\n".join(bad)


def test_every_price_detail_has_a_digit(cases: list[GoldCase]) -> None:
    # The verifier refuses a price with no figure in it as a misread.
    bad: list[str] = []
    for case in cases:
        for index, fact in enumerate(case.gold_facts):
            if fact.detail_field == "price_range" and not any(
                ch.isdigit() for ch in fact.detail_value or ""
            ):
                bad.append(f"{case.case_id} fact[{index}]: {fact.detail_value!r}")
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------
# Field completeness, against what the verifier requires
# --------------------------------------------------------------------------
def test_a_relational_kind_that_needs_a_name_has_one_unless_it_is_the_point(
    cases: list[GoldCase],
) -> None:
    # The verifier drops a nameless relationship or referred_by. A gold
    # fact of that shape is legitimate only when the case exists to test exactly
    # that, in which case it must be marked unquotable so it is excluded from
    # wrongful loss.
    bad: list[str] = []
    for case in cases:
        for index, fact in enumerate(case.gold_facts):
            if fact.kind not in ("relationship", "referred_by"):
                continue
            if not (fact.person_name or "").strip() and fact.quotable:
                bad.append(
                    f"{case.case_id} fact[{index}] is a {fact.kind} with no "
                    "person_name but is marked quotable; the verifier will always "
                    "drop it, so it would score as wrongful loss forever"
                )
    assert not bad, "\n".join(bad)


def test_a_contact_detail_has_both_a_field_and_a_value(cases: list[GoldCase]) -> None:
    bad: list[str] = []
    for case in cases:
        for index, fact in enumerate(case.gold_facts):
            if fact.kind != "contact_detail":
                continue
            if fact.detail_field is None or not (fact.detail_value or "").strip():
                bad.append(f"{case.case_id} fact[{index}]: a detail is a field and a value")
    assert not bad, "\n".join(bad)


def _textual(case: GoldCase) -> list:
    """
    The messages the transcript will actually render, in label order.

    Mirrors the application's own filter: a kind outside its textual set, or a
    row with no words, contributes no line and so consumes no label. Sorted by
    day, which is how the builder orders them.
    """
    from demo_app.context import TEXTUAL_KINDS

    rows = [
        seed
        for seed in case.messages
        if seed.kind in TEXTUAL_KINDS and (seed.body or seed.subject)
    ]
    return sorted(rows, key=lambda s: s.day)


# --------------------------------------------------------------------------
# Conversation state, against the guide's two-way-exchange rule
# --------------------------------------------------------------------------
def test_a_single_message_case_states_the_right_thing(cases: list[GoldCase]) -> None:
    """
    One message is not a conversation, and which side sent it decides the state.

    `no_conversation` is defined as no substantive two-way exchange, so a lone
    outbound note from the agent is that by definition however long ago it was
    sent. A lone inbound message is `awaiting_my_reply`: the contact spoke and
    nobody answered.

    Written because the annotation pass exposed this as inconsistent in the
    authored labels — some lone outbound notes were `dormant` and others
    `no_conversation`, which is a coin toss masquerading as a label. Agreement
    on `conversation_state` was 0.652 largely because of it.
    """
    expected = {"inbound": "awaiting_my_reply", "outbound": "no_conversation"}
    # States that legitimately override the rule because the message itself
    # settles the matter: a deal closed, or they said no.
    terminal = {"concluded_won", "concluded_lost"}

    wrong: list[str] = []
    for case in cases:
        rows = _textual(case)
        if len(rows) != 1:
            continue
        if case.gold_state in terminal:
            continue
        want = expected[rows[0].direction]
        if case.gold_state != want:
            wrong.append(
                f"{case.case_id}: one {rows[0].direction} message, "
                f"state is {case.gold_state!r}, rule says {want!r}"
            )
    assert not wrong, "\n".join(wrong)
