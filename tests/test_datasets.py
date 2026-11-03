"""
Dataset loading, and the content hash that makes a comparison honest.

The hash is the reason `compare` can refuse: a dataset edited between a baseline
and a re-run is the easiest way to manufacture an improvement, and this makes it
impossible to do by accident.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from evals.datasets import DatasetError, append, hash_file, load, write
from evals.models import ContactSeed, GoldCase, GoldFact, LabelProvenance, MessageSeed

PROV = LabelProvenance(
    labeller="t",
    labelled_at=datetime(2026, 11, 2, tzinfo=UTC),
    guide_version="1.1",
    method="model_only",
)


def case(case_id: str = "c1", quote: str = "call me in June") -> GoldCase:
    return GoldCase(
        case_id=case_id,
        stratum="typical",
        today=date(2026, 11, 2),
        contact=ContactSeed(first_name="Nadia", last_name="Okafor"),
        messages=[
            MessageSeed(day=date(2026, 3, 4), direction="inbound", kind="sms", body=quote)
        ],
        gold_facts=[GoldFact(kind="contact_request", source_label="m1", quote=quote)],
        gold_state="dormant",
        provenance=PROV,
    )


def test_a_written_dataset_loads_back_identically(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a"), case("b")])
    loaded = load(path, GoldCase)
    assert len(loaded) == 2
    assert [c.case_id for c in loaded.cases] == ["a", "b"]  # type: ignore[attr-defined]


def test_the_hash_identifies_the_bytes(tmp_path: Path) -> None:
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    write(one, [case("a")])
    write(two, [case("a")])
    assert hash_file(one) == hash_file(two)


def test_editing_a_case_changes_the_hash(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a", quote="call me in June")])
    before = load(path, GoldCase).content_hash
    write(path, [case("a", quote="call me in July")])
    assert load(path, GoldCase).content_hash != before


def test_a_header_is_skipped_on_load(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a")], header="Synthetic cases.\nSecond line.")
    assert path.read_text(encoding="utf-8").startswith("# Synthetic cases.")
    assert len(load(path, GoldCase)) == 1


def test_a_blank_line_is_not_a_case(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(load(path, GoldCase)) == 1


def test_a_duplicate_case_id_is_refused(tmp_path: Path) -> None:
    # Case ids key every report, so a duplicate would silently overwrite a
    # result rather than produce a visibly wrong one.
    path = tmp_path / "d.jsonl"
    write(path, [case("a"), case("a")])
    with pytest.raises(DatasetError, match="duplicate case_id"):
        load(path, GoldCase)


def test_malformed_json_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(DatasetError, match=r":2:|:3:"):
        load(path, GoldCase)


def test_a_field_the_schema_does_not_know_is_refused(tmp_path: Path) -> None:
    # extra="forbid" on the models. A field renamed in the schema and not in the
    # data has to fail loudly rather than score as absent.
    path = tmp_path / "d.jsonl"
    path.write_text('{"case_id": "a", "surprise": 1}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="does not match GoldCase"):
        load(path, GoldCase)


def test_an_empty_file_is_an_error_not_an_empty_run(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no cases found"):
        load(path, GoldCase)


def test_a_missing_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="No dataset at"):
        load(tmp_path / "nothing.jsonl", GoldCase)


def test_appending_adds_one_case(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a")])
    append(path, case("b"))
    assert len(load(path, GoldCase)) == 2


def test_a_subset_keeps_the_parents_hash(tmp_path: Path) -> None:
    # A slice is defined by its ids, not by a new file, so a report naming both
    # the hash and the ids stays reproducible.
    path = tmp_path / "d.jsonl"
    write(path, [case("a"), case("b"), case("c")])
    dataset = load(path, GoldCase)
    smoke = dataset.subset(["a", "c"])
    assert len(smoke) == 2
    assert smoke.content_hash == dataset.content_hash


def test_a_subset_of_unknown_ids_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write(path, [case("a")])
    with pytest.raises(DatasetError, match="no such case id"):
        load(path, GoldCase).subset(["a", "zzz"])


def test_smart_quotes_survive_the_round_trip(tmp_path: Path) -> None:
    # The adversarial near-miss cases depend on exact characters. On Windows the
    # default encoding is cp1252 and would mangle these, which is why `write`
    # sets UTF-8 explicitly.
    quote = "she said she’d definitely refer me — twice"
    path = tmp_path / "d.jsonl"
    write(path, [case("a", quote=quote)])
    loaded = load(path, GoldCase)
    assert loaded.cases[0].gold_facts[0].quote == quote  # type: ignore[attr-defined]
