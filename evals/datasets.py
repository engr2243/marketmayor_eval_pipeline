"""
Loading and identifying a dataset.

Two guarantees, both of which exist because a metric attached to the wrong
dataset is worse than no metric:

- **Validated on load.** Every line is parsed through its Pydantic model with
  `extra="forbid"`, so a field renamed in the schema and not in the data fails
  loudly at load rather than silently scoring as absent.
- **Identified by content.** A run records the hash of the bytes it read. Two
  runs are comparable only if that hash matches, and `compare` refuses
  otherwise. A dataset edited between a baseline and a re-run is the easiest
  way to manufacture an improvement, and this makes it impossible to do by
  accident.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)


class DatasetError(RuntimeError):
    """A dataset that cannot be trusted. Always names the file and the line."""


@dataclass(frozen=True)
class Dataset:
    """A loaded, validated dataset and the identity of the bytes behind it."""

    path: Path
    content_hash: str
    cases: tuple[BaseModel, ...]

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def short_hash(self) -> str:
        return self.content_hash[:12]

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[BaseModel]:
        return iter(self.cases)

    def subset(self, case_ids: Sequence[str]) -> Dataset:
        """
        A named slice, keeping the parent's hash.

        Used for the CI smoke set and for the blind-labelled subset. The hash
        stays the parent's because the slice is defined by the ids, not by a
        new file — and a report naming both the hash and the ids is
        reproducible.
        """
        wanted = set(case_ids)
        kept = tuple(c for c in self.cases if getattr(c, "case_id", None) in wanted)
        missing = wanted - {getattr(c, "case_id", None) for c in kept}
        if missing:
            raise DatasetError(
                f"{self.path}: no such case id(s): {sorted(missing)}"
            )
        return Dataset(self.path, self.content_hash, kept)


def hash_file(path: Path) -> str:
    """
    SHA-256 of the file's bytes.

    Bytes rather than parsed content: reordering lines or reformatting
    whitespace changes which dataset this is for reproducibility purposes, even
    though it does not change the cases. Being strict here is cheap.
    """
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path | str, model: type[M]) -> Dataset:
    """
    Read a JSONL dataset, validating every line against `model`.

    Blank lines and `#` comment lines are skipped, so a dataset can carry a
    header explaining what it is. Any other malformed line is an error naming
    the line number.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"No dataset at {path}")

    cases: list[M] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: invalid JSON — {exc}") from exc
            try:
                case = model.model_validate(payload)
            except ValidationError as exc:
                raise DatasetError(
                    f"{path}:{line_number}: does not match {model.__name__} — {exc}"
                ) from exc

            case_id = getattr(case, "case_id", None)
            if isinstance(case_id, str):
                if case_id in seen_ids:
                    raise DatasetError(
                        f"{path}:{line_number}: duplicate case_id {case_id!r}. "
                        "Case ids key every report and must be unique."
                    )
                seen_ids.add(case_id)
            cases.append(case)

    if not cases:
        raise DatasetError(f"{path}: no cases found")

    return Dataset(path=path, content_hash=hash_file(path), cases=tuple(cases))


def write(path: Path | str, cases: Sequence[BaseModel], *, header: str = "") -> Path:
    """
    Write a JSONL dataset, newest schema first.

    UTF-8 and `\\n` explicitly: this runs on Windows, where the default encoding
    is cp1252 and would mangle the smart quotes the adversarial cases depend on,
    and the default newline would make the content hash differ from the same
    file written anywhere else.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if header:
            for line in header.strip().splitlines():
                handle.write(f"# {line}\n")
        for case in cases:
            handle.write(case.model_dump_json() + "\n")
    return path


def append(path: Path | str, case: BaseModel) -> Path:
    """One case appended, for the labeller to save as it goes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(case.model_dump_json() + "\n")
    return path
