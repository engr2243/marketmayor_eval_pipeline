"""
The command line.

    python -m evals check   data/synthetic/extraction.jsonl
    python -m evals run     extraction --dataset ... --repeats 3 --batch
    python -m evals compare <run_a> <run_b>
    python -m evals label   data/synthetic/extraction.jsonl
    python -m evals report  <run_id>

`check` exists because it costs nothing: it loads and validates a dataset,
renders every prompt, and prints what a run would cost, without calling
anything. That is the right first command after editing a dataset, and it is
what CI runs on a pull request that only touches fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from evals import compare as compare_mod
from evals import report as report_mod
from evals.config import CONFIRM_ABOVE_CALLS, RunConfig, load_thresholds
from evals.datasets import DatasetError, load
from evals.models import GoldCase
from evals.runner import RunAborted, run_extraction
from evals.targets.extraction import Prepared as ExtractionPrepared

DEFAULT_EXTRACTION_DATASET = "data/synthetic/extraction.jsonl"
DEFAULT_ANNOTATED = "data/synthetic/extraction.annotated.jsonl"


def _confirm(estimate: float, calls: int, per_case_tokens: int) -> bool:
    """
    The over-fifty-calls gate from the brief.

    Prints what it knows and says which part is an assumption. Refuses by
    default on anything that is not an explicit yes, including a closed stdin —
    a run that starts because nobody was there to say no is the failure this
    exists to prevent.
    """
    print()
    print(f"  This run is {calls} call(s), over the {CONFIRM_ABOVE_CALLS}-call threshold.")
    print(f"  Measured input:  ~{per_case_tokens:,} tokens per case (from the rendered prompts)")
    print("  Assumed output:  ~900 tokens per case (an assumption, not a measurement)")
    print(f"  Estimated cost:  ${estimate:.2f}")
    print()
    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  No answer; not running.")
        return False
    return answer in ("y", "yes")


def cmd_check(args: argparse.Namespace) -> int:
    """Validate a dataset and price a run, without calling anything."""
    from evals.runner import preflight

    dataset = load(args.dataset, GoldCase)
    config = RunConfig(
        target="extraction",
        dataset_path=Path(args.dataset),
        repeats=args.repeats,
        batch=args.batch,
        model=args.model or "",
    )

    cases = [c for c in dataset.cases if isinstance(c, GoldCase)]
    prepared = asyncio.run(_prepare_all(cases))

    print(f"Dataset   {dataset.path}")
    print(f"Hash      {dataset.content_hash}")
    print(f"Cases     {len(cases)}")
    print(f"Model     {config.resolved_model()}")
    print()

    by_stratum: dict[str, int] = {}
    empty: list[str] = []
    gold_facts = 0
    unquotable = 0
    for case, prep in zip(cases, prepared, strict=True):
        by_stratum[case.stratum] = by_stratum.get(case.stratum, 0) + 1
        gold_facts += len(case.gold_facts)
        unquotable += sum(1 for f in case.gold_facts if not f.quotable)
        if not prep.has_content:
            empty.append(case.case_id)

    print("Strata    " + ", ".join(f"{k}={v}" for k, v in sorted(by_stratum.items())))
    print(f"Gold      {gold_facts} fact(s), {unquotable} marked not quotable")
    methods: dict[str, int] = {}
    for case in cases:
        methods[case.provenance.method] = methods.get(case.provenance.method, 0) + 1
    print("Labels    " + ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))
    if empty:
        print(f"Empty     {len(empty)} case(s) with no readable transcript: {', '.join(empty[:5])}")
        print("          (these are never sent, exactly as production skips them)")
    print()

    estimate, calls, per_case = preflight(dataset, config, prepared)
    print(f"A run would be {calls} call(s), ~{per_case:,} input tokens per case")
    print(f"Estimated cost ${estimate:.2f} ({'batch' if config.batch else 'live'} pricing)")
    if calls > CONFIRM_ABOVE_CALLS:
        print(f"Over the {CONFIRM_ABOVE_CALLS}-call threshold, so `run` will ask first.")
    return 0


async def _prepare_all(cases: list[GoldCase]) -> list[ExtractionPrepared]:
    """Render every case without calling anything."""
    from evals.targets import extraction as target

    return [await target.prepare(case) for case in cases]


def cmd_run(args: argparse.Namespace) -> int:
    if args.target != "extraction":
        print(
            "Only the extraction target is wired up. Scoring a tool-using chat "
            "agent needs tool-choice, tenant-isolation and answer-grounding "
            "scorers, which are a separate piece of work.",
            file=sys.stderr,
        )
        return 2

    dataset = load(args.dataset, GoldCase)
    config = RunConfig(
        target="extraction",
        dataset_path=Path(args.dataset),
        model=args.model or "",
        repeats=args.repeats,
        concurrency=args.concurrency,
        spend_cap_usd=args.spend_cap,
        batch=args.batch,
        limit=args.limit,
        case_ids=tuple(args.case or ()),
        label=args.label or "",
        assume_yes=args.yes,
    )

    try:
        result = asyncio.run(run_extraction(dataset, config, confirm=_confirm))
    except RunAborted as exc:
        print(str(exc), file=sys.stderr)
        return 3

    blind = [
        c.case_id
        for c in dataset.cases
        if isinstance(c, GoldCase) and c.provenance.method == "blind_human"
    ]
    summary = report_mod.summarise(
        result.meta, result.scores, blind_case_ids=blind, thresholds=load_thresholds()
    )
    markdown, data = report_mod.write_run(summary, result.scores)

    print(report_mod.render_markdown(summary))
    print()
    print(f"Report  {markdown}")
    print(f"Data    {data}")

    if result.failures:
        print(f"\n{len(result.failures)} call(s) failed outright:", file=sys.stderr)
        for case_id, error in result.failures[:10]:
            print(f"  {case_id}: {error}", file=sys.stderr)

    if args.gate and summary.failed_gates:
        print(
            f"\n{len(summary.failed_gates)} floor(s) not met; failing as asked.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(compare_mod.compare(args.run_a, args.run_b))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Re-print a stored run's report without re-running it."""
    payload = compare_mod.load_run(args.run)
    path = Path(payload.get("__path__", "")) if payload.get("__path__") else None
    stored = (
        path.parent / "report.md"
        if path
        else report_mod.RUNS_DIR / str(args.run) / "report.md"
    )
    if stored.is_file():
        print(stored.read_text(encoding="utf-8"))
        return 0
    print(f"No stored report for {args.run!r}", file=sys.stderr)
    return 1


def cmd_label(args: argparse.Namespace) -> int:
    from evals.labeller import run_labeller

    return run_labeller(
        dataset_path=Path(args.dataset),
        output_path=Path(args.out) if args.out else None,
        labeller=args.labeller,
        blind=args.blind,
        limit=args.limit,
        case_ids=tuple(args.case or ()),
    )


def _confirm_annotation(estimate: float, calls: int, per_case_tokens: int) -> bool:
    """The same over-fifty-calls gate, for the annotation pass."""
    print()
    print(f"  Annotating {calls} case(s) with a model, over the "
          f"{CONFIRM_ABOVE_CALLS}-call threshold.")
    print(f"  Measured input:  ~{per_case_tokens:,} tokens per case (from the rendered prompts)")
    print("  Assumed output:  ~1,200 tokens per case (an assumption, not a measurement)")
    print(f"  Estimated cost:  ${estimate:.2f}")
    print()
    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  No answer; not running.")
        return False
    return answer in ("y", "yes")


def cmd_annotate(args: argparse.Namespace) -> int:
    """
    Have a stronger model propose labels, into their own file.

    Never writes over the input: the agreement figure needs both label sets to
    still exist.
    """
    from evals.annotate import annotate, estimate, write_annotations

    dataset = load(args.dataset, GoldCase)
    cases = [c for c in dataset.cases if isinstance(c, GoldCase)]
    if args.limit:
        cases = cases[: args.limit]

    prepared = asyncio.run(_prepare_all(cases))
    cost, calls, per_case = estimate(prepared, args.model)

    destination = Path(args.out)
    if destination.resolve() == Path(args.dataset).resolve():
        print(
            "Refusing to annotate a file over itself: the agreement figure needs "
            "both label sets. Pass a different --out.",
            file=sys.stderr,
        )
        return 2

    print(f"Dataset    {dataset.path} @ {dataset.short_hash}")
    print(f"Cases      {len(cases)}")
    print(f"Annotator  {args.model}")
    print("Subject    not this model; the extraction model is what gets tested")
    print(f"Out        {destination}")
    print()

    if (
        calls > CONFIRM_ABOVE_CALLS
        and not args.yes
        and not _confirm_annotation(cost, calls, per_case)
    ):
        print("Not run.", file=sys.stderr)
        return 3

    run = asyncio.run(
        annotate(
            dataset,
            model=args.model,
            concurrency=args.concurrency,
            spend_cap_usd=args.spend_cap,
            limit=args.limit,
        )
    )
    path = write_annotations(run, destination)

    print(f"Annotated  {len(run.cases)} case(s) written to {path}")
    print(f"Facts      {run.total_facts} ({run.unquotable_facts} marked not quotable)")
    print(f"Calls      {run.calls}, ${run.spent_usd:.4f}")
    if run.unverbatim:
        print(
            f"Unverbatim {run.unverbatim} quote(s) the annotator claimed were "
            "quotable but could not copy; corrected to quotable=false"
        )
    if run.failures:
        print(f"Failed     {len(run.failures)} case(s), each keeping its previous label:")
        for outcome in run.failures[:10]:
            print(f"             {outcome.case_id}: {outcome.error}")
    if run.capped:
        print("CAPPED     the spend ceiling was reached; this pass is incomplete")
    print()
    print("Next: label a blind subset, then `python -m evals agreement`.")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    """How far two label files agree, per field."""
    from evals import agreement as agreement_mod

    a = load(args.file_a, GoldCase)
    b = load(args.file_b, GoldCase)
    report = agreement_mod.compare_labels(
        agreement_mod.load_cases(a),
        agreement_mod.load_cases(b),
        label_a=args.label_a or Path(args.file_a).stem,
        label_b=args.label_b or Path(args.file_b).stem,
    )
    print(agreement_mod.render_markdown(report))
    if report.cases_compared == 0:
        return 1
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """
    One reference dataset: human labels win over model labels.

    The merged file carries mixed provenance, which is what lets a report show
    the blind subset separately from the rest.
    """
    from evals import agreement as agreement_mod
    from evals.datasets import write

    primary = agreement_mod.load_cases(load(args.primary, GoldCase))
    override = agreement_mod.load_cases(load(args.override, GoldCase))
    merged = agreement_mod.merge(primary, override)

    methods: dict[str, int] = {}
    for case in merged:
        methods[case.provenance.method] = methods.get(case.provenance.method, 0) + 1

    path = write(
        args.out,
        merged,
        header=(
            f"Reference labels: {Path(args.primary).name} with "
            f"{Path(args.override).name} taking precedence.\n"
            "Mixed provenance by design — reports split the blind subset out."
        ),
    )
    print(f"wrote {len(merged)} case(s) to {path}")
    print("methods: " + ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Market Mayor's evaluation pipeline for its LLM call sites.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate a dataset and price a run; calls nothing")
    check.add_argument("dataset", nargs="?", default=DEFAULT_EXTRACTION_DATASET)
    check.add_argument("--repeats", type=int, default=1)
    check.add_argument("--batch", action="store_true")
    check.add_argument("--model", default="")
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="run a dataset against a target")
    run.add_argument("target", choices=["extraction", "chat"])
    run.add_argument("--dataset", default=DEFAULT_EXTRACTION_DATASET)
    run.add_argument("--model", default="", help="override the application's configured model")
    run.add_argument("--repeats", type=int, default=1, help="measure run-to-run spread")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--spend-cap", type=float, default=15.0, help="USD ceiling for this run")
    run.add_argument(
        "--batch", action="store_true", help="use the Message Batches path; half price"
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--case", action="append", help="run only these case ids; repeatable")
    run.add_argument("--label", default="", help='short name, e.g. "baseline"')
    run.add_argument("--yes", action="store_true", help="skip the over-50-calls prompt (CI only)")
    run.add_argument("--gate", action="store_true", help="exit non-zero if a floor is missed")
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="per-metric delta between two runs")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.set_defaults(func=cmd_compare)

    report = sub.add_parser("report", help="reprint a stored run's report")
    report.add_argument("run")
    report.set_defaults(func=cmd_report)

    label = sub.add_parser("label", help="label a dataset, case by case")
    label.add_argument("dataset", nargs="?", default=DEFAULT_EXTRACTION_DATASET)
    label.add_argument("--out", default=None, help="where to write; defaults beside the input")
    label.add_argument("--labeller", default="", help="recorded on every label")
    label.add_argument(
        "--blind",
        action="store_true",
        help="hide any existing labels; records the result as blind_human",
    )
    label.add_argument("--limit", type=int, default=None)
    label.add_argument("--case", action="append")
    label.set_defaults(func=cmd_label)

    annotate = sub.add_parser(
        "annotate",
        help="have a stronger model propose labels, into their own file",
    )
    annotate.add_argument("dataset", nargs="?", default=DEFAULT_EXTRACTION_DATASET)
    annotate.add_argument("--out", default=DEFAULT_ANNOTATED)
    annotate.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="the annotator. Must not be the model under test.",
    )
    annotate.add_argument("--concurrency", type=int, default=4)
    annotate.add_argument("--spend-cap", type=float, default=15.0)
    annotate.add_argument("--limit", type=int, default=None)
    annotate.add_argument("--yes", action="store_true", help="skip the over-50-calls prompt")
    annotate.set_defaults(func=cmd_annotate)

    agree = sub.add_parser("agreement", help="how far two label files agree, per field")
    agree.add_argument("file_a")
    agree.add_argument("file_b")
    agree.add_argument("--label-a", default="")
    agree.add_argument("--label-b", default="")
    agree.set_defaults(func=cmd_agreement)

    merge = sub.add_parser(
        "merge", help="one reference dataset; the override's labels win"
    )
    merge.add_argument("primary")
    merge.add_argument("override")
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge)

    synth = sub.add_parser(
        "synthesise", help="regenerate the committed synthetic dataset from code"
    )
    synth.set_defaults(func=lambda _args: __import__(
        "evals.synthesise", fromlist=["main"]
    ).main())

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        return int(result or 0)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2
    except compare_mod.ComparisonError as exc:
        print(f"Comparison error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
