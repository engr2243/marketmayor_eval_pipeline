# marketmayor_eval_pipeline

The evaluation pipeline for [Market Mayor](https://marketmayor.ai)'s LLM
features. Market Mayor is relationship intelligence for real-estate agents: it
reads an agent's message history with a contact and extracts dated, quoted,
attributable facts.

This exists to answer a question the product's own unit tests could not: **when
you add a guard to stop a model inventing things, how many true things does the
guard destroy?**

> **The production application is not in this repository.** Market Mayor's
> prompts, schema and extraction code are private. `demo_app/` is a
> representative stand-in — the same shape of pipeline, written generically — so
> the scoring code can be read, run and tested end to end without them.
> Everything under `evals/` is the real evaluation pipeline.

```bash
python -m evals check
```

Validates the dataset, renders every prompt, and prints what a run would cost —
without calling anything.

---

## The problem it was built for

Extraction drops any fact whose quote cannot be found verbatim in the message it
cites. That rule is cheap, deterministic and very effective: a hallucinated
commitment has no real quote to anchor to, so it cannot survive.

Precision against invented facts is therefore high **by construction**, and
measuring it again proves nothing. The number nobody had was the other side of
the same guard, split into two quantities that must never be added together:

| | |
|---|---|
| **Wrongful loss** | The model found a fact the labels call real and quotable, and the verifier discarded it anyway. |
| **Correct loss** | The model invented or misquoted something and the verifier caught it. The guard working. |

The ratio between them is the verifier's precision *as a filter*. For an agent
being shown a contact's history, a destroyed true fact costs a call that should
have been made — which is why this is the number worth reporting.

### Two findings that need no labels

Labelling time is the scarcest input in any evaluation, so it matters which
findings do not need it.

**Near-miss rejections.** A quote rejected as absent which *is* present once
typographic punctuation is folded — curly apostrophes to straight, em dashes to
hyphens. The model copied the words and not the characters. That is wrongful loss
provable from the strings alone, and the fix is one line.

```
exact            the production check passes
punctuation      fails production, passes once punctuation is folded
high_similarity  most of the quote is present in order, but not all
absent           genuinely not in that message
```

**Cap loss.** Extraction keeps only the best N facts per contact. Every fact
discarded there had already passed verification — certified as grounded, then
thrown away, sorted by confidence, so a low-confidence dated commitment loses to
a high-confidence durable preference.

---

## Where to look first

| File | Why |
|---|---|
| `evals/scorers/verification.py` | The headline. Wrongful versus correct loss, the near-miss taxonomy, cap loss. |
| `evals/scorers/fields.py` | One matching rule per field type, and the four-level quote-presence taxonomy the headline rests on. |
| `evals/metrics.py` | Why every number carries its `n`, and why an undefined ratio stays undefined. |
| `evals/agreement.py` | How much two sets of labels agree — and why it refuses to compare a file against itself. |
| `evals/targets/extraction.py` | The adapter. Calls the application's real prompt, schema, client and verifier. |
| `tests/test_verification.py` | The headline metric pinned against the application's own verifier. |

---

## Design decisions worth arguing with

**The application's functions are called, never copied.** The transcript builder,
the prompt, the schema, the extraction call and the verifier are all imported
from the application under test. A second implementation of any of them would
mean this could pass while the product failed.

**Rejection reasons are attributed without reimplementing the verifier.** The
verifier returns a rejection *histogram*, not per-fact reasons — the right design
for it, and awkward for a scorer that needs to know why one specific fact died.
Rather than duplicate the classification, each discarded fact is replayed through
the real verifier *on its own*, which leaves exactly one entry in the histogram.
That entry is the reason.

**Every number carries its sample size.** A `Ratio` is a numerator and a
denominator and only becomes a float when something asks. An undefined ratio —
nothing in the denominator — stays undefined rather than quietly becoming zero,
because "no cases measured" and "every case failed" are opposite findings that a
bare `0.0` renders identically.

**A change smaller than the run-to-run spread is not an improvement**, and
`compare` says so in those words rather than printing two numbers and leaving the
reader to squint. `--repeats N` is what makes that claim possible.

**`compare` refuses runs that are not comparable.** If the dataset, the model,
the environment or the batching mode moved between two runs, the delta is a
difference between two different experiments. It prints every reason.

**Floors are provisional until a baseline exists.** A threshold invented before
anything has been measured is a guess, and one that sits inside the run-to-run
spread fails at random. `thresholds.yaml` marks which of its values are
provisional and which are structural.

**Labels are checked against their own source text.** A mistyped gold quote is
the most dangerous bug in an evaluation: it scores the model down for missing
something that was never there, and the resulting number looks entirely
plausible. `tests/test_synthetic_dataset.py` verifies mechanically that every
fact marked quotable really is findable in the message it cites, that every fact
marked unquotable really is not, and that the dates obey the anchoring rule. It
has already caught two bad labelling rules.

---

## How the labels are made

This matters more than any individual metric, because every number derived from
the labels inherits their quality.

```bash
python -m evals annotate                                      # a stronger model proposes labels
python -m evals label --blind --limit 15 --out blind.jsonl     # a human labels a subset blind
python -m evals agreement data/synthetic/extraction.annotated.jsonl blind.jsonl
python -m evals merge extraction.annotated.jsonl blind.jsonl --out reference.jsonl
```

Three properties make this defensible rather than circular:

**Nothing grades itself.** The annotator is a different, stronger model than the
one under test.

**The annotator reads the labelling guide, not the application's prompt.**
Labelling and extracting are different tasks. Handing the annotator the
extraction prompt would make it agree with the subject by construction, which is
the subtlest way to get a flattering number.

**Agreement is measured and published.** It is symmetric — Dice over facts rather
than precision and recall, because neither side is the truth — reported per
field, and it singles out `quotable`, because the headline metric is defined in
terms of that flag. If the annotator and the human disagree about what is
quotable, the headline is softer than it looks whichever side is right.

`agreement` **skips any pair whose two labels came from the same party.** That is
not a nicety: annotating with `--limit` writes the whole dataset with only the
first N annotated, so a naive comparison pairs most cases with themselves and
reads near 1.0 while measuring nothing. This was a real bug, found because its
own output looked too good.

Every label records who made it, when, under which guide version, and by which
method (`blind_human`, `model_assisted_human`, `model_only`). Reports keep the
blind subset separate, so it stays visible whether pre-labels pulled the
judgement. See `docs/labelling-guide.md`.

---

## Commands

```bash
python -m evals run extraction --repeats 3 --batch --label baseline
```

Any run over 50 calls shows its estimate and asks first. `--batch` routes through
the Message Batches API at half price, and is the more representative path since
the product's own scheduled pass is batched. `--gate` makes a missed floor a
non-zero exit, which is what CI uses.

```bash
python -m evals compare <run_a> <run_b>
```

```bash
python -m evals synthesise
```

Regenerates the dataset from `evals/synthesise.py`. The cases live in code
because several turn on exact characters a hand-edited JSONL file would mangle,
and because they are easier to review as code.

---

## Setup

Python 3.11+.

```bash
pip install -r requirements.txt
```

```bash
export EVAL_ANTHROPIC_API_KEY=sk-...
```

Read from the environment and nowhere else; never logged, never written to a run
record. `EVAL_ANTHROPIC_API_KEY` takes precedence over `ANTHROPIC_API_KEY` so
evaluation spend is billed separately from the application's own and can never
draw against a user's budget. Every run takes a spend cap and stops when it
reaches it.

```bash
python -m ruff check .
python -m mypy .
python -m pytest tests
```

---

## A sample report

`run` writes Markdown and JSON per run. The Markdown opens with the run's
identity — model, effort, environment, dataset content hash, prompt hash, commit
— and then:

```
## Headline — verification loss

| Metric                                | Value                          |
|---------------------------------------|--------------------------------|
| Wrongful loss                         | 0.118 (n=34) · 0.118 ±0.021    |
| Correct loss                          | 0.882 (n=34)                   |
| Near miss, of quotes rejected         | 0.407 (n=27) · 0.407 ±0.038    |
| Punctuation-only, of quotes rejected  | 0.296 (n=27)                   |
| Cap loss, of facts certified          | 0.061 (n=82)                   |
| Unsupported surviving (tripwire)      | 0.000 (n=71)                   |
```

The shape is the point rather than the figures, which come from whatever it is
run against. Note what the layout forces: every value carries its `n`, the spread
sits beside the mean wherever repeats allow it, and the tripwire that should
always read zero is labelled as such rather than presented as an achievement.

---

## Scope

Covers structured fact extraction. Market Mayor's chat agent is a second target:
scoring a tool-using agent needs tool-choice, tenant-isolation and
answer-grounding scorers, and the structure here anticipates them without
pretending to include them.

A model judges only where no deterministic check exists, and such a judge has to
be calibrated against human labels and report its agreement before anything is
allowed to gate on it.

## Licence

MIT.
