# Reusable domain metrics

Start with a metric pack, map its fields to your evaluation records, and calculate
scores locally. Packs are included for **general, finance, code, enterprise,
legal, and medical** tasks. Browse definitions, fields, and downloadable examples
in the [metric catalog](https://jev-dataops.vercel.app/metrics/).

The evaluator consumes predictions and references that you supply. It does not
generate model answers, call a judge model, execute generated code, or certify
domain correctness. External test results and human ratings are explicit inputs.
The legal and medical packs address literature/research evaluation, not decisions
about individual cases or patients. All bundled examples are synthetic.

## Run a pack in three commands

Install this repository with `pip install -e .`, then:

```bash
jev-dataops list-metrics

jev-dataops init-metrics \
  --domain finance \
  --output ./finance-metrics

jev-dataops evaluate-metrics \
  --input ./finance-metrics/example.jsonl \
  --pack ./finance-metrics/metrics.json \
  --output ./finance-results
```

Both output directories must be new. The starter directory contains a complete
`metrics.json` configuration and `example.jsonl` input. Replace the example with
your own held-out predictions; do not present fixture scores as model performance.
To use the packaged definition without copying it first:

```bash
jev-dataops evaluate-metrics \
  --input ./predictions.jsonl \
  --domain enterprise \
  --output ./enterprise-results
```

No training extras, GPU, API key, or network access are required. Input is UTF-8
JSONL: one object per line, at most 1 MiB per line. Nested field paths work with
JSON objects; CSV input and array-index paths are not supported by this command.
Duplicate object keys and nonstandard `NaN`/`Infinity` constants are rejected
in input files, packs, and the `json_valid` syntax check.

## What a pack defines

| Field | Purpose |
| --- | --- |
| `schema_version` | Configuration format version; currently `1` |
| `domain`, `name`, `version`, `description` | Identity and intended scope of this metric pack |
| `field_schema` | Dotted input paths, types, and descriptions |
| `metrics[].id` | Stable identifier used in score files and reports |
| `name`, `description` | What the metric measures and how to interpret it |
| `operator` | An allowlisted local calculation |
| `fields` | Map calculation roles to declared input paths |
| `parameters` | Operator settings, such as numeric tolerances |
| `unit`, `direction`, `aggregation` | All current operators produce a 0–1 fraction, higher is better, aggregated by mean |
| `evidence` | `computed`, `external`, or `human`: where the supplied measurement comes from |
| `limitations` | What the metric cannot establish |
| `target` | Optional user-defined minimum mean and minimum coverage; bundled targets are `null` |

For example, `"prediction": "prediction.amount"` means: read the `amount` key
inside the row's `prediction` object. To adapt a pack to your existing schema,
change this mapping and its matching `field_schema` entry. You do not need to
rename every column in your source system.

Supported field types are `string`, `number`, `integer`, `boolean`, and
`string_array`. Numbers must be finite; a boolean is not an integer, and a
numeric string is not a number. An absent required path is **missing**. An
explicit `null` or wrong type is **invalid**. Optional provenance fields can be
documented without being used in a calculation; their presence is not evidence
that the evaluator has verified their authenticity.

## Calculations and interpretation

Each metric first produces a score per record, then averages the **evaluated**
scores. This is a macro average: each evaluated record has equal weight.

| Operator | Definition | Boundary |
| --- | --- | --- |
| `exact_match` | 1 when prediction and reference match after Unicode NFC normalization and collapsing whitespace; otherwise 0 | Case and punctuation matter; no semantic matching |
| `token_f1` | F1 from the multiset overlap of whitespace-delimited prediction/reference tokens | Lexical overlap only; languages without whitespace word boundaries need a different tokenizer/metric |
| `numeric_tolerance` | 1 when `abs(prediction - reference) <= max(absolute_tolerance, relative_tolerance * abs(reference))` | When unit fields are configured, unit mismatch scores 0; no unit/currency conversion or number extraction |
| `set_precision` | Size of predicted/reference ID intersection divided by predicted ID count | Measures ID agreement, not whether a cited source supports a claim |
| `set_recall` | Size of predicted/reference ID intersection divided by reference ID count | Requires a curated reference ID set; not evidence completeness in the real world |
| `ratio` | Supplied nonnegative integer numerator divided by denominator | Mean per-record fractions, not a pooled count ratio and not HumanEval pass@k |
| `boolean` | Supplied `true` becomes 1; `false` becomes 0 | The check that produced the flag runs outside this evaluator |
| `rubric_score` | `(value - min) / (max - min)`, using an integer rating inside the configured bounds | Reviewer instructions and rating anchors must be calibrated for the task |
| `json_valid` | 1 when the supplied string parses as strict JSON and matches the requested top-level type | Syntax/type check only, not JSON Schema conformance or factual correctness |

The text operators require a nonempty reference. Empty predictions score zero.
Set inputs contain nonempty string IDs; duplicate IDs count once. Set precision
is not applicable when both sets are empty; it is zero for an empty prediction
with a nonempty reference. Set recall is not applicable when its reference set
is empty. A ratio with denominator zero is not applicable. Out-of-range ratings,
negative counts, numerator greater than denominator, and non-finite numbers are
invalid measurements; they are not clamped into a valid score.

For code tasks, run tests in your own isolated execution environment and provide
their counts. A high partial-test fraction can still hide a failing critical
test. For human metrics, use the supplied 0–4 anchors as a starting point, train
reviewers on shared examples, and retain reviewer/rubric identifiers. Missing
human annotations remain visible in coverage; the evaluator never invents them.

## Read scores together with coverage

An evaluation creates:

| Artifact | Contents |
| --- | --- |
| `report.json` | Pack/input identity, per-metric means, counts, coverage, and optional target results |
| `scores.jsonl` | Line/record identifiers, per-metric status, score, and bounded reason descriptions |
| `metrics.json` | Snapshot of the definition used for this run |

Raw prediction/reference text is not copied into the score file. Keep the input
under your own access controls, together with the reported checksums.

Every metric reports `evaluated`, `missing`, `invalid`, and `not_applicable`
counts. **Coverage = evaluated / (total rows − not-applicable rows).** Missing
and invalid values stay in the coverage denominator. When the denominator is
zero, coverage is undefined (`null`); when no rows are evaluated, the mean is
also `null`.

For example, if 10 records contain only two usable numerical predictions and
both are correct, the evaluated mean is 1.0 but coverage is 0.2. That is not
evidence of 100% accuracy on the dataset. Always report both figures and inspect
the rows without scores. Malformed JSONL aborts the run without publishing a
partial output directory, rather than quietly dropping records.

There is no automatic overall quality score or deployment approval. Metrics
have different evidence requirements, and human annotations should not be
combined with deterministic checks without an explicit evaluation design.

## Customize fields, tolerances, and targets

Copy the closest pack with `init-metrics`, then edit its JSON:

1. Narrow the task in `description` and record a new `version`.
2. Choose the metrics relevant to your task; remove unrelated ones.
3. Map their fields to your actual predictions, frozen references, test results,
   and review records. Keep field definitions accurate.
4. Adjust tolerances and reviewer anchors using a validation set.
5. Set targets only after calibration; retain a separate final test set.

For a chosen metric, an optional target can look like this:

```json
"target": {
  "minimum": 0.9,
  "minimum_coverage": 1.0
}
```

These are illustrative operator-chosen values, not industry standards. A target
requires enough coverage as well as a sufficient mean. With no evaluated rows,
there is no pass. Targets are evaluated independently; the CLI's successful exit
means the evaluation completed, not that a model met every target.

The evaluator validates pack structure, field types, operator names, and
parameters before writing results. Configurations are declarative JSON; they
cannot contain Python functions, shell commands, downloaded plugins, or arbitrary
expressions. A new calculation requires an implementation and tests in
[`domain_metrics.py`](../jev_dataops/domain_metrics.py).

## Connect this to the screening → training → evaluation workflow

Screening rubrics decide which **training records** to keep or review. Domain
metric packs evaluate supplied **task predictions and observations**. Changing
a metric pack does not change JEV's screening rubric or train a model.

After SFT or verl training, generate predictions with the base model and the
trained model on the same external held-out task set. Join each output to the
same reference by a stable record ID, retain model/data versions, and run the
same pack separately on the two JSONL files. Compare scores **and coverage**,
using validation records for tuning and reserving a final test set. Subtask or
source comparisons can be made by evaluating separate, explicitly versioned
input subsets; the command does not infer these groups.

These packs do not automatically become RL rewards. The shipped verl launcher
continues to use its verified-answer exact-match reward. A domain-specific
reward needs an online scoring implementation, validation against reward
exploitation, and a clear relationship to the task benchmark.

Metric names and references describe calculation conventions, not endorsement
of these starter packs. See the [domain catalog](METRIC_CATALOG.md) for field
lists and sources, and the [domain integration guide](DOMAIN_GUIDE.md) for
screening and data preparation.
