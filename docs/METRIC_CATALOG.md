# Domain metric catalog

The six packs in [`jev_dataops/metric_packs`](../jev_dataops/metric_packs/) describe
reusable checks over **provided predictions and independent evaluation evidence**.
They do not call a judge model, run submitted code, retrieve source documents,
or produce legal, financial, or clinical judgments. Each metric reports its own
mean and coverage; the pack does not define an overall model-quality score.

| Pack | Included metrics | Evidence needed beyond predictions |
| --- | --- | --- |
| [General](../jev_dataops/metric_packs/general.json) | Answer exact match; token F1; JSON object syntax; imported output-schema pass rate; human support | Reference answers, external schema-validator flags, independent support ratings |
| [Finance](../jev_dataops/metric_packs/finance.json) | Numeric value and unit accuracy; reporting-period accuracy; evidence-ID recall; human support | Verified numeric values, units, periods, required source IDs, independent support ratings |
| [Code](../jev_dataops/metric_packs/code.json) | Independent test pass fraction; JSON envelope syntax; imported requirement checks; human maintainability | Executed test counts from an independent runner, requirement-check flags, independent maintenance ratings |
| [Enterprise](../jev_dataops/metric_packs/enterprise.json) | Source-ID precision and recall; answerability accuracy; human groundedness | Curated source sets and answerability labels for a fixed corpus snapshot, independent groundedness ratings |
| [Legal](../jev_dataops/metric_packs/legal.json) | Jurisdiction accuracy; citation-ID recall; imported applicability review; human scope | Curated research labels and citation sets, qualified independent applicability and scope annotations |
| [Medical](../jev_dataops/metric_packs/medical.json) | Study-design accuracy; evidence-ID recall; evidence-label accuracy; human support | Curated research labels and evidence sets, qualified independent evidence-support annotations |

## What the numbers mean

All metrics use a 0-1 scale with higher values indicating more agreement or
favorable supplied evidence. Aggregation is the **mean of evaluated row scores**,
not a pooled count across all rows. For example, test fractions of 8/8 and 1/4
produce `(1 + 0.25) / 2 = 0.625`, not `9/12`. This is not HumanEval pass@k.

Every `target` is `null`. Choose acceptance thresholds and minimum coverage only
after checking representative held-out examples and the cost of each error.
Thresholds, numeric tolerances, and label vocabularies are task choices. These
metrics are not automatically attached to screening gates or training rewards.

The `evidence` field distinguishes three kinds of input:

- **computed:** the scorer calculates a deterministic comparison or syntax check
  from the supplied fields. The correctness of reference labels is still external.
- **external:** the scorer aggregates a supplied validator flag or test count.
  It does not reproduce or verify the external check.
- **human:** the scorer aggregates supplied independent review annotations.
  It does not perform the review or authenticate the reviewer.

### Operator definitions

| Operator | Per-row score | Important boundary |
| --- | --- | --- |
| `exact_match` | 1 for equal NFC/whitespace-normalized strings, otherwise 0 | Case and punctuation remain significant; paraphrases can score 0. |
| `token_f1` | Multiset precision/recall F1 over whitespace-separated normalized tokens | Lexical overlap is not factual or semantic correctness; tokenization is not language-aware. |
| `numeric_tolerance` | 1 when units match, if supplied, and absolute error is at most `max(absolute_tolerance, relative_tolerance × abs(reference))` | No unit, currency, percentage, or scale conversion. Finance starts with absolute tolerance 0.01 and relative tolerance 0; revise for your task. |
| `set_precision` | Size of predicted/reference intersection divided by predicted set size | IDs are deduplicated and matched exactly. Empty predictions with nonempty references score 0; two empty sets are not applicable. |
| `set_recall` | Size of predicted/reference intersection divided by reference set size | An empty reference set is not applicable. Extra irrelevant citations do not reduce recall. |
| `ratio` | Nonnegative integer numerator divided by denominator | Numerator cannot exceed denominator. Zero denominator is not applicable. |
| `boolean` | `true` maps to 1; `false` maps to 0 | Accepts actual booleans, not strings such as `"true"` or numeric flags. |
| `rubric_score` | Integer review rating divided by 4 for these packs | Only 0, 1, 2, 3, 4 are valid. The scorer does not assign the rating. |
| `json_valid` | 1 for a strict JSON object string, otherwise 0 | Parses the object envelope only; this is not application-schema or code validation. |

Strings, numbers, integers, booleans, and arrays must have the declared types.
Do not encode missing evidence as zero, an empty reference set, or a fabricated
review. Missing and invalid required fields are excluded from a metric's mean
and appear in its coverage accounting. Defined not-applicable cases are excluded
from the eligible denominator: inspect those counts alongside coverage.

## Human-review anchors

Each human score's complete 0-4 anchors live in its `field_schema` description,
so an exported pack carries the review instructions with it. Reviewers should
inspect the actual output, source evidence, and task specification before
assigning a score. Keep adjudication policy and reviewer provenance with the
dataset; these starter anchors do not establish reviewer agreement on their own.

| Pack / field | What the reviewer examines |
| --- | --- |
| General / `human_review.support_score` | Reference support, completeness, and task scope |
| Finance / `human_review.support_score` | Evidence for the result, assumptions, units, and reporting period |
| Code / `human_review.maintainability_score` | Readability, structure, and documented project conventions |
| Enterprise / `human_review.groundedness_score` | Support for answer claims in the authorized corpus snapshot |
| Legal / `human_review.scope_score` | Annotated applicability conditions, exceptions, and uncertainty |
| Medical / `human_review.support_score` | Support for research findings and preservation of study limitations |

Legal applicability also uses a separately supplied boolean review annotation.
Neither legal metric makes a legal determination. Medical metrics evaluate
annotated research records; they do not establish clinical safety or support
patient diagnosis or treatment decisions.

Optional traceability fields include `metadata.dataset_version`,
`metadata.model_version`, `human_review.reviewer_id`, and
`human_review.rubric_version`. Code and general packs also suggest runner or
validator versions. **Unused fields are documentation only:** the engine does
not validate their presence, authenticity, or types unless a metric references
them. Archive the actual evidence and evaluation configuration separately.

## Synthetic examples and expected behavior

Each pack has a corresponding three-line file in
[`jev_dataops/metric_examples`](../jev_dataops/metric_examples/). The lines are
explicitly labeled `favorable`, `unfavorable`, and `missing_evidence`. Predictions,
external-check results, and reviewer annotations are fictional; no model, code
runner, validator, or human reviewer was used to generate a measured result.

With the shipped files, each metric evaluates two rows and reports one missing
row: coverage is **2/3**, not 100%. Most metric means are **0.5**. Two deliberate
exceptions make interpretation visible:

- General token F1 is **0.875**: changing "blue" to "red" leaves substantial
  word overlap even though exact match fails. High overlap can hide a wrong fact.
- Code test fraction is **0.625**, because the unfavorable synthetic row still
  supplies a partial pass count of 1/4. Those numbers are illustrative annotations,
  not evidence that code was executed.

Finance's unfavorable row uses the right numeric value with the wrong unit and
period. A numeric-value match alone would hide those errors. Legal and medical
examples use fictional jurisdictions, authorities, and study IDs rather than
real-world advice or patient records.

## Primary references and comparability

These references provide context for task and evidence design. The packs are
JEV DataOps templates, **not implementations of the referenced benchmark scorers**,
and their results are not comparable to those benchmarks' published scores.

- General: [SQuAD](https://arxiv.org/abs/1606.05250) is a reference-based answer
  evaluation benchmark; this pack uses its own explicitly stated normalization.
- Finance: [FinQA](https://github.com/czyssrs/FinQA) connects financial questions,
  numerical answers, and supporting facts. This pack does not execute its programs.
- Code: [HumanEval](https://github.com/openai/human-eval) evaluates generated
  programs; this pack only reads independently supplied test counts.
- Enterprise: [ALCE](https://github.com/princeton-nlp/ALCE) studies generated
  answers with citations. ID overlap here is not ALCE's citation evaluation.
- Legal: [LegalBench](https://github.com/HazyResearch/legalbench) provides legal
  reasoning tasks; this pack's fictional fixtures are separate from those tasks.
- Medical: [PubMedQA](https://github.com/pubmedqa/pubmedqa) provides biomedical
  research question-answering data; this pack uses its own research-label fields.
- JSON: [RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) specifies the JSON data
  format. Application-schema validity requires a separate validator.
