---
{
  "title": "An unmeasured feature is not a negative result",
  "date": "2026-10-12",
  "slug": "missing-is-not-zero",
  "track": "architecture",
  "kind": "Research note",
  "summary": "What heterogeneous genomic data teach me about model inputs, masking, and honest evaluation.",
  "tags": ["Missing data", "Genomics", "Evaluation"],
  "published": true
}
---

A zero can mean that a measured event was absent. It can also be a value inserted because no measurement was available. Those two cases can look identical to a prediction model.

In my work with heterogeneous genomic data, I need to keep that distinction visible from cohort construction through evaluation.

## Two inputs, two meanings

Suppose a binary feature records whether a variant was detected. If the assay covers the feature and the result is negative, zero is an observed value. If the assay does not cover it, the feature is missing.

One useful representation is a pair of matrices. The value matrix stores the model input. The observation mask records which entries were measured. Filling a missing entry with zero does not remove the need to understand its mask.

## A baseline is a modeling choice

Zero filling can be a practical baseline. It gives every sample the same input dimension and makes a simple comparison possible. But it does not establish that unmeasured values are negative observations.

I want an evaluation to state which missingness information the model receives. If one model sees a mask and another does not, that difference is part of the comparison.

## What I check

I separate feature availability from event frequency. A rare event measured in nearly every sample differs from an event measured only in a small subset. I also examine availability by center and panel, because pooled summaries can hide structured gaps.

For prediction, I ask whether an improvement persists within centers and across cancer types. A gain in the pooled average can come with a different tradeoff for a smaller group.

## Why this matters for architecture

The encoder can only work with the information and assumptions supplied to it. A more complex architecture does not repair an ambiguous definition of zero.

My design question is therefore concrete. How should a representation use observed values and measurement coverage while retaining the information needed for the scientific task?

That question motivates the [SCGEF research direction](/research/scgef/). It also motivates simple ablations before more complex modeling.
