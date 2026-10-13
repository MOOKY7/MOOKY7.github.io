---
{
  "title": "What should a model learn to predict?",
  "date": "2026-10-12",
  "slug": "predicting-representations",
  "track": "architecture",
  "kind": "Reading note",
  "summary": "A statistical lens on representation prediction, masking, and the questions I would test before applying these ideas to genomic data.",
  "tags": ["JEPA", "Representations", "Evaluation"],
  "published": true
}
---

The target of a learning problem is a modeling decision. When I read an architecture paper, I first ask what the objective requires the model to preserve, and what it permits the model to ignore.

## The idea I am studying

I-JEPA predicts representations of hidden image regions from a visible context. Its target is an embedding rather than a reconstruction of the original pixels. The paper also emphasizes the choice of context and target masks. That choice helps define which relationships the model has a chance to learn. [Read the original paper](https://arxiv.org/abs/2301.08243).

That is the architectural idea. My interest is the statistical question it raises. If a representation is good at predicting another representation, under what conditions does it retain the information that a later task needs?

## The question from genomics

In heterogeneous cancer data, a missing measurement may reflect a panel design. A model can therefore find patterns in what was measured as well as in the measurements themselves.

My hypothesis is that the masking scheme should be treated as part of the scientific model. Artificially hiding random entries and encountering a panel that never measures those entries are different evaluation settings. Success in the first does not establish success in the second.

This is a proposed connection to my research, not a claim made by the image paper or a completed genomic experiment.

## A test I would trust

I would start with three separate questions.

1. Does the representation retain information for the target task?
2. How much does it reveal about the measurement setting?
3. What happens when the missingness pattern changes?

The first comparison should hold the patient partitions and downstream evaluation procedure fixed. A second comparison should vary the masking policy while keeping the training budget comparable. A final stress test should reserve a measurement setting during training and evaluate it separately.

A useful result would include failures. If the method improves pooled accuracy but harms one center, the center specific result belongs beside the pooled result.

## What I want to build next

A small implementation that exposes the masking policy, prediction target, and evaluation split as separate choices. The first deliverable should be a reproducible experiment with clear controls. Scale can come after the experiment answers a precise question.

## Source

Assran and colleagues. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), 2023. The genomics connection and proposed tests above are my interpretation.
