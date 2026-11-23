"""Aspect-based sentiment analysis on SemEval-2014 Task 4."""

# Project background (design notes).
#
# What ABSA is: normal (document-level) sentiment analysis gives one label to a whole text.
# Aspect-based sentiment analysis (ABSA) gives a sentiment to each aspect the text mentions.
# In "The battery is weak but the screen is excellent" the document is mixed, but ABSA
# outputs battery -> negative and screen -> positive, which is more useful for a product team.
#
# Sub-tasks: (1) aspect extraction: find the aspect terms in a review. (2) Given-aspect
# sentiment classification: label the sentiment of an aspect that is already known.
# (3) Combining both is end-to-end ABSA. This project solves all three: LSTM and BERT for (2),
# a BERT token classifier and a prompted LLM for (1), and extractor + BERT classifier (and the
# LLM alone) end to end.
#
# Code quality: a Python package with separated layers (data, models, extraction, evaluation,
# inference) and tests that enforce the import rules; ruff linting, strict mypy type checking,
# pytest with a 95% coverage floor on Python 3.10-3.13, and checks that block datasets,
# checkpoints and secrets from the repository. Every change goes through a pull request with CI.
#
# Limitations: single seed; data from a public mirror; conflict dropped; the LSTM uses random
# embeddings; one prompt and one 7B LLM; overlapping aspects cannot be BIO-tagged; the extractor
# is not wired into the demo.
# Next steps: several seeds on the official data, pre-trained embeddings (e.g. GloVe) for the
# LSTM, a larger or better-prompted LLM, and joint extraction-plus-sentiment in the demo.
#
# Main takeaway: pre-trained transformers fine-tuned on the task (BERT) clearly beat a
# from-scratch LSTM and a prompted 7B model, and for full ABSA extraction is the bottleneck.
__version__ = "0.1.0"
