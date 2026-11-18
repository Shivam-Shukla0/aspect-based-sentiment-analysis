"""Shared evaluation protocol.

All approaches must be scored on the original SemEval-2014 test split.
"""

from absa.evaluation.metrics import ClassMetrics, EvaluationResult, evaluate

__all__ = ["ClassMetrics", "EvaluationResult", "evaluate"]
