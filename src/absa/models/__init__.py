"""Sentiment classifiers and language-model helpers.

The classifiers take (review, aspect) and return a sentiment label.

Everything except the tokenizer names (``Vocabulary`` and friends) needs
``torch``, and training/checkpointing/evaluation also needs the shared
evaluation module's ``scikit-learn`` dependency; all of it is imported
lazily (PEP 562) on first access, so importing this package for the
lightweight names never requires either to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from absa._lazy import lazy_getattr
from absa.models.llm import HttpLLMClient, LLMClient, MockLLMClient, parse_llm_json
from absa.models.llm_validation import (
    ValidatedAspect,
    ValidationReport,
    retry_llm_inference,
    validate_aspect_predictions,
)
from absa.models.lstm_tokens import (
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
    pad_or_truncate,
    tokenize,
)

if TYPE_CHECKING:
    from absa.models.bert_classifier import (
        BertConfig,
        BertEpoch,
        BertResult,
        evaluate_bert_test,
        load_bert,
        predict_bert,
        run_bert_training,
        train_bert,
    )
    from absa.models.bert_extractor import (
        ExtractorConfig,
        ExtractorEpoch,
        ExtractorResult,
        load_extractor,
        predict_spans,
        run_extractor_training,
        train_extractor,
    )
    from absa.models.lstm_attention import (
        AbsaLstmModel,
        AspectAttention,
        LstmEncoder,
        ModelConfig,
        masked_mean,
    )
    from absa.models.lstm_dataset import AspectDataset, build_embedding, collate, make_dataloader
    from absa.models.lstm_evaluation import (
        ErrorAnalysis,
        PredictionError,
        TestEvaluationReport,
        analyze_errors,
        evaluate_test_set,
        predict,
    )
    from absa.models.lstm_training import (
        EpochMetrics,
        LstmConfig,
        TrainingResult,
        load_checkpoint,
        run_training,
        save_checkpoint,
        train,
    )

__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "AbsaLstmModel",
    "AspectAttention",
    "AspectDataset",
    "BertConfig",
    "BertEpoch",
    "BertResult",
    "EpochMetrics",
    "ErrorAnalysis",
    "ExtractorConfig",
    "ExtractorEpoch",
    "ExtractorResult",
    "HttpLLMClient",
    "LLMClient",
    "LstmConfig",
    "LstmEncoder",
    "MockLLMClient",
    "ModelConfig",
    "PredictionError",
    "TestEvaluationReport",
    "TrainingResult",
    "ValidatedAspect",
    "ValidationReport",
    "Vocabulary",
    "analyze_errors",
    "build_embedding",
    "collate",
    "evaluate_bert_test",
    "evaluate_test_set",
    "load_bert",
    "load_checkpoint",
    "load_extractor",
    "make_dataloader",
    "masked_mean",
    "pad_or_truncate",
    "parse_llm_json",
    "predict",
    "predict_bert",
    "predict_spans",
    "retry_llm_inference",
    "run_bert_training",
    "run_extractor_training",
    "run_training",
    "save_checkpoint",
    "tokenize",
    "train",
    "train_bert",
    "train_extractor",
    "validate_aspect_predictions",
]

_LAZY = {
    "ExtractorConfig": "absa.models.bert_extractor",
    "ExtractorEpoch": "absa.models.bert_extractor",
    "ExtractorResult": "absa.models.bert_extractor",
    "load_extractor": "absa.models.bert_extractor",
    "predict_spans": "absa.models.bert_extractor",
    "run_extractor_training": "absa.models.bert_extractor",
    "train_extractor": "absa.models.bert_extractor",
    "BertConfig": "absa.models.bert_classifier",
    "BertEpoch": "absa.models.bert_classifier",
    "BertResult": "absa.models.bert_classifier",
    "evaluate_bert_test": "absa.models.bert_classifier",
    "load_bert": "absa.models.bert_classifier",
    "predict_bert": "absa.models.bert_classifier",
    "run_bert_training": "absa.models.bert_classifier",
    "train_bert": "absa.models.bert_classifier",
    "AbsaLstmModel": "absa.models.lstm_attention",
    "AspectAttention": "absa.models.lstm_attention",
    "AspectDataset": "absa.models.lstm_dataset",
    "EpochMetrics": "absa.models.lstm_training",
    "ErrorAnalysis": "absa.models.lstm_evaluation",
    "LstmConfig": "absa.models.lstm_training",
    "LstmEncoder": "absa.models.lstm_attention",
    "ModelConfig": "absa.models.lstm_attention",
    "PredictionError": "absa.models.lstm_evaluation",
    "TestEvaluationReport": "absa.models.lstm_evaluation",
    "TrainingResult": "absa.models.lstm_training",
    "analyze_errors": "absa.models.lstm_evaluation",
    "build_embedding": "absa.models.lstm_dataset",
    "collate": "absa.models.lstm_dataset",
    "evaluate_test_set": "absa.models.lstm_evaluation",
    "load_checkpoint": "absa.models.lstm_training",
    "make_dataloader": "absa.models.lstm_dataset",
    "masked_mean": "absa.models.lstm_attention",
    "predict": "absa.models.lstm_evaluation",
    "run_training": "absa.models.lstm_training",
    "save_checkpoint": "absa.models.lstm_training",
    "train": "absa.models.lstm_training",
}


def __getattr__(name: str) -> Any:
    """Resolve ``_LAZY`` names by importing their submodule on first access."""
    return lazy_getattr(__name__, _LAZY, name)
