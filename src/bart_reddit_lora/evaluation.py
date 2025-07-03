"""
Metrics computation module for sequence-to-sequence models.

This module provides a factory function to create a `compute_metrics` callable
for Hugging Face's `Trainer`. The returned function computes ROUGE-L, BLEU, and
BERTScore (F1) on decoded model predictions versus labels.
"""

import numpy as np
import evaluate
from transformers import EvalPrediction
from typing import Callable, Dict, Any, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def build_compute_metrics(
    tok: PreTrainedTokenizerBase, 
    num_process_workers: int = 2
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Create a metrics computation function for use with Hugging Face `Trainer`.

    Args:
        tokenizer: A Hugging Face tokenizer for decoding predictions/labels.
        num_process_workers: Number of worker processes for metric computation.

    Returns:
        A callable that takes an `EvalPrediction` and returns a dict with:
          - "rougeL": ROUGE-L score (%)
          - "bleu": BLEU score (%)
          - "bertscore_f1": average BERTScore F1
    """
    rouge = evaluate.load("rouge")  # longest-substring overlap
    bleu = evaluate.load("bleu")  # n-gram precision
    bertscore = evaluate.load("bertscore")  # semantic similarity

    def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute ROUGE-L, BLEU, and BERTScore given model predictions and labels.

        Args:
            eval_pred: An `EvalPrediction` with `predictions` and `label_ids`.

        Returns:
            A dict mapping metric names to rounded scores.
        """
        preds, labels = eval_pred.predictions, eval_pred.label_ids

        # handle tuple output (some models return (generated_ids, ...))
        if isinstance(preds, tuple):
            preds = preds[0]

        # decode
        decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)

        # metrics
        rouge_l = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            num_process_workers=num_process_workers,
        )["rougeL"]
        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],  # BLEU expects list-of-lists
            smooth=True,
            num_process_workers=num_process_workers,
        )["bleu"]
        bert_f1 = np.mean(
            bertscore.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                lang="en",
                num_process_workers=num_process_workers,
            )["f1"]
        )

        # round for nice logging
        return {
            "rougeL": round(rouge_l * 100, 4),
            "bleu": round(bleu_score * 100, 4),
            "bertscore_f1": round(bert_f1, 4),
        }

    return _compute_metrics
