"""
Module for building a ROUGE-L metric computation function
for Hugging Face Seq2SeqTrainer.
"""

import numpy as np
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, PreTrainedTokenizerBase, EvalPrediction
from typing import Callable, Dict


def build_compute_metrics(tok: PreTrainedTokenizerBase) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Create a compute_metrics function for Seq2SeqTrainer that returns the ROUGE-L score.

    Args:
        tok (PreTrainedTokenizerBase): Tokenizer for decoding predictions and labels.

    Returns:
        Callable[[EvalPrediction], Dict[str, float]]: Function computing "rougeL" percentage.
    """
    rouge = evaluate.load("rouge", keep_in_memory=True)   # keep_in_memory avoids disk I/O

    # 2️⃣  Metric fn: decode → strip → compute → return only rougeL
    def compute_metrics(eval_pred):
        """
        Decode predictions and references, compute ROUGE-L, and return as percentage.

        Args:
            eval_pred (EvalPrediction): Object with .predictions and .label_ids.

        Returns:
            Dict[str, float]: Dictionary with key "rougeL" and its percentage score.
        """
        preds, labels = eval_pred
        if isinstance(preds, tuple):          # HF sometimes returns (logits, ...)
            preds = preds[0]

        # Replace label pad tokens (-100) so they can be decoded
        labels = np.where(labels != -100, labels, tok.pad_token_id)

        decoded_preds  = tok.batch_decode(preds, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
        decoded_labels = tok.batch_decode(labels, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)

        # Strip white-space/newlines that can hurt ROUGE scores
        decoded_preds  = [s.strip() for s in decoded_preds]
        decoded_labels = [s.strip() for s in decoded_labels]

        score_dict = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,        # standard setting for ROUGE-* in HF evaluate
        )

        # HF’s rouge.compute() returns fractional scores; convert to %
        rougeL = round(score_dict["rougeL"] * 100, 4)

        return {"rougeL": rougeL}

    return compute_metrics



