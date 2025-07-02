import numpy as np
import evaluate
from transformers import EvalPrediction


def build_compute_metrics(tok, num_process_workers: int = 2):
    """Return a closure that Hugging Face's Trainer can call."""
    rouge = evaluate.load("rouge")  # longest-substring overlap
    bleu = evaluate.load("bleu")  # n-gram precision
    bertscore = evaluate.load("bertscore")  # semantic similarity

    def _compute_metrics(eval_pred: EvalPrediction):
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
