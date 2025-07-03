"""
Fine-tune a BART model with LoRA on Reddit Q&A data using Hugging Face Transformers.

This module handles argument parsing, data loading (CSV or SQuAD), tokenization,
model initialization (with optional SDPA attention), training with early stopping,
and optional evaluation/pushing to the Hugging Face Hub.
"""

import logging
import random
import numpy as np
import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from typing import Tuple, List, Optional, Dict
from pathlib import Path
from bart_reddit_lora.utils import load_environ_vars, print_trainable_parameters
from bart_reddit_lora.model import build_base_model, build_peft_model
from bart_reddit_lora.data import tokenize_and_format
from uuid import uuid4

logger = logging.getLogger(__name__)

# data-loading toggles
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataArgs:
    """
    Configuration for data file paths and loading options.
    """
    train_file: Path = PROJECT_ROOT / "data/processed/train.csv"
    validation_file: Path = PROJECT_ROOT / "data/processed/val.csv"
    test_file: Path = PROJECT_ROOT / "data/processed/test.csv"
    train_sample_file: Path = PROJECT_ROOT / "data/processed/train_sample.csv"
    train_sample: bool = field(
        default=False, metadata={"help": "Use the mini CSV for smoke tests if True."}
    )
    use_squad: bool = field(
        default=False, metadata={"help": "If True, ignore CSVs and load SQuAD instead."}
    )


# training & LoRA extras — extend HF’s own Seq2SeqTrainingArguments
@dataclass
class CustomTrainingArgs(Seq2SeqTrainingArguments):
    """
    Extends Seq2SeqTrainingArguments with LoRA-specific and project defaults.
    """
    # overriding the hf defaults
    output_dir: str = field(
        default="outputs/bart-base-reddit-lora",
        metadata={"help": "Prefix folder for all checkpoints/run logs."},
    )
    num_train_epochs: int = 12
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 3e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5
    label_smoothing_factor: float = 0.1
    weight_decay: float = 0.01

    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 1
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False

    fp16: bool = True
    push_to_hub: bool = False
    report_to: str = "wandb"
    run_name: str = field(
        default_factory=lambda: f"guide-{uuid4().hex[:8]}",
        metadata={"help": "W&B & HF run name."},
    )  # run_name keeps sweeps tidy & makes single-GPU debugging easy
    label_names: List[str] = field(default_factory=lambda: ["labels"])

    # additional custom args
    peft_rank: int = field(default=128, metadata={"help": "LoRA adapter rank (r)."})
    hf_hub_repo_id: str | None = None
    run_test: bool = field(
        default=False,
        metadata={"help": "If True, run the test-split evaluation after training."},
    )
    use_sdpa_attention: bool = field(
        default=True, metadata={"help": "Enable Sdpa for mem-efficient kernel."}
    )


def parse_args() -> tuple[DataArgs, CustomTrainingArgs]:
    """
    Parse command-line arguments into DataArgs and CustomTrainingArgs.

    Raises:
        SystemExit: if `--push_to_hub` is set without `--hf_hub_repo_id`.
    """
    parser = HfArgumentParser((DataArgs, CustomTrainingArgs))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # validations
    if training_args.push_to_hub and not training_args.hf_hub_repo_id:
        parser.error("--hf_hub_repo_id is required when --push_to_hub is set")

    # set wandb for logging
    training_args.report_to = "wandb"

    return data_args, training_args


def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, Torch) for deterministic runs."""
    random.seed(seed)  # vanilla Python Random Number Generator (RNG)
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # CPU-side torch RNG
    torch.cuda.manual_seed_all(seed)  # all GPU RNGs
    torch.backends.cudnn.deterministic = True  # force deterministic conv kernels
    torch.backends.cudnn.benchmark = False  # trade speed for reproducibility


def main() -> None:
    """Entry point: parse args, prepare data, train, evaluate, and optionally push."""
    logging.basicConfig(level=logging.INFO)
    data_args, training_args = parse_args()
    load_environ_vars()

    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {DEVICE}")

    # reproducibility
    set_seed(training_args.seed)
    logger.info(f"Set seed: {training_args.seed}")

    # ---------- Data Preprocessing ----------
    # load either CSVs or SQuAD for a quick pipeline sanity check
    if data_args.use_squad:
        # 1) pull down SQuAD
        raw = load_dataset("squad")

        # 2) map to simple Q/A pairs (first answer only)
        def to_qa(ex):
            return {"question": ex["question"], "answer": ex["answers"]["text"][0]}

        ds = raw.map(to_qa, remove_columns=raw["train"].column_names)
    else:
        # load from your processed CSVs
        data_files = {
            "train": str(
                data_args.train_sample_file
                if data_args.train_sample
                else data_args.train_file
            ),
            "validation": str(data_args.validation_file),
            "test": str(data_args.test_file),
        }
        ds = load_dataset("csv", data_files=data_files)
    # # load and tokenize dataset
    # # load CSVs
    # data_files = {
    #     "train": str(
    #         data_args.train_sample_file
    #         if data_args.train_sample
    #         else data_args.train_file
    #     ),
    #     "validation": str(data_args.validation_file),
    #     "test": str(data_args.test_file),
    # }

    # ds = load_dataset("csv", data_files=data_files)
    ds_tok, tok = tokenize_and_format(ds)

    # initialize base model and LoRA
    base_model = build_base_model()
    if training_args.use_sdpa_attention:
        base_model.config.attn_implementation = "sdpa"
    logger.info(
        f"Base model trainable params:\n{print_trainable_parameters(base_model)}"
    )
    lora_model = build_peft_model(base_model, training_args.peft_rank)
    logger.info(
        f"LoRA model (peft_rank={training_args.peft_rank}) trainable params:\n{print_trainable_parameters(lora_model)}"
    )

    # ---------- Train ----------
    # data collator: dynamic padding per batch
    data_collator = DataCollatorForSeq2Seq(
        tok,
        model=lora_model,
        padding="longest",  # or "max_length"
        label_pad_token_id=-100,
        pad_to_multiple_of=8,  # tensor-core friendly
    )

    # initialize trainer & train
    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ---------- Save, Test or Push ----------
    # evaluate test
    if training_args.run_test:
        logger.info("Running final test-set evaluation...")
        metrics = trainer.evaluate(ds_tok["test"])
        logger.info(f"Test metrics:\n{metrics}")
    else:
        logger.info("Skipping test evaluation.")

    # save model & tokenizer to output_dir
    trainer.save_model()
    logger.info("Saved LoRA model and tokenizer")

    # push to hub
    if training_args.push_to_hub:
        logger.info("Pushing to Huggingface hub...")
        trainer.push_to_hub(
            repo_id=training_args.hf_hub_repo_id,
            finetuned_from="facebook/bart-base",
            commit_message="Finetuned from bart-base on reddit-1k",
        )


if __name__ == "__main__":
    main()
