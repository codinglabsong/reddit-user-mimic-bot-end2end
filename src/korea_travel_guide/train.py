import argparse
import logging
import random
import numpy as np
import torch
import wandb
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from pathlib import Path
from korea_travel_guide.utils import load_environ_vars, print_trainable_parameters
from korea_travel_guide.model import build_base_model, build_peft_model
from korea_travel_guide.data import tokenize_and_format
from korea_travel_guide.evaluation import build_compute_metrics
from uuid import uuid4

logger = logging.getLogger(__name__)

# Model-specific knobs
@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="facebook/bart-base",
        metadata={"help": "HF checkpoint or path to a local BART."}
    )

# Data-loading toggles
PROJECT_ROOT = Path(__file__).resolve().parents[2]
@dataclass
class DataArgs:
    train_file: Path = PROJECT_ROOT / "data/processed/train.csv"
    validation_file: Path = PROJECT_ROOT / "data/processed/val.csv"
    test_file: Path = PROJECT_ROOT / "data/processed/test.csv"
    train_sample: bool = field(
        default=False,
        metadata={"help": "Use the mini CSV for smoke tests if True."}
    )

# Training & LoRA extras — extend HF’s own Seq2SeqTrainingArguments
@dataclass
class CustomTrainingArgs(Seq2SeqTrainingArguments):
    peft_rank: int = field(default=8,  metadata={"help": "LoRA adapter rank (r)."})
    adapter_path: str = field(
        default="outputs/bart-base-korea-travel-guide-lora",
        metadata={"help": "Prefix folder for all checkpoints/run logs."}
    )
    warmup_ratio: float = 0.05                      # you exposed this already
    push_to_hub: bool = False                       # mirror the old flag
    hf_hub_repo_id: str | None = None
    # run_name keeps sweeps tidy & makes single-GPU debugging easy
    run_name: str = field(
        default_factory=lambda: f"guide-{uuid4().hex[:8]}",
        metadata={"help": "W&B & HF run name."}
    )
    
    # bool for whether to do the final test-set evaluation
    run_test: bool = field(
        default=True,
        metadata={"help": "If False, skip the test-split evaluation after training."}
    )

def parse_args() -> tuple[ModelArgs, DataArgs, CustomTrainingArgs]:
    """Parse CLI → three dataclass objects in one line."""
    parser = HfArgumentParser((ModelArgs, DataArgs, CustomTrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # validations
    if training_args.push_to_hub and not training_args.hf_hub_repo_id:
        parser.error("--hf_hub_repo_id is required when --push_to_hub is set")

    # isolate each run’s artefacts (good for sweeps)
    run_id = os.environ.get("WANDB_RUN_ID", uuid4().hex[:8])
    training_args.output_dir = f"{training_args.adapter_path}/{run_id}"

    # set wandb for logging
    training_args.report_to = "wandb"

    return model_args, data_args, training_args


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()

    # client hyperparams
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    # p.add_argument(
    #     "--peft_rank",
    #     type=int,
    #     default=8,
    #     help="LoRA adapter rank (r) - controls adapter capacity.",
    # )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for optimizer.",
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs."
    )
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training.",
    )
    p.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size per device during evaluation.",
    )
    # p.add_argument(
    #     "--warmup_ratio",
    #     type=float,
    #     default=0.05,
    #     help="Fraction of total steps used for linear warm-up.",
    # )

    # other client params
    # p.add_argument(
    #     "--adapter_path",
    #     type=str,
    #     default="outputs/bart-base-korea-travel-guide-lora",
    #     help="Output directory / model identifier prefix.",
    # )
    # p.add_argument(
    #     "--train_sample",
    #     action="store_true",
    #     help="If set, use the sample training datset for smoke tests.",
    # )
    # p.add_argument(
    #     "--push_to_hub",
    #     action="store_true",
    #     help="If set, push the trained model & tokenizer to the HF Hub.",
    # )
    # p.add_argument(
    #     "--hf_hub_repo_id",
    #     type=str,
    #     default=None,
    #     help="Your HF Hub repo (e.g. username/model-name). Required if --push_to_hub.",
    # )
    p.add_argument(
        "--skip_test",
        action="store_false",
        dest="do_test",
        help="Skip evaluation on the test split after training.",
    )
   
    # validations
    # args = p.parse_args()
    # if args.push_to_hub and not args.hf_hub_repo_id:
    #     p.error("--hf_hub_repo_id is required when --push_to_hub is set")

    return args


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
    model_args, data_args, training_args = parse_args()
    load_environ_vars()
    
    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {DEVICE}")

    # reproducibility
    set_seed(training_args.seed)
    logger.info(f"Set seed: {training_args.seed}")
    
    # ---------- Data Preprocessing ----------
    # load and tokenize dataset
    # load CSVs
    data_files = {
        "train": str(
            project_root 
            / "data/processed"
            / ("train_sample.csv" if cfg.train_sample else "train.csv")
        ),
        "validation": str(project_root / "data/processed/val.csv"),
        "test":       str(project_root / "data/processed/test.csv"),
    }
    
    ds = load_dataset(
        "csv",
        data_files=data_files
    )
    ds_tok, tok = tokenize_and_format(ds)
    
    # initialize base model and LoRA
    base_model = build_base_model()
    logger.info(f"Base model trainable params:\n{print_trainable_parameters(base_model)}")
    lora_model = build_peft_model(base_model, cfg.peft_rank)
    logger.info(
        f"LoRA model (peft_rank={cfg.peft_rank}) trainable params:\n{print_trainable_parameters(lora_model)}"
    )
    
    # ---------- Train ----------
    # setup trainer and train
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.adapter_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        max_grad_norm=0.5,
        label_smoothing_factor=0.1,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        predict_with_generate=True,  # essential for cusstom metrics
        push_to_hub=cfg.push_to_hub,
        report_to="wandb",
        run_name=f"guide-{uuid4().hex[:8]}",
        label_names=["labels"],
    )
    
    # data collator: dynamic padding per batch
    data_collator = DataCollatorForSeq2Seq(
        tok, model=lora_model, 
        padding="longest",  # or "max_length"
        label_pad_token_id=-100
    )

    # initialize trainer & train
    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tok),
    )

    trainer.train()
    
    # ---------- Save, Test or Push ----------
    # evaluate test
    if cfg.do_test:
        logger.info("Running final test-set evaluation...")
        metrics = trainer.evaluate(ds_tok["test"])
        logger.info(f"Test metrics:\n{metrics}")
    else:
        logger.info("Skipping test evaluation.")

    # save model & tokenizer to output_dir
    trainer.save_model()
    logger.info("Saved LoRA model and tokenizer")

    # push to hub
    if cfg.push_to_hub:
        logger.info("Pushing to Huggingface hub...")
        trainer.push_to_hub(
            repo_id=cfg.hf_hub_repo_id,
            finetuned_from="facebook/bart-base",
            commit_message="Finetuned from bart-base on reddit-1k",
        )


if __name__ == "__main__":
    main()