import logging
import random
import numpy as np
import torch
from contextlib import nullcontext
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
from typing import List
from pathlib import Path
from korea_travel_guide.utils import load_environ_vars, print_trainable_parameters
from korea_travel_guide.model import build_base_model, build_peft_model
from korea_travel_guide.data import tokenize_and_format
from korea_travel_guide.evaluation import build_compute_metrics
from uuid import uuid4

logger = logging.getLogger(__name__)

# data-loading toggles
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataArgs:
    train_file: Path = PROJECT_ROOT / "data/processed/train.csv"
    validation_file: Path = PROJECT_ROOT / "data/processed/val.csv"
    test_file: Path = PROJECT_ROOT / "data/processed/test.csv"
    train_sample_file: Path = PROJECT_ROOT / "data/processed/train_sample.csv"
    train_sample: bool = field(
        default=False, metadata={"help": "Use the mini CSV for smoke tests if True."}
    )


# training & LoRA extras — extend HF’s own Seq2SeqTrainingArguments
@dataclass
class CustomTrainingArgs(Seq2SeqTrainingArguments):
    # overriding the hf defaults
    output_dir: str = field(
        default="outputs/bart-base-korea-travel-guide-lora",
        metadata={"help": "Prefix folder for all checkpoints/run logs."},
    )
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 5
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.05
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    max_grad_norm: float = 0.5
    # label_smoothing_factor: float = 0.1
    weight_decay: float = 0.01
    save_total_limit: int = 2
    fp16: bool = True
    predict_with_generate: bool = True
    push_to_hub: bool = False
    report_to: str = "wandb"
    run_name: str = field(
        default_factory=lambda: f"guide-{uuid4().hex[:8]}",
        metadata={"help": "W&B & HF run name."},
    )  # run_name keeps sweeps tidy & makes single-GPU debugging easy
    label_names: List[str] = field(default_factory=lambda: ["labels"])

    # additional custom args
    peft_rank: int = field(default=32, metadata={"help": "LoRA adapter rank (r)."})
    hf_hub_repo_id: str | None = None
    run_test: bool = field(
        default=False,
        metadata={"help": "If True, run the test-split evaluation after training."},
    )
    use_flash_attention: bool = field(
        default=True, metadata={"help": "Whether to enable Flash Attention v1."}
    )


def parse_args() -> tuple[DataArgs, CustomTrainingArgs]:
    """Parse CLI → three dataclass objects in one line."""
    parser = HfArgumentParser((DataArgs, CustomTrainingArgs))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # validations
    if training_args.push_to_hub and not training_args.hf_hub_repo_id:
        parser.error("--hf_hub_repo_id is required when --push_to_hub is set")

    # # isolate each run’s artefacts (good for sweeps)
    # run_id = os.environ.get("WANDB_RUN_ID", uuid4().hex[:8])
    # training_args.output_dir = f"{training_args.output_dir}/{run_id}"

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
    # load and tokenize dataset
    # load CSVs
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
    ds_tok, tok = tokenize_and_format(ds)

    # initialize base model and LoRA
    base_model = build_base_model()
    logger.info(
        f"Base model trainable params:\n{print_trainable_parameters(base_model)}"
    )
    lora_model = build_peft_model(base_model, training_args.peft_rank)
    logger.info(
        f"LoRA model (peft_rank={training_args.peft_rank}) trainable params:\n{print_trainable_parameters(lora_model)}"
    )

    # from torch.utils.data import DataLoader

    # data_collator = DataCollatorForSeq2Seq(
    #     tok,
    #     model=lora_model,
    #     padding="longest",
    #     label_pad_token_id=-100,
    # )

    # batch = next(iter(DataLoader(ds_tok["train"], batch_size=2, collate_fn=data_collator )))
    # # 1) decode inputs normally
    # print("INPUTS:")
    # print(tok.batch_decode(batch["input_ids"], skip_special_tokens=True))

    # # 2) map -100 → pad_token_id before decoding labels
    # labels = batch["labels"].detach().cpu().numpy()
    # labels = np.where(labels != -100, labels, tok.pad_token_id)

    # print("LABELS:")
    # print(tok.batch_decode(labels, skip_special_tokens=True))

    # import sys
    # sys.exit()

    # ---------- Train ----------
    # toggle flash attention
    if training_args.use_flash_attention:
        logger.info("Using flash attention")
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        )
    else:
        logger.info("Skipping flash attention")
        ctx = nullcontext()
    # ctx = nullcontext()

    # data collator: dynamic padding per batch
    data_collator = DataCollatorForSeq2Seq(
        tok,
        model=lora_model,
        padding="longest",  # or "max_length"
        label_pad_token_id=-100,
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

    with ctx:
        trainer.train()

    # ---------- Save, Test or Push ----------
    # evaluate test
    if training_args.run_test:
        logger.info("Running final test-set evaluation...")
        with ctx:
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
