import argparse
import logging
import random
import numpy as np
import torch
import os
from pathlib import Path
from korea_travel_guide.utils import load_environ_vars
from korea_travel_guide.model import build_base_model, build_peft_model
from korea_travel_guide.utils import load_environ_vars

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and evaluation."""
    p = argparse.ArgumentParser()

    # client hyperparams
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    p.add_argument(
        "--peft_rank",
        type=int,
        default=32,
        help="LoRA adapter rank (r) - controls adapter capacity.",
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate for optimizer.",
    )
    p.add_argument(
        "--num_train_epochs", type=int, default=4, help="Number of training epochs."
    )
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size per device during training.",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Batch size per device during evaluation.",
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear warm-up.",
    )

    # other client params
    p.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/bart-base-korea-travel-guide-lora",
        help="Output directory / model identifier prefix.",
    )
    p.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, push the trained model & tokenizer to the HF Hub.",
    )
    p.add_argument(
        "--hf_hub_repo_id",
        type=str,
        default=None,
        help="Your HF Hub repo (e.g. username/model-name). Required if --push_to_hub.",
    )
    p.add_argument(
        "--skip_test",
        action="store_false",
        dest="do_test",
        help="Skip evaluation on the test split after training.",
    )
   
    # validations
    args = p.parse_args()
    if args.push_to_hub and not args.hf_hub_repo_id:
        p.error("--hf_hub_repo_id is required when --push_to_hub is set")

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
    cfg = parse_args()
    load_environ_vars()
    
    # ---------- Initialization ----------
    # choose device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using: {DEVICE}")

    # reproducibility
    set_seed(cfg.seed)
    logger.info(f"Set seed: {cfg.seed}")
    
    # ---------- Data Preprocessing ----------
    # download and tokenize dataset
    current_dir = Path().resolve()
    project_root = current_dir.parent.parent
    print(project_root)

    import sys
    sys.exit()

    # # load CSVs
    # ds = load_dataset(
    #     "csv",
    #     data_files={
    #     "train":      "data/processed/train.csv",
    #     "validation": "data/processed/val.csv",
    #     "test":       "data/processed/test.csv",
    #     }
    # )

    # # data collator: dynamic padding per batch
    #     tokenizer, model=model, 
    #     padding="longest",  # or "max_length"
    #     label_pad_token_id=-100
    # )

    # # training arguments
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="outputs/bart-base-korea-lora",
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=32,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_strategy="steps",
    #     logging_steps=50,
    #     num_train_epochs=5,
    #     learning_rate=1e-4,
    #     fp16=True,
    # )

    # # initialize trainer & train
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized["train"],
    #     eval_dataset=tokenized["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     predict_with_generate=True,  # <-- essential for cusstom metrics
    # )

    # trainer.train()
    

if __name__ == "__main__":
    main()