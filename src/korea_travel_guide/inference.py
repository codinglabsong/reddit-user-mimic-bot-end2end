import logging
import torch
from contextlib import nullcontext
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)
from pathlib import Path
from korea_travel_guide.utils import load_environ_vars
from korea_travel_guide.model import build_base_model, load_peft_model_for_inference
from korea_travel_guide.data import tokenize_and_format
from korea_travel_guide.evaluation import build_compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class InferenceArgs:
    mode: str = field(
        default="test",
        metadata={
            "help": "`test` runs metrics on the test set; `predict` runs on raw texts."
        },
    )
    batch_size: int = field(
        default=32, metadata={"help": "Batch size for evaluation or prediction."}
    )
    texts: list[str] = field(
        default_factory=list,
        metadata={"help": "One or more input texts for `predict` mode."},
    )
    use_flash_attention: bool = field(
        default=True, metadata={"help": "Enable Flash Attention v1 (via sdp_kernel)."}
    )


def parse_args():
    parser = HfArgumentParser(InferenceArgs)
    (inf_args,) = parser.parse_args_into_dataclasses()
    return inf_args


def main():
    logging.basicConfig(level=logging.INFO)
    load_environ_vars()
    inf_args = parse_args()

    # set device
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Using device: {device}")

    # load tokenizer + model
    tok = AutoTokenizer.from_pretrained("facebook/bart-base")
    base_model = build_base_model()
    model = load_peft_model_for_inference(base_model)

    # prepare Flash‐Attention context
    if inf_args.use_flash_attention and device >= 0:
        logger.info("Using flash attention")
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        )
    else:
        logger.info("Skipping flash attention v1")
        ctx = nullcontext()

    # tokenize & format depending on mode
    if inf_args.mode == "test":
        # load dataset
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        data_files = {
            "train": str(PROJECT_ROOT / "data/processed/train.csv"),
            "validation": str(PROJECT_ROOT / "data/processed/val.csv"),
            "test": str(PROJECT_ROOT / "data/processed/test.csv"),
        }
        ds = load_dataset("csv", data_files=data_files)
        ds_tok, _ = tokenize_and_format(ds)
        data_collator = DataCollatorForSeq2Seq(
            tok, model=model, padding="longest", label_pad_token_id=-100
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir="outputs/inference",
                per_device_eval_batch_size=inf_args.batch_size,
                predict_with_generate=True,
                report_to=[],
            ),
            eval_dataset=ds_tok["test"],
            data_collator=data_collator,
            tokenizer=tok,
            compute_metrics=build_compute_metrics(tok),
        )

        with ctx:
            pred_output = trainer.predict(ds_tok["test"])
        metrics = pred_output.metrics
        logger.info(f"Test metrics: {metrics}")

    elif inf_args.mode == "predict":
        if not inf_args.texts:
            raise ValueError("`--texts` is required in `predict` mode.")
        # batch‐tokenize your inputs
        enc = tok(
            inf_args.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with ctx:
            # fast batched generate
            out = model.generate(
                **enc,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )

        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for inp, pred in zip(inf_args.texts, decoded):
            logger.info(f"Input: {inp}\nOutput: {pred}")

    else:
        raise ValueError(f"Unsupported mode: {inf_args.mode}")


if __name__ == "__main__":
    main()
