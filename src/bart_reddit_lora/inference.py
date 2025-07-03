import logging
import torch
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
from bart_reddit_lora.utils import load_environ_vars
from bart_reddit_lora.model import build_base_model, load_peft_model_for_inference
from bart_reddit_lora.data import tokenize_and_format
from bart_reddit_lora.evaluation import build_compute_metrics

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
    num_process_workers: int = field(
        default=2,
        metadata={"help": "Number of workers to parallelize n-gram counting."},
    )
    use_sdpa_attention: bool = field(
        default=True, metadata={"help": "Enable Sdpa for mem-efficient kernel."}
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # load tokenizer + model
    tok = AutoTokenizer.from_pretrained("facebook/bart-base")
    base_model = build_base_model()
    if inf_args.use_sdpa_attention:
        base_model.config.attn_implementation = "sdpa"
    model = load_peft_model_for_inference(base_model)
    model.to(device)

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
                generation_max_length=384,
                report_to=[],
            ),
            eval_dataset=ds_tok["test"],
            data_collator=data_collator,
            tokenizer=tok,
            compute_metrics=build_compute_metrics(tok, inf_args.num_process_workers),
        )

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
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # fast batched generate (with arguments for higher quality generations)
        out = model.generate(
            **enc,
            max_length=500,
            num_beams=5,  # improves quality
            do_sample=True,  # add stochasticity
            length_penalty=1.2,  # >1 favors longer answers
            repetition_penalty=1.3,  # >1 penalizes reuse of the same token
            no_repeat_ngram_size=3,  # block exact n-gram repeats
            top_p=0.9,  # nucleus sampling for diversity
            temperature=0.8,  # nucleus sampling for diversity
            early_stopping=True,  # stop on EOS to avoid garbage at the end
            eos_token_id=tok.eos_token_id,
        )

        decoded = tok.batch_decode(out, skip_special_tokens=True)
        for inp, pred in zip(inf_args.texts, decoded):
            logger.info(f"Input: {inp}\nOutput: {pred}")

    else:
        raise ValueError(f"Unsupported mode: {inf_args.mode}")


if __name__ == "__main__":
    main()
