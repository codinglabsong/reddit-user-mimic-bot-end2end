"""Test suite for data preprocessing utilities in the `bart_reddit_lora.data` module."""

from pathlib import Path
import sys
import pandas as pd
from datasets import Dataset, DatasetDict
from bart_reddit_lora import data as data_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_clean_text_removes_noise():
    """Ensure `clean_text` strips HTML, code fences, URLs, emojis, quote blocks, and bot footer."""
    raw = "<p>Hello</p>\n```code```\nhttps://example.com :smile:\n&gt; quote\n*I am a bot beep*"
    assert data_mod.clean_text(raw) == "Hello > quote"


def test_preprocess_cleans_fields():
    """Verify `preprocess` cleans HTML tags from 'question' and 'answer' fields, preserving other fields."""
    qa_raw = [
        {"question": "<b>Q?</b>", "answer": "<i>A!</i>", "subreddit": "t", "url": "u"}
    ]
    cleaned = data_mod.preprocess(qa_raw)
    assert cleaned == [{"question": "Q?", "answer": "A!", "subreddit": "t", "url": "u"}]


def test_split_and_save(tmp_path):
    """Test `split_and_save` divides DataFrame into train/val/test and writes CSV files."""
    df = pd.DataFrame(
        {
            "question": [f"q{i}" for i in range(10)],
            "answer": [f"a{i}" for i in range(10)],
            "subreddit": ["s"] * 10,
            "url": ["u"] * 10,
        }
    )
    data_mod.split_and_save(df, tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    val = pd.read_csv(tmp_path / "val.csv")
    test = pd.read_csv(tmp_path / "test.csv")
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_tokenize_and_format(monkeypatch):
    """Confirm `tokenize_and_format` applies tokenizer to inputs and targets with correct truncation."""
    ds = Dataset.from_dict({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})
    ds = DatasetDict({"train": ds, "val": ds})

    class DummyTok:
        def __init__(self):
            self.truncation_side = "right"

        def __call__(
            self,
            text=None,
            text_target=None,
            max_length=None,
            truncation=True,
            padding=False,
        ):
            seqs = text if text_target is None else text_target
            if isinstance(seqs, list):
                return {
                    "input_ids": [[1] * min(max_length, len(t.split())) for t in seqs],
                    "attention_mask": [
                        [1] * min(max_length, len(t.split())) for t in seqs
                    ],
                }
            return {
                "input_ids": [1] * min(max_length, len(seqs.split())),
                "attention_mask": [1] * min(max_length, len(seqs.split())),
            }

    monkeypatch.setattr(
        data_mod,
        "AutoTokenizer",
        type("AT", (), {"from_pretrained": staticmethod(lambda *a, **k: DummyTok())}),
    )

    ds_tok, tok = data_mod.tokenize_and_format(
        ds, checkpoint="none", max_input_length=4, max_target_length=3
    )
    item = ds_tok["train"][0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
