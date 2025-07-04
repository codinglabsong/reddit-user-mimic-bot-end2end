"""Test suite for model construction and loading utilities in `bart_reddit_lora.model` module."""

from unittest.mock import patch
from pathlib import Path
import sys
from bart_reddit_lora import model as model_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_build_base_model_uses_checkpoint():
    """`build_base_model` should call `BartForConditionalGeneration.from_pretrained` with the provided checkpoint."""
    sentinel = object()
    with patch.object(
        model_mod.BartForConditionalGeneration, "from_pretrained", return_value=sentinel
    ) as mock_fp:
        result = model_mod.build_base_model("checkpoint")
    assert result is sentinel
    mock_fp.assert_called_once_with("checkpoint")


def test_build_peft_model_constructs_config():
    """`build_peft_model` should create a LoraConfig and apply `get_peft_model` with correct args."""
    base = object()
    sentinel_model = object()
    captured = {}

    def fake_config(**kwargs):
        captured["config"] = kwargs

        class C:
            pass

        return C()

    def fake_get_peft_model(bm, cfg):
        captured["bm"] = bm
        captured["cfg"] = cfg
        return sentinel_model

    with patch.object(model_mod, "LoraConfig", side_effect=fake_config), patch.object(
        model_mod, "get_peft_model", side_effect=fake_get_peft_model
    ):
        result = model_mod.build_peft_model(
            base,
            r=4,
            lora_dropout=0.2,
            bias="none",
            target_modules=("x",),
            modules_to_save=("y",),
        )

    assert result is sentinel_model
    assert captured["bm"] is base
    assert captured["config"] == {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["x"],
        "lora_dropout": 0.2,
        "bias": "none",
        "modules_to_save": ["y"],
    }


def test_load_peft_model_for_inference_calls_eval():
    """`load_peft_model_for_inference` should load from pretrained and set model to eval mode."""
    base = object()

    class Dummy:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

    dummy = Dummy()

    with patch.object(
        model_mod.PeftModel, "from_pretrained", return_value=dummy
    ) as mock_load:
        result = model_mod.load_peft_model_for_inference(base, "path")

    mock_load.assert_called_once_with(base, "path")
    assert result is dummy
    assert dummy.evaluated
