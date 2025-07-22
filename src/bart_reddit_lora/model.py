"""
This module provides utilities to build and load BART models with LoRA adapters
using the PEFT library.
"""

from collections.abc import Sequence
from transformers import BartForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel


def build_base_model(
    checkpoint: str = "facebook/bart-base",
) -> BartForConditionalGeneration:
    """Load a pretrained BART model for conditional generation.

    Args:
        checkpoint (str): Model identifier or path to pretrained weights.

    Returns:
        BartForConditionalGeneration: The loaded BART model.
    """
    base_model = BartForConditionalGeneration.from_pretrained(checkpoint)
    return base_model


def build_peft_model(
    base_model: BartForConditionalGeneration,
    r: int = 8,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "fc1",
        "fc2",
    ),
    modules_to_save: Sequence[str] = ("lm_head",),
) -> PeftModel:
    """Wrap a base BART model with a LoRA adapter configuration.

    Args:
        base_model (BartForConditionalGeneration): The pretrained BART model.
        r (int): LoRA rank (low-rank decomposition dimension).
        lora_dropout (float): Dropout probability for LoRA layers.
        bias (str): How to handle bias in LoRA ('none', 'all', or 'lora_only').
        target_modules (Sequence[str]): Names of modules to apply LoRA to.
        modules_to_save (Sequence[str]): Names of modules to keep trainable.

    Returns:
        PeftModel: The LoRA-adapted model ready for fine-tuning.
    """
    config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=list(modules_to_save),
    )

    lora_model = get_peft_model(base_model, config)
    return lora_model


def load_peft_model_for_inference(
    base_model: BartForConditionalGeneration,
    adapter_path: str = "outputs/bart-base-reddit-lora",
) -> PeftModel:
    """Load a fine-tuned LoRA adapter and set the model to evaluation mode.

    Args:
        base_model (BartForConditionalGeneration): The base BART model architecture.
        adapter_path (str): Path to the directory containing LoRA adapter weights.

    Returns:
        PeftModel: The model with loaded adapter, in eval mode.
    """
    inference_model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    return inference_model
