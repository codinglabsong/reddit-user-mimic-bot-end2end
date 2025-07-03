"""
Utilities for environment configuration and reporting model trainable parameters.
"""

import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PreTrainedModel

load_dotenv()

logger = logging.getLogger(__name__)


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Compute and return a summary of trainable vs. total parameters in a model.

    Args:
        model (PreTrainedModel): A Hugging Face Transformers model instance.

    Returns:
        str: Formatted string showing number of trainable parameters, total parameters,
             and percentage of parameters that are trainable.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"


def load_environ_vars(wandb_project: str = "bart-base-reddit-lora"):
    """
    Load and set up environment variables for Hugging Face Hub and Weights & Biases.

    - Logs into Hugging Face Hub if HUGGINGFACE_TOKEN is set.
    - Sets the WANDB_PROJECT environment variable.

    Args:
        wandb_project (str): The default W&B project name to set.
    """
    # hf hub
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged into Hugging Face Hub")
    else:
        logger.info("No HF token found to log into Hugging Face Hub")

    # wandb
    os.environ["WANDB_PROJECT"] = wandb_project
    logger.info(f"W&B project set to '{wandb_project}'")
