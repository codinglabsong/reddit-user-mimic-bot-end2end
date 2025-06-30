import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PreTrainedModel

load_dotenv()

logger = logging.getLogger(__name__)


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: any HF Transformers model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"


def load_environ_vars(wandb_project: str = "bart-base-korea-travel-guide-lora"):
    # hf hub
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged into Hugging Face Hub")
    else:
        logger.info("No HF token found to log into Hugging Face Hub")

    # wandb
    os.environ["WANDB_PROJECT"] = wandb_project
