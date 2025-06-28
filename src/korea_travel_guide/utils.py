import logging
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

logger = logging.getLogger(__name__)

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