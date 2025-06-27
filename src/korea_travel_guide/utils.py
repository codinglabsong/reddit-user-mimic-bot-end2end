import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

def load_environ_vars():
    # hf hub
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)
        print("Logged into Hugging Face Hub")
    else:
        print("No HF token found to log into Hugging Face Hub")
    