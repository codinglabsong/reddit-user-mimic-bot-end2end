import torch
import gradio as gr
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration
from peft import PeftModel


def load_model():
    """
    Load environment variables, tokenizer, and the fine-tuned LoRA model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    # Use efficient attention if desired
    base_model.config.attn_implementation = "sdpa"

    # Load PEFT (LoRA) model for inference
    model = PeftModel.from_pretrained(
        base_model, "outputs/bart-base-reddit-lora"
    ).eval()
    model.to(device)
    model.eval()

    return tokenizer, model, device


# Load once at startup
tokenizer, model, device = load_model()


def predict(text: str) -> str:
    """
    Generate a response for a single input text.
    """
    # Tokenize and move inputs to device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Generate with both beam search and sampling for diversity
    outputs = model.generate(
        **inputs,
        max_length=500,
        num_beams=5,
        do_sample=True,
        length_penalty=1.2,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        top_p=0.9,
        temperature=0.8,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the first generated sequence
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=5, placeholder="Ask a Question", label="Your Question"),
        outputs=gr.Textbox(label="Model Output"),
        title="Bart-Reddit-LoRA Inference",
        description="Enter your prompt and click Submit to get the model's response.",
        allow_flagging="never",
    )
    interface.launch()


if __name__ == "__main__":
    main()
