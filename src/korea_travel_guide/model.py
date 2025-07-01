from transformers import BartForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel


def build_base_model(
    checkpoint: str = "facebook/bart-base",
) -> BartForConditionalGeneration:

    base_model = BartForConditionalGeneration.from_pretrained(checkpoint)
    return base_model


def build_peft_model(
    base_model: BartForConditionalGeneration,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: list[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    modules_to_save: list[str] = ("lm_head",),
) -> PeftModel:
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=list(modules_to_save),
    )

    lora_model = get_peft_model(base_model, config)
    return lora_model


def load_peft_model_for_inference(
    base_model: BartForConditionalGeneration,
    adapter_path: str = "outputs/bart-base-korea-travel-guide-lora",
) -> PeftModel:
    inference_model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    return inference_model
