from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class SummarizerConfig:
    model_name: str = "google/mt5-small"


def get_model_and_tokenizer(config: SummarizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    return model, tokenizer
