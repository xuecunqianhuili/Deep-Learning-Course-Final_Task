from typing import Optional, Dict
from datasets import load_dataset, DatasetDict

SUMMARIZATION_COLUMN_MAP = {
    "cnn_dailymail": {"article": "text", "highlights": "summary"},
    "xsum": {"document": "text", "summary": "summary"},
    "xlsum_zh": {"text": "text", "summary": "summary"},
}


def unify_columns(ds: DatasetDict, mapping: Dict[str, str]) -> DatasetDict:
    def _rename(example):
        return {
            "text": example[mapping["text"]],
            "summary": example[mapping["summary"]],
        }

    return DatasetDict({
        split: split_ds.map(_rename, remove_columns=split_ds.column_names)
        for split, split_ds in ds.items()
    })


def load_summarization_dataset(name: str = "xlsum_zh", subset_size: Optional[int] = None) -> DatasetDict:
    """Load a summarization dataset and normalize to columns {text, summary}.
    Args:
        name: dataset identifier ("cnn_dailymail", "xsum", or "xlsum_zh").
        subset_size: if provided, truncate each split to this many samples for quick runs.
    Returns:
        DatasetDict with keys train/validation/test and columns text, summary.
    """
    if name == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", version="3.0.0")
        mapping = SUMMARIZATION_COLUMN_MAP["cnn_dailymail"]
    elif name == "xsum":
        ds = load_dataset("xsum")
        mapping = SUMMARIZATION_COLUMN_MAP["xsum"]
    elif name == "xlsum_zh":
        # Load XLSum Chinese Simplified
        ds = load_dataset("csebuetnlp/xlsum", name="chinese_simplified", trust_remote_code=True)
        mapping = SUMMARIZATION_COLUMN_MAP["xlsum_zh"]
    else:
        raise ValueError(f"Unsupported dataset: {name}")


    # Build mapping to target keys
    reverse_map = {v: k for k, v in mapping.items()}
    # rename to text/summary via map
    def _rename(example):
        return {
            "text": example[reverse_map.get("text", list(mapping.keys())[0])],
            "summary": example[reverse_map.get("summary", list(mapping.keys())[1])],
        }

    unified = DatasetDict({
        split: split_ds.map(_rename, remove_columns=split_ds.column_names)
        for split, split_ds in ds.items()
    })

    if subset_size:
        for split in list(unified.keys()):
            unified[split] = unified[split].select(range(min(subset_size, len(unified[split]))))

    return unified


def tokenize_summarization(ds: DatasetDict, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
    """Tokenize summarization dataset for seq2seq training."""
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        preprocess_function,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing dataset",
    )
    return tokenized
