from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from typing import Union

def tokenizeDataset(
    examples: Union[Dataset, DatasetDict],
    tokenizer: AutoTokenizer,
    input_key: str = "article",
    output_key: str = "highlights",
    prefix: str = "summarize: ",
    max_input_length: int = 1024,
    max_output_length: int = 128
) -> DatasetDict:
    inputs = [prefix + doc for doc in examples[input_key]]
    model_inputs = tokenizer(
        inputs,
        padding = "max_length",
        max_length = max_input_length,
        truncation = True
    )

    labels = tokenizer(
        text_target = examples[output_key],
        padding = "max_length",
        max_length = max_output_length, 
        truncation = True
    )

    # Remove untokenized data
    model_inputs["labels"] = labels["input_ids"]
    model_inputs = model_inputs.remove_columns(examples["train"].column_names)

    return model_inputs
