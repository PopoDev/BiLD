from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Adafactor
from datasets import load_dataset, DatasetDict

import os
import evaluate
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import pandas as pd

# Set cache to local1/ directory to get around 10GB home directory limit
os.environ['HF_HOME'] = '/local1/hfs/CSE481N_Project/cache' 
os.environ['TRANSFORMERS_CACHE'] = '/local1/hfs/CSE481N_Project/cache' 

# Load the T5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
config_small = AutoConfig.from_pretrained("google-t5/t5-small")
model_small = AutoModelForSeq2SeqLM.from_pretrained(
    "google-t5/t5-small",
    config=config_small,
)

# Load dataset
xsum_dataset = load_dataset('EdinburghNLP/xsum')

# For debugging
new_dataset_dict = DatasetDict()
for dataset_name, dataset in xsum_dataset.items():
    # Select the first 10 rows
    first_10_rows = dataset.select(range(10))
    # Add the first 10 rows to the new DatasetDict
    new_dataset_dict[dataset_name] = first_10_rows

xsum_dataset = new_dataset_dict


def preprocess_function(examples):
    inputs = []
    targets = []
    for i in range(len(examples['document'])):
        if examples['document'][i] is not None and examples['summary'][i] is not None:
            inputs.append(examples['document'][i])
            targets.append(examples['summary'][i])


    inputs = ["summarize: " + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)

    labels = tokenizer(targets, max_length=50, padding="max_length", truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets = xsum_dataset.map(preprocess_function, batched=True, batch_size=16)

# Function for computing ROUGE-L scores (for summarization evaluation)
rouge_score = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 in the predictions as we can't decode them
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    return result


# Data collator for encoder/decoder model (shifts labels by 1 for decoding objective)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_small)
tokenized_datasets = tokenized_datasets.remove_columns(
    xsum_dataset["train"].column_names
)
features = [tokenized_datasets["train"][i] for i in range(2)]


# Set the training arguments: We want to train for exactly 500k steps
# On XSUM use 2e-4 learning rate, on CNN/DailyMail use 5e-4
# Use AdaFactor optimizer, etc. See paper for full hyperparameter list
max_steps = 50000
batch_size = 16
gradient_accumulation_steps = 1
lr = 2e-4
model_name = "google-t5/t5-small".split("/")[-1]
logging_steps = 1
# logging_steps = len(tokenized_datasets["train"]) // batch_size

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-xsum",
    evaluation_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_total_limit=1,
    max_steps=max_steps,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=False,
    generation_max_length=50
)

# Instantiate trainer to finetune model on XSUM/CNN_DailyNews
optimizer = Adafactor(model_small.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=lr)

trainer = Seq2SeqTrainer(
    model_small,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

trainer.train()