from transformers import (
    T5ForConditionalGeneration,    
    T5Tokenizer, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM, 
    AutoConfig, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Adafactor,
    HfArgumentParser
)
from datasets import load_dataset, DatasetDict
from huggingface_hub import notebook_login

import os
import evaluate
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import pandas as pd

from .preprocessers import xsum_preprocess_function


def run_training(parser: HfArgumentParser):
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    cache_dir = model_args.cache_dir
    model_name = model_args.model_name

    # Load the specified model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )

    # Load dataset
    dataset = load_dataset(data_args.dataset_name, cache_dir=cache_dir)
    
    # For debugging
    if data_args.dataset_size != -1:
        new_dataset_dict = DatasetDict()
        for dataset_name, dataset in dataset.items():
            selected_rows = dataset.select(range(data_args.dataset_size))
            new_dataset_dict[dataset_name] = selected_rows
        dataset = new_dataset_dict

    if data_args.dataset_name == 'EdinburghNLP/xsum':
        preprocess_function = xsum_preprocess_function
    else:
        raise ValueError(f"Dataset {preprocess_function} does not have a corresponding preprocess function.")

    # Preprocess dataset to prepare for training
    tokenized_datasets = dataset.map(preprocess_function, 
                                     batched=True, 
                                     batch_size=training_args.per_device_train_batch_size, 
                                     load_from_cache_file=True,
                                     fn_kwargs={"tokenizer": tokenizer})


    # Function for computing ROUGE-L scores (for summarization evaluation)
    def rouge_compute_metrics(eval_pred):
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


    # Setup compute metrics for evaluation (ROUGE-L/BLEU for summarization/translation)
    rouge_score = evaluate.load("rouge", cache_dir=cache_dir)
    if data_args.metric == 'rouge':
        compute_metrics = rouge_compute_metrics
    else:
        raise ValueError(f"Metric could not be found for {data_args.metric}")

    # Data collator for encoder/decoder model (shifts labels by 1 for decoding objective)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(
        dataset["train"].column_names
    )

    # Extract hyperparameters for training
    logging_steps = len(tokenized_datasets["train"]) // training_args.per_device_train_batch_size
    args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        save_total_limit=training_args.save_total_limit,
        max_steps=training_args.max_steps,
        predict_with_generate=training_args.predict_with_generate,
        logging_steps=logging_steps,
        push_to_hub=training_args.push_to_hub,
        generation_max_length=training_args.generation_max_length
    )


    optimizer = Adafactor(model.parameters(), 
                          scale_parameter=False, 
                          relative_step=False, 
                          warmup_init=False, 
                          lr=training_args.learning_rate)
    

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    trainer.train()
    trainer.push_to_hub(commit_message="Training complete", tags="summarization")
