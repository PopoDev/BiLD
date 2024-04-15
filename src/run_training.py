from training.run import run_training
from training.arguments import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    run_training(parser)