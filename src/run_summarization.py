from experiments.summarization.run import run_summarization
from experiments.summarization.arguments import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    run_summarization(parser)