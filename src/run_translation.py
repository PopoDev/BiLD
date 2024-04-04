from experiments.translation.run import run_translation
from experiments.translation.arguments import ModelArguments, DataTrainingArguments
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    run_translation(parser)