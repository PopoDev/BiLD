# CSE481N Reproducibility Project

This repo's goal is to reproduce the results of the paper [Speculative Decoding with Big Little Decoder (BiLD)](https://proceedings.neurips.cc/paper_files/paper/2023/file/7b97adeafa1c51cf65263459ca9d0d7c-Paper-Conference.pdf) by Kim et al. (2023). The paper is published in NeurIPS 2023 and the code can be found on the [official repository](https://github.com/kssteven418/BigLittleDecoder).

## Introduction

Big Little Decoder is a simple framework that enables **faster generative inference**. 
It can dramatically accelerate text generation by ~2x, without compromising performance on a variety of text generation scenarios. 
Furthermore, it is a simple **plug-and-play** solution that requires no training or architecture redesign.

Here's the key underlying idea:

1. BiLD offloads the majority of simple word decisions to a smaller model, and only switches the control back to the larger model when needed.
2. The small model **"fallbacks"** to the large model, when it runs into a hard-to-predict word.
3. In case the small model makes a misstep, the larger model can **"rollback"** the predictions to correct the error
4. This **collaborative text generation** combines the small model's fast autoregressive execution with the large model's accurate and efficient non-autoregressive execution!

# Reproduction

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Evaluation

We provide the scripts to evaluate the pretrained models on the datasets used in the paper.

### Translation

To evaluate the pretrained models on the translation datasets, run the following command:

```bash
# IWSLT2017 with unaligned model
CUDA_VISIBLE_DEVICES=0 ./run_iwslt2017_unaligned.sh

# WMT2014 with aligned model
CUDA_VISIBLE_DEVICES=0 ./run_wmt2014_aligned.sh
```

### Summarization

We will provide the scripts to evaluate the pretrained models on the summarization datasets soon.

## Results

### IWSLT2017

| IWSLT2017           | Original           | Reproduction        |
|---------------------|--------------------|---------------------|
| BiLD (unaligned)    | 40.33 & 1.43x      | 40.33 & 1.46x       |
| BiLD (aligned)      | 40.24 & 1.62x      | 40.09 & 1.52x       |

- For the unaligned model, we were able to reproduce the BLEU score of 40.33 with a speedup of 1.46x using RB, FB= (2, 0.6).
- For the aligned model, we were able to reproduce the BLEU score of 40.09 with a speedup of 1.52x using RB, FB= (3, 0.9).

### WMT2014

We will provide the results for the WMT2014 dataset soon.

### XSUM

We will provide the results for the XSUM dataset soon.

### CNNDM

We will provide the results for the CNNDM dataset soon.

## Pretrained Checkpoints

### Authors' Checkpoints

The authors provided the finetuned checkpoints used in the paper.

| Dataset |  Model | Link |
| -------- | -------- | -------- | 
| IWSLT-2017-De-En    |  mT5-small  |  [link](https://huggingface.co/kssteven/mT5-small-iwslt2017-de-en) | 
| IWSLT-2017-De-En    |  mT5-small (aligned)  |  [link](https://huggingface.co/kssteven/mT5-small-iwslt2017-de-en-bild-aligned) | 
| IWSLT-2017-De-En    |  mT5-large  |  [link](https://huggingface.co/kssteven/mT5-large-iwslt2017-de-en) | 
| WMT-2014-De-En    |  mT5-small  |  [link](https://huggingface.co/kssteven/mT5-small-wmt2014-de-en) | 
| WMT-2014-De-En    |  mT5-small (aligned)  |  [link](https://huggingface.co/kssteven/mT5-small-wmt2014-de-en-bild-aligned) | 
| WMT-2014-De-En    |  mT5-large  |  [link](https://huggingface.co/kssteven/mT5-large-wmt2014-de-en) | 
| XSUM    |  T5-small  |  [link](https://huggingface.co/kssteven/T5-small-xsum) | 
| XSUM    |  T5-small (aligned)  |  [link](https://huggingface.co/kssteven/T5-small-xsum-bild-aligned) | 
| XSUM    |  T5-large  |  [link](https://huggingface.co/kssteven/T5-large-xsum) | 
| CNNDM    |  T5-small  |  [link](https://huggingface.co/kssteven/T5-small-cnndm) | 
| CNNDM    |  T5-small (aligned) |  [link](https://huggingface.co/kssteven/T5-small-cnndm-bild-aligned) | 
| CNNDM    |  T5-large  |  [link](https://huggingface.co/kssteven/T5-large-cnndm) | 

### Aligned Models
We are currently training our own aligned models to reproduce the methodology used in the paper. We will provide the links to the checkpoints once they are available.