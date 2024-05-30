# Self trained small model results

This folder contains the benchmark results of our
self-trained small models, as output directly from
the benchmark (run) scripts. The folder names are
of the format `{type}-{dataset}`. Below is a key
of type prefixes:

| Prefix | Type |
|----------|----------|
|   vn  | Vanilla (large model only) |
|   ft  | Fine-Tuned (unaligned) |
|   al  | Aligned |

## Results

Below is a table containing the performance of the BiLD
model architecture using our self-trained small models

|                  | Speedup | Fallback Percentage | Rollback Percentage | Rouge L / BLEU  |
|------------------|---------|---------------------|---------------------|-----------------|
| **XSum Vanilla** | NA      | NA                  | NA                  | 35.10 (Rouge L) |
| **XSum Aligned** | 1.62x   | 10.90%              | 4.125%              | 34.28 (Rouge L) |
| **XSum Unaligned** | 1.35x   | 30.43%              | 3.008%              | 34.26 (Rouge L) |
| **WMT14 Vanilla** | NA      | NA                  | NA                  | 31.88 (BLEU)    |
| **WMT14 Aligned** | 1.60x   | 3.230%              | 2.327%              | 28.71 (BLEU)    |
| **WMT14 Unaligned** | 1.41x   | 16.36%              | 2.359%              | 27.81 (BLEU)    |
