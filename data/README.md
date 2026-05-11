# Data Preparation

This repository does not redistribute third-party benchmark datasets, GloVe embeddings, or pre-trained model checkpoints.

The experiments are based on public benchmark resources. Users should obtain the original datasets from official or publicly available sources and organize them into the formats described below.

The most important requirement is that the processed files match the expected file names, fields, split definitions, and row counts used by the scripts.

## Recommended Directory Structure

```text
data/
├── SICK-R/
│   └── SICK.txt
├── STS-B/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── STS-X/
│   ├── STS-12/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   ├── STS-13/
│   │   └── test.jsonl
│   ├── STS-14/
│   │   └── test.jsonl
│   ├── STS-15/
│   │   └── test.jsonl
│   └── STS-16/
│       └── test.jsonl
├── QQP/
│   ├── train.tsv
│   └── dev.tsv
└── GloVe/
    └── glove.840B.300d.txt
```

The scripts can also be run with custom paths through command-line arguments. The structure above is only the recommended organization.

## SICK-R

Expected file:

```text
data/SICK-R/SICK.txt
```

Expected format: tab-separated text file.

Required columns:

```text
SemEval_set
sentence_A
sentence_B
relatedness_score
```

Expected split labels:

```text
TRAIN
TRIAL
TEST
```

The scripts use:

```text
TRAIN -> training set
TRIAL -> validation set
TEST  -> test set
```

The relatedness score is expected to be in the original SICK-R range `[1, 5]`. The scripts normalize it internally as:

```text
normalized_score = (relatedness_score - 1.0) / 4.0
```

Expected split sizes used in the experiments:

```text
TRAIN: 4439
TRIAL: 495
TEST : 4906
```

## STS-B

Expected files:

```text
data/STS-B/train.jsonl
data/STS-B/validation.jsonl
data/STS-B/test.jsonl
```

Expected format: JSON Lines. Each line should be one JSON object:

```json
{"sentence1": "...", "sentence2": "...", "score": 3.8}
```

Required fields:

```text
sentence1
sentence2
score
```

The score is expected to be in the original STS range `[0, 5]`. The scripts normalize it internally as:

```text
normalized_score = score / 5.0
```

Expected split sizes used in the experiments:

```text
train      : 5749
validation : 1500
test       : 1379
```

## STS-12 to STS-16

The STS-12 to STS-16 experiments use a leave-one-year-out pooled STS protocol.

Expected atomic files:

```text
data/STS-X/STS-12/train.jsonl
data/STS-X/STS-12/test.jsonl
data/STS-X/STS-13/test.jsonl
data/STS-X/STS-14/test.jsonl
data/STS-X/STS-15/test.jsonl
data/STS-X/STS-16/test.jsonl
```

Each JSONL line should have the following format:

```json
{"sentence1": "...", "sentence2": "...", "score": 4.2}
```

Required fields:

```text
sentence1
sentence2
score
```

The score is expected to be in `[0, 5]`, and the scripts normalize it internally as:

```text
normalized_score = score / 5.0
```

### Leave-One-Year-Out Protocol

For each target year `Y` in:

```text
STS-12, STS-13, STS-14, STS-15, STS-16
```

the official test split of year `Y` is held out as the test set.

The training pool is constructed from:

```text
STS-12 official train split
+
official test splits of all non-target years
```

The scripts then perform:

```text
1. canonical sentence-pair deduplication
2. target-test contamination filtering
3. stratified train/validation split based on score bins
```

Important:

```text
Do not use cumulative train.jsonl files from STS-13, STS-14, STS-15, or STS-16 as atomic training sources.
```

The scripts include defensive checks to avoid accidentally loading such cumulative training files.

Expected final split sizes used in the experiments:

```text
STS-12: train 9134,  validation 2282, test 3108
STS-13: train 10256, validation 2560, test 1500
STS-14: train 8516,  validation 2126, test 3750
STS-15: train 9056,  validation 2262, test 3000
STS-16: train 10506, validation 2623, test 1186
```

## QQP

Expected files:

```text
data/QQP/train.tsv
data/QQP/dev.tsv
```

Expected format: tab-separated text file.

The scripts support either GLUE-style column names:

```text
question1
question2
is_duplicate
```

or HuggingFace-style aliases:

```text
sentence1
sentence2
label
```

The label should be binary:

```text
0 = non-duplicate
1 = duplicate
```

The QQP scripts use the training file for training-size experiments and the development file as the held-out evaluation set.

Typical training-size settings include:

```text
30000
50000
100000
all
```

For the trainable QQP models, an internal validation split is sampled from the selected training subset for early stopping and learning-rate scheduling.

## GloVe

Expected file:

```text
data/GloVe/glove.840B.300d.txt
```

The scripts expect the 300-dimensional GloVe format:

```text
word value_1 value_2 ... value_300
```

GloVe embeddings are used as frozen word embeddings in the trainable GloVe-based models.

## Pre-trained Model Checkpoints

This repository does not redistribute large pre-trained model checkpoints.

Scripts using BERT, RoBERTa, SimCSE, Sentence-T5, or E5 expect local HuggingFace-compatible checkpoint folders. Please pass the corresponding local paths through the command-line arguments of each script.

## Sanity Checks

Before running full experiments, verify that:

1. Required files exist.
2. Required columns or fields are present.
3. Split sizes match the expected values above.
4. Score ranges are correct.
5. GloVe dimensions are 300.
6. QQP labels are binary.
7. STS-X LOO scripts are using atomic STS files only.

These checks are important because different public mirrors or reprocessed versions of the same benchmark may use slightly different field names, split names, or file structures.
