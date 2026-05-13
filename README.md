# QSTS: Quantum Semantic Textual Similarity on Quantum Circuits

This repository provides the code for a quantum semantic textual similarity (QSTS) model implemented with parameterized quantum circuits, together with the experimental code used for benchmark comparisons, ablation studies, robustness analysis, scalability analysis, generalization analysis, and theoretical circuit analysis.

The repository is organized to make the experimental workflow transparent and reproducible. It includes the proposed QSTS model, classical lexical baselines, classical neural baselines, large pre-trained model baselines, hybrid quantum baselines, topology/effectiveness comparisons, circuit-depth and circuit-width studies, noise-robustness simulations, QQP generalization experiments, and expressibility/entangling-capability analyses.

## Repository Structure

```text
.
├── Baselines/
│   ├── SICK-R/
│   ├── STS-B/
│   └── STS-X/
├── Performance/
│   ├── Effectiveness/
│   ├── Scalability/
│   ├── Robustness/
│   └── Generalization/
├── Theoretical_Analysis/
├── data/
│   └── README.md
├── README.md
└── requirements.txt
```

## Included Experiments

### Main QSTS experiments

The repository includes QSTS implementations for the main semantic textual similarity benchmarks:

```text
SICK-R
STS-B
STS-12 to STS-16 under the leave-one-year-out protocol
```

The main STS experiments use word embeddings, amplitude encoding, parameterized textual feature learning circuits, and similarity estimation through the quantum overlap between learned sentence states.

### Classical lexical baselines

The code includes:

```text
BoW
HashVec
GloVe
```

These baselines are provided for SICK-R, STS-B, and STS-12 to STS-16.

### Classical neural baselines

The code includes standard trainable neural sentence encoders:

```text
CNN
SANN
GRU
BiGRU
```

These models use the same general preprocessing and evaluation protocol as QSTS wherever applicable.

### Large pre-trained model baselines

The code includes scripts for large pre-trained language models and sentence encoders, including frozen, fine-tuned, and zero-shot settings where applicable.

The repository does not redistribute model checkpoints. Users should prepare local HuggingFace-compatible checkpoint folders as required by each script and pass paths through command-line arguments.

### Hybrid quantum baselines

The code includes hybrid quantum baseline scripts based on quantum feed-forward and quantum convolutional designs.

Scripts named `QMLP_*` implement the QFFN-QAOA and QFFN-HEA baselines described in the paper. The two variants can be selected by:

```bash
--variant qaoa
--variant hea
```

QCNN-based hybrid quantum baselines are also included.

### Effectiveness analysis

The `Performance/Effectiveness/` folder contains scripts for circuit-topology and reference-circuit comparisons.

Naming notes:

```text
A2A denotes the all-to-all topology.
NN denotes nearest-neighbor topology.
CB denotes circuit-block topology.
ref denotes the reference circuit family.
```

### Scalability analysis

The `Performance/Scalability/` folder contains scripts for circuit-depth and circuit-width studies.

### Robustness analysis

The `Performance/Robustness/` folder contains noise-related simulation scripts for evaluating the robustness of QSTS under different noise settings and hardware-motivated perturbations.

### Generalization analysis

The `Performance/Generalization/` folder contains QQP experiments for evaluating data-scale and cross-task generalization. QQP experiments report classification metrics such as Accuracy, F1, and AUC.

### Theoretical circuit analysis

The `Theoretical_Analysis/` folder contains scripts and result files for expressibility and entangling-capability analysis of different circuit configurations. These experiments quantify candidate and reference quantum circuit blocks independently of downstream datasets.

## Data

The benchmark datasets, GloVe embeddings, and pre-trained model checkpoints are not included in this repository.

Please see:

```text
data/README.md
```

for the expected dataset formats, file names, directory organization, and split information.

The experiments use public benchmark resources including:

```text
SICK-R
STS Benchmark
SemEval STS 2012-2016
QQP
GloVe 840B 300d
```

For reproducible use, explicitly passing paths is recommended.

Example:

```bash
python Baselines/SICK-R/QSTS_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt
```

The exact script names may differ depending on the final folder organization. Please check each script's `--help` message for supported arguments.

## Environment

A Python environment should include:

```text
pennylane
numpy
pandas
scipy
scikit-learn
torch
transformers
```

If a `requirements.txt` file is provided, install it with:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch version matching your CUDA environment.

## Running Experiments

Most scripts support command-line arguments for data paths, embedding paths, log directories, random seeds, and model-specific options.

### SICK-R QSTS

```bash
python Baselines/SICK-R/QSTS_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42
```

### STS-B QSTS

```bash
python Baselines/STS-B/QSTS_STSB.py \
  --train_file ./data/STS-B/train.jsonl \
  --val_file ./data/STS-B/validation.jsonl \
  --test_file ./data/STS-B/test.jsonl \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42
```

### STS-X LOO QSTS

```bash
python Baselines/STS-X/QSTS_STSX.py \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42
```

The STS-X scripts use leave-one-year-out data construction. Please follow `data/README.md` carefully to organize the required atomic STS files.

### QQP QSTS

```bash
python Performance/Generalization/QQP_QSTS.py \
  --qqp_train_file ./data/QQP/train.tsv \
  --qqp_dev_file ./data/QQP/dev.tsv \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --train_size 30000 \
  --seed 42
```

The exact QQP script name may differ depending on the final repository organization.

## Reproducibility Notes

For strict reproducibility:

1. Use the same processed data files.
2. Use the same GloVe file.
3. Use the same local pre-trained checkpoints.
4. Use the same random seeds.
5. Keep the same package versions where possible.
6. Verify that dataset split sizes match those described in `data/README.md`.

## Notes on File Naming

Some script names use shortened experimental labels:

```text
FT  = full fine-tuning
FZ  = frozen-encoder / frozen-head setting
A2A = all-to-all topology
QMLP_* = QFFN-QAOA and QFFN-HEA hybrid quantum baselines
```

These names are retained for consistency with the experimental scripts.
