# QSTS: Quantum Semantic Textual Similarity on Quantum Circuits

This repository provides the source code for the **Quantum Semantic Textual Similarity (QSTS)** model and all accompanying experimental scripts used in the paper. The model is implemented on parameterized quantum circuits (PQCs) and consists of three blocks: a quantum encoder block, a textual feature learning block, and an STS analysis block based on the SWAP test. The repository covers the full experimental pipeline reported in the paper, including benchmark comparison, effectiveness, scalability, robustness, and generalization analyses, together with theoretical circuit analyses of expressibility and entangling capability.

All commands below are intended to be executed from the repository root unless a subsection explicitly changes directory.

## Repository Structure

```text
.
├── Baselines/
│   ├── SICK-R/
│   │   ├── Our_Model/                # QSTS_SICK_R.py
│   │   ├── Word_Embedding/           # BoW / HashVec / GloVe
│   │   ├── Neural_Networks/          # CNN / SANN / GRU / BiGRU
│   │   ├── Quantum_Models/           # QMLP_Hybrid (QFFN-QAOA / QFFN-HEA) / QCNN_Hybrid
│   │   └── Large_Pretrained_Models/  # BERT, RoBERTa, SimCSE, Sentence-T5, E5
│   ├── STS-B/                        # same five subfolders (BERT, RoBERTa only for large models)
│   └── STS12-16/                     # same five subfolders, all under the LOO protocol
├── Performance/
│   ├── Effectiveness/                # Six-configuration circuit comparison on SICK-R
│   ├── Scalability/                  # Depth (D2, D3) and width (W4, W7) ablations
│   ├── Robustness/                   # Channel-level and per-gate noisy-inference scripts
│   └── Generalization/               # QQP paraphrase identification (data scale + cross task)
├── Theoretical_Analysis/
│   ├── exp/                          # Expressibility (KL divergence between PQC and Haar)
│   ├── ent/                          # Entangling capability (Meyer-Wallach measure)
│   ├── results_exp.csv               # Precomputed expressibility results
│   └── results_ent.csv               # Precomputed entangling-capability results
├── data/
│   └── README.md                     # Dataset preparation guide
├── LICENSE
├── README.md
└── requirements.txt
```

## Model Overview

The main QSTS configuration on the STS benchmarks uses **13 qubits** in total: 1 ancilla qubit for SWAP-test readout, and two 6-qubit registers (one per sentence) that load 64-dimensional amplitude-encoded sentence representations. The textual feature learning block adopts a sandwich-structured candidate ansatz **S(θ_b) · CB5 · S(θ_f)** under the circuit-block (CB) topology, with universal single-qubit rotations (Rz–Ry–Rz) and parameter-free CNOT entanglers. This configuration uses **72 trainable quantum parameters**.

The robustness and QQP generalization experiments use a smaller **9-qubit configuration** (1 ancilla + two 4-qubit registers, 16-dimensional amplitude encoding, 48 trainable quantum parameters) to keep noisy-circuit simulation tractable while preserving the same architecture. This 9-qubit setting corresponds to the W4 width configuration in `Performance/Scalability/SICK-R_W4.py`.

## Included Experiments

### Main QSTS experiments

QSTS is evaluated on seven public benchmarks:

```text
SICK-R
STS-B
STS-12, STS-13, STS-14, STS-15, STS-16   (under the leave-one-year-out protocol)
```

The main scripts use word embeddings, amplitude encoding, the candidate CB-topology textual feature learning block, and a SWAP-test-based STS analysis block whose ancilla `⟨Z⟩` expectation gives the similarity score `|⟨ψ|φ⟩|²`.

### Baselines

Three classical word-embedding baselines: `BoW`, `HashVec`, `GloVe`.

Four trainable classical neural-encoder baselines: `CNN`, `SANN` (self-attention neural network, single-head Transformer-style encoder), `GRU`, `BiGRU`.

Large pre-trained model baselines: `BERT` and `RoBERTa` (frozen-encoder + linear head, denoted `FH` or `FZ`, and full fine-tuning, denoted `FT`); `SimCSE` (frozen-head, full fine-tuning, and zero-shot) with both supervised and unsupervised checkpoints; `Sentence-T5` and `E5` in the zero-shot setting. SimCSE / Sentence-T5 / E5 baselines are provided on **SICK-R only**, in line with the extended-comparison study reported in the paper; STS-B and STS12-16 cover BERT and RoBERTa only.

Hybrid quantum baselines: `QMLP_Hybrid_*` implements **QFFN-QAOA** and **QFFN-HEA**, selectable via `--variant qaoa` or `--variant hea`. `QCNN_Hybrid_*` implements a quantum convolutional hybrid network. These baselines produce classical sentence vectors from PQC outputs and use cosine similarity for the final score, in contrast to QSTS, which evaluates similarity directly in quantum Hilbert space via the SWAP test.

### Effectiveness analysis (`Performance/Effectiveness/`)

Six circuit configurations are evaluated on SICK-R: the reference and candidate circuit families, each instantiated under the three topologies NN (nearest-neighbor), CB (circuit-block), and AA (all-to-all). The candidate CB configuration corresponds to the main QSTS model and is therefore evaluated through the main `QSTS_SICK_R.py` script; the remaining five configurations are provided here:

```text
SICK-R_NNref.py     SICK-R_NNQSTS.py
SICK-R_CBref.py     (CB-candidate = main QSTS, run via QSTS_SICK_R.py)
SICK-R_A2Aref.py    SICK-R_A2AQSTS.py
```

### Scalability analysis (`Performance/Scalability/`)

Depth and width ablations on SICK-R. Depth scripts stack 2 or 3 sandwich blocks (`SICK-R_D2.py`, `SICK-R_D3.py`; the default D1 setting is again the main QSTS model). Width scripts vary the per-sentence register width: `SICK-R_W4.py` (4 qubits / 16-dim) and `SICK-R_W7.py` (7 qubits / 128-dim); the default W6 (6 qubits / 64-dim) is again the main QSTS model. Trainable quantum parameter counts across these settings range from 48 (W4) to 216 (D3).

`SICK-R_W4.py` additionally produces the trained checkpoint required by the robustness scripts.

### Robustness analysis (`Performance/Robustness/`)

Two complementary noisy-inference studies on a 9-qubit QSTS configuration ("clean training, noisy inference"). The first sweeps four standard quantum channels (bit-flip, phase-flip, depolarizing, amplitude-damping) at noise levels `p ∈ {0.001, 0.01, 0.05, 0.1, 0.2}`, with the channel inserted between the textual feature learning block and the STS analysis block (`SICK-R_W4_QSTS_Noise.py`). The second is a hardware-motivated stress test that simultaneously injects single-qubit gate error, two-qubit gate error, and ancilla readout error, and ablates each source by toggling (`SICK-R_W4_QSTS_PerGateNoise.py`). Both scripts require a clean W4 checkpoint trained in advance from `Performance/Scalability/SICK-R_W4.py`.

### Generalization analysis (`Performance/Generalization/`)

QQP paraphrase identification at training sizes 30K / 50K / 100K on the QQP dev set (n = 40,430). The SWAP-test output of QSTS is used as the paraphrase score with a fixed decision threshold of 0.5. Reported metrics: Accuracy, F1, and AUC. Folder contents:

```text
QSTS_QQP.py              QSTS, 9-qubit / 16-dim, sandwich CB ansatz
CNN_QQP.py               classical CNN baseline
BiGRU_QQP.py             classical BiGRU baseline
SimCSE_ZeroShot_QQP.py   modern sentence-encoder baseline (zero-shot)
```

### Theoretical circuit analysis (`Theoretical_Analysis/`)

Quantifies expressibility (KL divergence between the PQC fidelity distribution and the Haar distribution, on a 6-qubit system, n_bin = 75, sample sizes 1500 / 2500 / 5000 / 7500) and entangling capability (Meyer-Wallach measure under uniform-random parameters). Six configurations are evaluated for expressibility (`Ref-NN, Ref-CB, Ref-AA, Cand-NN, Cand-CB, Cand-AA`), and the three candidate configurations for entangling capability. Precomputed summaries are shipped as `results_exp.csv` and `results_ent.csv`.

## Data Preparation

This repository does not redistribute third-party datasets, word-embedding files, or large pre-trained model checkpoints. See `data/README.md` for the expected file names, directory organization, required fields, score-normalization conventions, and split sizes. The experiments use the following public resources:

```text
SICK-R
STS Benchmark (STS-B)
SemEval STS 2012-2016
QQP (Quora Question Pairs)
```

Additionally, the GloVe-based baseline scripts (and any other script that takes a `--glove_file` argument) require the public GloVe 840B 300d file, which should be placed at the path passed via `--glove_file`.

Note on directory naming: the dataset directory is `data/STS-X/STS-12/...` (see `data/README.md`), while the corresponding code directory is `Baselines/STS12-16/`. The atomic STS files in `data/STS-X/` are deliberately *not* cumulative; the leave-one-year-out training pool is built explicitly via `--train_files` (see below).

## Environment

The experiments were run with **Python 3.10.14** on Linux x86_64. Install all pinned dependencies via:

```bash
pip install -r requirements.txt
```

Key packages (versions pinned in `requirements.txt`):

```text
pennylane==0.36.0          pennylane-lightning==0.36.0
torch==2.3.1               transformers==4.41.2
numpy==1.26.4              pandas==2.2.2
scipy==1.13.1              scikit-learn==1.5.0
```

PennyLane 0.36 with `pennylane-lightning` 0.36 is tested against Python 3.10; newer Python versions may require rebuilding `pennylane-lightning` from source. The default `torch==2.3.1` wheel ships with CUDA 12.1 support, but the experiments were executed on CPU nodes and GPU is not required (see Hardware below).

### Hardware

All simulations reported in the paper were executed on CPU partitions of the **National Supercomputer Center in Jinan**. Each compute node is configured as:

```text
CPU       : 2 × Intel Xeon Gold 6258R @ 2.7 GHz  (56 cores per node)
Memory    : 192 GB / 384 GB / 1.5 TB nodes
Peak FLOPs: 4.8 TFLOPS per node
Interconnect : InfiniBand HDR
Storage   : full-flash Lustre file system
```

PennyLane CPU simulation is sufficient for all experiments; GPU is not required. Long quantum-circuit experiments such as STS12-16 LOO QSTS and QQP-100K benefit from the larger-memory nodes.

## Local Pre-trained Checkpoints

Large pre-trained models are not redistributed. The corresponding scripts expect local HuggingFace-compatible checkpoint directories and use different argument names for different model families:

| Model family                | Argument          | Notes |
|-----------------------------|-------------------|-------|
| BERT / RoBERTa (FT, FZ/FH)  | `--model_path`    | Pass the model directory directly. |
| SimCSE (FT, FZ/FH, ZS)      | `--model_base_dir` + `--simcse_variant {sup, unsup}` | Resolved as `<model_base_dir>/{sup,unsup}-simcse-bert-base-uncased`. |
| Sentence-T5, E5 (ZS)        | `--model_base_dir` | Resolved as `<model_base_dir>/<canonical-model-dir>` (see each script). |

Each model directory should contain standard HuggingFace files (`config.json`, tokenizer files, model weights).

## Running the Main QSTS Experiments

### SICK-R QSTS

```bash
python Baselines/SICK-R/Our_Model/QSTS_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/SICK-R_QSTS
```

### STS-B QSTS

```bash
python Baselines/STS-B/Our_Model/QSTS_STSB.py \
  --train_file ./data/STS-B/train.jsonl \
  --val_file   ./data/STS-B/validation.jsonl \
  --test_file  ./data/STS-B/test.jsonl \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/STS-B_QSTS
```

### STS-12 to STS-16 QSTS (Leave-One-Year-Out)

For each target year Y ∈ {STS-12, STS-13, STS-14, STS-15, STS-16}, the official test split of year Y is held out as the test set, and the training pool is the union of the STS-12 official train split with the official test splits of the other four years. The `--train_files` argument **must be the five atomic files listed below**; do not pass the cumulative `train.jsonl` files of STS-13 / 14 / 15 / 16, which are concatenations of prior atomic files and would cause heavy duplication. The script defensively deduplicates and contamination-filters the pool, but the canonical recipe is to supply atomic files only.

Example: target year STS-12.

```bash
python Baselines/STS12-16/Our_Model/QSTS_STSX.py \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --train_files \
      ./data/STS-X/STS-12/train.jsonl \
      ./data/STS-X/STS-13/test.jsonl \
      ./data/STS-X/STS-14/test.jsonl \
      ./data/STS-X/STS-15/test.jsonl \
      ./data/STS-X/STS-16/test.jsonl \
  --test_file          ./data/STS-X/STS-12/test.jsonl \
  --leave_out_test_file ./data/STS-X/STS-12/test.jsonl \
  --dataset_name STS-12 \
  --seed 42 \
  --log_dir ./logs/STS-12_QSTS
```

For a target year Y other than STS-12, the `--train_files` list should contain `STS-12/train.jsonl` together with the official `test.jsonl` files of all non-target STS years; the target year's own `test.jsonl` must **not** appear in `--train_files`, and is passed only through `--test_file` and `--leave_out_test_file`. For example, when the target year is STS-13, the training pool should contain `STS-12/train.jsonl`, `STS-12/test.jsonl`, `STS-14/test.jsonl`, `STS-15/test.jsonl`, and `STS-16/test.jsonl`. Expected per-year split sizes after deduplication and contamination filtering are listed in `data/README.md`.

## Running Baselines

Baseline scripts are organized as:

```text
Baselines/<dataset>/Word_Embedding/
Baselines/<dataset>/Neural_Networks/
Baselines/<dataset>/Quantum_Models/
Baselines/<dataset>/Large_Pretrained_Models/
```

Most trainable baselines share a uniform command-line interface (dataset paths, `--glove_file` where applicable, `--seed` or `--seeds`, `--log_dir`).

Naming note: STS-B large-model scripts use the longer suffixes `_FineTune_STSB` / `_Frozen_STSB`, and STS-B hybrid quantum scripts use `_Hybrid_Baseline_STSB`; SICK-R uses `_FT` / `_FZ` and `_Hybrid_SICK_R`, while STS12-16 uses `_FineTune_STSX` / `_Frozen_STSX` and `_STSX`. The arguments are otherwise consistent across datasets.

### SICK-R baseline examples

Classical neural baseline (multi-seed):

```bash
python Baselines/SICK-R/Neural_Networks/CNN_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seeds 0 1 2 3 42 \
  --log_dir ./logs/CNN_SICK_R
```

Hybrid quantum baseline (QFFN-QAOA variant):

```bash
python Baselines/SICK-R/Quantum_Models/QMLP_Hybrid_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --variant qaoa \
  --seeds 0 1 2 3 42 \
  --log_dir ./logs/QMLP_QAOA_SICK_R
```

Pre-trained encoder, full fine-tuning:

```bash
python Baselines/SICK-R/Large_Pretrained_Models/BERT_FT_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --model_path /path/to/bert-base-uncased \
  --seeds 0 1 2 3 42 \
  --log_dir ./logs/BERT_FT_SICK_R
```

SimCSE supervised, frozen-head:

```bash
python Baselines/SICK-R/Large_Pretrained_Models/SimCSE_FZ_SICK_R.py \
  --data_file ./data/SICK-R/SICK.txt \
  --model_base_dir /path/to/simcse-checkpoints \
  --simcse_variant sup \
  --seeds 0 1 2 3 42 \
  --log_dir ./logs/SimCSE_sup_FZ_SICK_R
```

### STS-B baseline example

```bash
python Baselines/STS-B/Word_Embedding/GloVe_STSB.py \
  --train_file ./data/STS-B/train.jsonl \
  --val_file   ./data/STS-B/validation.jsonl \
  --test_file  ./data/STS-B/test.jsonl \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --log_dir ./logs/GloVe_STSB
```

### STS12-16 baseline example (LOO protocol)

**All** STS12-16 baseline scripts (word-embedding, neural, quantum hybrid, large-model) take the same LOO arguments as `QSTS_STSX.py`. Example: CNN baseline targeting STS-12.

```bash
python Baselines/STS12-16/Neural_Networks/CNN_STSX.py \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --train_files \
      ./data/STS-X/STS-12/train.jsonl \
      ./data/STS-X/STS-13/test.jsonl \
      ./data/STS-X/STS-14/test.jsonl \
      ./data/STS-X/STS-15/test.jsonl \
      ./data/STS-X/STS-16/test.jsonl \
  --test_file          ./data/STS-X/STS-12/test.jsonl \
  --leave_out_test_file ./data/STS-X/STS-12/test.jsonl \
  --dataset_name STS-12 \
  --seeds 0 1 2 3 42 \
  --log_dir ./logs/CNN_STSX_STS-12
```

For STS12-16 large-model baselines, additionally pass `--model_path /path/to/<checkpoint>` (BERT, RoBERTa). Use each script's `--help` for the complete argument list.

## Effectiveness Analysis

Each of the following scripts runs one of the five non-main configurations on SICK-R with the same arguments as the main QSTS script:

```bash
python Performance/Effectiveness/SICK-R_NNref.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/Eff_NNref
```

Replace `SICK-R_NNref.py` with `SICK-R_NNQSTS.py`, `SICK-R_CBref.py`, `SICK-R_A2Aref.py`, or `SICK-R_A2AQSTS.py` for the other configurations. The CB-candidate configuration is the main QSTS model, obtained from `Baselines/SICK-R/Our_Model/QSTS_SICK_R.py`.

## Scalability Analysis

```bash
# Width ablation (W4 = 16-dim / 4 qubits per sentence; W7 = 128-dim / 7 qubits)
python Performance/Scalability/SICK-R_W4.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/Scal_W4

python Performance/Scalability/SICK-R_W7.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/Scal_W7

# Depth ablation (D2 / D3 = 2 / 3 stacked sandwich blocks)
python Performance/Scalability/SICK-R_D2.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/Scal_D2
# (analogous for SICK-R_D3.py)
```

The W6 (default 64-dim, 6 qubits per sentence) and D1 (single sandwich block) baselines are again the main QSTS model.

`SICK-R_W4.py` writes a `*_best.pt` checkpoint under `--log_dir`; this file is the input to the robustness scripts below.

## Robustness Analysis

The robustness scripts perform noisy inference on a **clean-trained W4 QSTS checkpoint**. Both follow the same two-step workflow.

**Step 1 — train the W4 checkpoint** (if not already produced by the scalability run above):

```bash
python Performance/Scalability/SICK-R_W4.py \
  --data_file ./data/SICK-R/SICK.txt \
  --glove_file ./data/GloVe/glove.840B.300d.txt \
  --seed 42 \
  --log_dir ./logs/Scal_W4
# This writes ./logs/Scal_W4/<base_name>_best.pt
```

**Step 2 — noisy-inference evaluation.**

Channel-level sweep. Supported channels: `none`, `bitflip`, `phaseflip`, `depolarizing`, `amplitude_damping`. The paper sweeps `p ∈ {0.001, 0.01, 0.05, 0.1, 0.2}` per channel:

```bash
python Performance/Robustness/SICK-R_W4_QSTS_Noise.py \
  --checkpoint_path ./logs/Scal_W4/<base_name>_best.pt \
  --data_file ./data/SICK-R/SICK.txt \
  --noise_channel depolarizing \
  --noise_level 0.05 \
  --seed 42 \
  --log_dir ./logs/Noise_depolarizing_p0.05
```

Per-gate hardware-motivated stress test. Representative IBM device profiles (used in the paper for the "all combined" rows):

```text
Heron-class : --p_1q 0.0005 --p_2q 0.003  --p_readout 0.010
Eagle-class : --p_1q 0.002  --p_2q 0.008  --p_readout 0.015
```

To ablate a single error source, set the other two probabilities to `0.0` (e.g. `--p_1q 0.0005 --p_2q 0.0 --p_readout 0.0` for "1q only"). Example (all-combined Heron):

```bash
python Performance/Robustness/SICK-R_W4_QSTS_PerGateNoise.py \
  --checkpoint_path ./logs/Scal_W4/<base_name>_best.pt \
  --data_file ./data/SICK-R/SICK.txt \
  --p_1q 0.0005 \
  --p_2q 0.003 \
  --p_readout 0.010 \
  --profile_label all_combined_heron \
  --seed 42 \
  --log_dir ./logs/PerGate_Heron_all
```

Notes on the noise scripts' `--seed`: it is used only to select the matching W4 checkpoint by filename. The noisy-inference pass itself is deterministic, because `default.mixed` performs exact density-matrix evolution. To produce multi-seed robustness statistics, train multiple W4 checkpoints under different seeds and run the noise scripts on each.

The optional `--max_test_samples` flag in the per-gate script is intended for wall-clock timing pilots only; truncated runs are not comparable with full-test-set results.

## Generalization Analysis (QQP)

QSTS on QQP with stratified training subsets:

```bash
python Performance/Generalization/QSTS_QQP.py \
  --qqp_train_file ./data/QQP/train.tsv \
  --qqp_dev_file   ./data/QQP/dev.tsv \
  --glove_file     ./data/GloVe/glove.840B.300d.txt \
  --train_size 30000 \
  --seed 42 \
  --log_dir   ./logs/QQP_QSTS_30K \
  --cache_dir ./cache/QQP_30K
```

`--train_size` accepts `30000`, `50000`, `100000`, or `all`. The dev set is used as the held-out test set. The script automatically carves an internal validation split out of the selected training subset for early stopping (controlled by `--internal_val_ratio`), and persists the stratified subsample under `--cache_dir` for reuse across runs.

QQP baselines run with the same `--qqp_train_file / --qqp_dev_file / --train_size` interface:

```bash
python Performance/Generalization/CNN_QQP.py    --qqp_train_file ... --qqp_dev_file ... --glove_file ... --train_size 30000
python Performance/Generalization/BiGRU_QQP.py  --qqp_train_file ... --qqp_dev_file ... --glove_file ... --train_size 30000

# SimCSE zero-shot (no training data needed beyond the dev set)
python Performance/Generalization/SimCSE_ZeroShot_QQP.py \
  --qqp_dev_file   ./data/QQP/dev.tsv \
  --model_base_dir /path/to/simcse-checkpoints \
  --simcse_variant sup \
  --log_dir ./logs/QQP_SimCSE_sup_ZS
```

Reported metrics: Accuracy, F1, and AUC. For QSTS / CNN / BiGRU, paraphrase predictions are obtained with a fixed decision threshold of 0.5.

## Theoretical Circuit Analysis

Precomputed summary CSVs are shipped in the repository:

```text
Theoretical_Analysis/results_exp.csv     expressibility (KL divergence) for the six configurations
Theoretical_Analysis/results_ent.csv     entangling capability (Meyer-Wallach measure) for the three candidate configurations
```

The expressibility and entangling-capability scripts each import a sibling `circuits_pennylane` module, so they must be run from inside their own subdirectory:

```bash
cd Theoretical_Analysis/exp
python run_expressibility.py --out_dir ./output_exp --seed 0
```

```bash
cd Theoretical_Analysis/ent
python run_experiment_entangling.py --out_dir ./output_ent --seed 0
```

Both scripts write a `results.csv` and a `results.json` metadata file under the specified `--out_dir`. Each row reports an ansatz name (`Ref-NN / Ref-CB / Ref-AA / Cand-NN / Cand-CB / Cand-AA` for expressibility; the three candidate variants for entanglement), the sample size, and the corresponding summary statistics. The default settings reproduce the paper's protocol: 6-qubit register, parameters sampled uniformly from `[0, 2π]`, sample sizes `{1500, 2500, 5000, 7500}`, and `n_bin = 75` for the expressibility histogram.

## Outputs

By default, most trainable scripts write to `./logs` unless overridden by `--log_dir`. Typical artifacts:

```text
<base_name>.log         text training log: per-epoch validation metrics + final test-set Pearson r, Spearman ρ, MSE
<base_name>_best.pt     best-checkpoint weights selected by validation performance
```

The QQP experiments additionally use `--cache_dir` to persist stratified subsamples (`./cache_QQP/` by default), so that the same `--train_size` subset can be reused across seeds. The theoretical-analysis scripts write `results.csv` and `results.json` under `--out_dir`.

## Multi-seed Evaluation

The main results in the paper are reported as mean ± std over the seed set `{0, 1, 2, 3, 42}`. Scripts that accept `--seeds` (most classical neural, hybrid quantum, and pre-trained baselines) run the full sweep internally and emit a mean-and-std summary line. Scripts that accept only `--seed` (the main QSTS scripts and most LOO baselines) should be invoked once per seed; aggregate the per-seed Pearson and Spearman values from the resulting `*.log` files. Zero-shot sentence-encoder baselines (`SimCSE_ZeroShot_*`, `SentenceT5_ZeroShot_*`, `E5_ZeroShot_*`) are deterministic and are therefore evaluated once.

## Reproducibility Notes

For strict reproducibility:

1. Use the processed dataset files described in `data/README.md` and verify the split sizes listed there.
2. Use the same word-embedding file across runs that take `--glove_file` (the public GloVe 840B 300d file used by the corresponding baselines).
3. Use the same local HuggingFace pre-trained checkpoints (or pin them via `--model_path` / `--model_base_dir`).
4. Use the seed set `{0, 1, 2, 3, 42}` for multi-seed results, or `--seed 42` for single-seed runs.
5. Keep package versions consistent with `requirements.txt`.
6. Run commands from the repository root unless a subsection explicitly changes directory (currently only `Theoretical_Analysis/exp/` and `Theoretical_Analysis/ent/`).
7. Train the W4 checkpoint **before** running the robustness scripts, and pass its full path via `--checkpoint_path`.

## Notes on File Naming

Common shorthand used across script names:

```text
FT / FineTune    = full fine-tuning of the pre-trained encoder
FZ / Frozen / FH = frozen encoder with a trainable linear head
ZS / ZeroShot    = no trainable parameters; pre-trained encoder used off the shelf
NN / CB / AA     = nearest-neighbor / circuit-block / all-to-all qubit-connectivity topologies
ref              = reference circuit family (parameterized Rx, Rz + CRX entanglers)
QSTS / Cand      = candidate circuit family (universal Rz-Ry-Rz + parameter-free CNOT entanglers)
QMLP_*           = hybrid quantum feed-forward baselines (QFFN-QAOA, QFFN-HEA)
QCNN_*           = hybrid quantum convolutional baseline
D1 / D2 / D3     = 1 / 2 / 3 stacked sandwich blocks in the textual feature learning module
W4 / W6 / W7     = 4 / 6 / 7 qubits per sentence register (16- / 64- / 128-dim amplitude encoding)
STS-X / STSX     = STS-12 to STS-16 under the leave-one-year-out protocol
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
