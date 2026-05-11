

from typing import Dict, Iterable, List, Set, Tuple
import argparse
import math
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates import AmplitudeEmbedding
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
NUM_QUBITS = 9
EMBEDDING_DIM = 300
REDUCED_DIM = 16
BATCH_SIZE = 16
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EPS = 1e-12
NUM_QUANTUM_PARAMS = 48


W4_BASELINE_PEARSON_BY_SEED = {
    0:  0.816398,
    1:  0.797183,
    2:  0.791955,
    3:  0.756857,
    42: 0.807049,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tokenize_text(sentence: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", str(sentence))


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(file_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    data = pd.read_csv(data_path, sep="\t", keep_default_na=False)
    train_data = data[data["SemEval_set"] == "TRAIN"].reset_index(drop=True)
    val_data = data[data["SemEval_set"] == "TRIAL"].reset_index(drop=True)
    test_data = data[data["SemEval_set"] == "TEST"].reset_index(drop=True)
    return train_data, val_data, test_data


def build_vocab(data_sources: Iterable[pd.DataFrame]) -> Set[str]:
    vocab: Set[str] = set()
    for df in data_sources:
        for item in df.itertuples():
            vocab.update(tokenize_text(item.sentence_A))
            vocab.update(tokenize_text(item.sentence_B))
    return vocab


def build_word_to_idx(vocab: Set[str]) -> Dict[str, int]:
    
    clean_vocab = {w for w in vocab if w not in (PAD_TOKEN, UNK_TOKEN)}
    word_to_idx = {word: idx + 2 for idx, word in enumerate(sorted(clean_vocab))}
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[UNK_TOKEN] = 1
    return word_to_idx


class SentencePairDataset(Dataset):
    def __init__(self, data: pd.DataFrame, word_to_idx: Dict[str, int]):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.data = []
        for _, row in data.iterrows():
            sent1 = row["sentence_A"]
            sent2 = row["sentence_B"]
            label = (float(row["relatedness_score"]) - 1.0) / 4.0
            self.data.append((
                self.sentence_to_indices(sent1),
                self.sentence_to_indices(sent2),
                label,
            ))

    def sentence_to_indices(self, sentence: str) -> torch.Tensor:
        words = tokenize_text(sentence)
        if not words:
            words = [UNK_TOKEN]
        indices = [self.word_to_idx.get(w, self.unk_idx) for w in words]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate_fn(batch):
    sents1, sents2, labels = zip(*batch)
    sents1_padded = pad_sequence(sents1, batch_first=True, padding_value=0)
    sents2_padded = pad_sequence(sents2, batch_first=True, padding_value=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    return sents1_padded, sents2_padded, labels_tensor


dev_clean = qml.device("default.qubit", wires=NUM_QUBITS)
dev_noisy = qml.device("default.mixed", wires=NUM_QUBITS)


def single_u(params, wires=None):
    
    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def qsts_circuit_clean(params, wires=None):
    


    single_u(params[0:3],   wires=wires[0])
    single_u(params[3:6],   wires=wires[1])
    single_u(params[6:9],   wires=wires[2])
    single_u(params[9:12],  wires=wires[3])


    for _ in range(3):
        qml.CNOT(wires=[wires[3], wires[0]])
        qml.CNOT(wires=[wires[2], wires[3]])
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])


    single_u(params[12:15], wires=wires[0])
    single_u(params[15:18], wires=wires[1])
    single_u(params[18:21], wires=wires[2])
    single_u(params[21:24], wires=wires[3])


    single_u(params[24:27], wires=wires[4])
    single_u(params[27:30], wires=wires[5])
    single_u(params[30:33], wires=wires[6])
    single_u(params[33:36], wires=wires[7])


    for _ in range(3):
        qml.CNOT(wires=[wires[7], wires[4]])
        qml.CNOT(wires=[wires[6], wires[7]])
        qml.CNOT(wires=[wires[5], wires[6]])
        qml.CNOT(wires=[wires[4], wires[5]])


    single_u(params[36:39], wires=wires[4])
    single_u(params[39:42], wires=wires[5])
    single_u(params[42:45], wires=wires[6])
    single_u(params[45:48], wires=wires[7])


def _two_qubit_to_local_p(p_2q: float) -> float:
    
    if p_2q < 0.0 or p_2q > 1.0:
        raise ValueError(f"p_2q must be in [0, 1], got {p_2q}")
    return 1.0 - math.sqrt(1.0 - p_2q)


def _single_qubit_u_with_noise(params, wire, p_1q):
    
    qml.RZ(params[0], wires=wire)
    if p_1q > 0.0:
        qml.DepolarizingChannel(p_1q, wires=wire)
    qml.RY(params[1], wires=wire)
    if p_1q > 0.0:
        qml.DepolarizingChannel(p_1q, wires=wire)
    qml.RZ(params[2], wires=wire)
    if p_1q > 0.0:
        qml.DepolarizingChannel(p_1q, wires=wire)


def _cnot_with_noise(control, target, p_local_2q):
    
    qml.CNOT(wires=[control, target])
    if p_local_2q > 0.0:
        qml.DepolarizingChannel(p_local_2q, wires=control)
        qml.DepolarizingChannel(p_local_2q, wires=target)


def qsts_circuit_per_gate_noisy(params, wires, p_1q: float, p_local_2q: float):
    


    _single_qubit_u_with_noise(params[0:3],   wires[0], p_1q)
    _single_qubit_u_with_noise(params[3:6],   wires[1], p_1q)
    _single_qubit_u_with_noise(params[6:9],   wires[2], p_1q)
    _single_qubit_u_with_noise(params[9:12],  wires[3], p_1q)


    for _ in range(3):
        _cnot_with_noise(wires[3], wires[0], p_local_2q)
        _cnot_with_noise(wires[2], wires[3], p_local_2q)
        _cnot_with_noise(wires[1], wires[2], p_local_2q)
        _cnot_with_noise(wires[0], wires[1], p_local_2q)


    _single_qubit_u_with_noise(params[12:15], wires[0], p_1q)
    _single_qubit_u_with_noise(params[15:18], wires[1], p_1q)
    _single_qubit_u_with_noise(params[18:21], wires[2], p_1q)
    _single_qubit_u_with_noise(params[21:24], wires[3], p_1q)


    _single_qubit_u_with_noise(params[24:27], wires[4], p_1q)
    _single_qubit_u_with_noise(params[27:30], wires[5], p_1q)
    _single_qubit_u_with_noise(params[30:33], wires[6], p_1q)
    _single_qubit_u_with_noise(params[33:36], wires[7], p_1q)


    for _ in range(3):
        _cnot_with_noise(wires[7], wires[4], p_local_2q)
        _cnot_with_noise(wires[6], wires[7], p_local_2q)
        _cnot_with_noise(wires[5], wires[6], p_local_2q)
        _cnot_with_noise(wires[4], wires[5], p_local_2q)


    _single_qubit_u_with_noise(params[36:39], wires[4], p_1q)
    _single_qubit_u_with_noise(params[39:42], wires[5], p_1q)
    _single_qubit_u_with_noise(params[42:45], wires[6], p_1q)
    _single_qubit_u_with_noise(params[45:48], wires[7], p_1q)


@qml.transforms.merge_amplitude_embedding
@qml.qnode(dev_clean, interface="torch", diff_method=None)
def quantum_func_clean(inp1, inp2, params):
    
    qml.Hadamard(wires=0)
    AmplitudeEmbedding(features=inp1, wires=range(1, NUM_QUBITS // 2 + 1),
                       normalize=True, pad_with=0.0)
    AmplitudeEmbedding(features=inp2, wires=range(NUM_QUBITS // 2 + 1, NUM_QUBITS),
                       normalize=True, pad_with=0.0)
    qsts_circuit_clean(params, wires=range(1, NUM_QUBITS))
    for i in range(NUM_QUBITS // 2):
        qml.CSWAP(wires=[0, i + 1, i + 1 + NUM_QUBITS // 2])
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.transforms.merge_amplitude_embedding
@qml.qnode(dev_noisy, interface="torch", diff_method=None)
def quantum_func_per_gate_noisy(
    inp1,
    inp2,
    params,
    p_1q: float,
    p_local_2q: float,
    p_readout: float,
):
    
    qml.Hadamard(wires=0)
    AmplitudeEmbedding(features=inp1, wires=range(1, NUM_QUBITS // 2 + 1),
                       normalize=True, pad_with=0.0)
    AmplitudeEmbedding(features=inp2, wires=range(NUM_QUBITS // 2 + 1, NUM_QUBITS),
                       normalize=True, pad_with=0.0)

    qsts_circuit_per_gate_noisy(
        params,
        wires=range(1, NUM_QUBITS),
        p_1q=p_1q,
        p_local_2q=p_local_2q,
    )

    for i in range(NUM_QUBITS // 2):
        qml.CSWAP(wires=[0, i + 1, i + 1 + NUM_QUBITS // 2])
    qml.Hadamard(wires=0)


    if p_readout > 0.0:
        qml.BitFlip(p_readout, wires=0)

    return qml.expval(qml.PauliZ(0))


class QuantumNeuralNetwork(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
        quantum_param_seed: int,
    ):
        super().__init__()


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False

        self.linear = nn.Linear(embedding_dim, reduced_dim).double()


        quantum_rng = torch.Generator()
        quantum_rng.manual_seed(int(quantum_param_seed))
        self.params = nn.Parameter(
            torch.rand(NUM_QUANTUM_PARAMS, generator=quantum_rng, dtype=torch.float64)
            * 2 * math.pi
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        mask = mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        x = x * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / lengths

    @staticmethod
    def prepare_quantum_input(x: torch.Tensor) -> torch.Tensor:

        norm = torch.linalg.norm(x)
        if torch.isnan(norm) or torch.isinf(norm) or norm <= EPS:
            y = torch.zeros_like(x)
            y[0] = 1.0
            return y
        return x / (norm + EPS)

    def forward(
        self,
        x_text1: torch.Tensor,
        x_text2: torch.Tensor,
        p_1q: float = 0.0,
        p_2q: float = 0.0,
        p_readout: float = 0.0,
    ) -> torch.Tensor:
        

        embedded_text1 = self.embedding(x_text1)
        reduced_text1 = self.linear(embedded_text1)
        mask1 = x_text1 != 0
        text_avg1 = self.masked_mean(reduced_text1, mask1)

        embedded_text2 = self.embedding(x_text2)
        reduced_text2 = self.linear(embedded_text2)
        mask2 = x_text2 != 0
        text_avg2 = self.masked_mean(reduced_text2, mask2)

        use_noisy = (p_1q > 0.0) or (p_2q > 0.0) or (p_readout > 0.0)


        p_local_2q = _two_qubit_to_local_p(p_2q) if use_noisy else 0.0

        outputs = []
        batch_size = x_text1.size(0)
        for i in range(batch_size):
            inp1 = self.prepare_quantum_input(text_avg1[i])
            inp2 = self.prepare_quantum_input(text_avg2[i])
            if use_noisy:
                qout = quantum_func_per_gate_noisy(
                    inp1, inp2, self.params,
                    p_1q, p_local_2q, p_readout,
                ).reshape(1)
            else:
                qout = quantum_func_clean(inp1, inp2, self.params).reshape(1)
            outputs.append(qout)
        return torch.cat(outputs, dim=0)


def safe_pearsonr(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size < 2 or predictions.size < 2:
        return 0.0
    if np.allclose(targets, targets[0]) or np.allclose(predictions, predictions[0]):
        return 0.0
    value = pearsonr(targets, predictions)[0]
    return 0.0 if np.isnan(value) else float(value)


def safe_spearmanr(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size < 2 or predictions.size < 2:
        return 0.0
    if np.allclose(targets, targets[0]) or np.allclose(predictions, predictions[0]):
        return 0.0
    value = spearmanr(targets, predictions)[0]
    return 0.0 if np.isnan(value) else float(value)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    p_1q: float,
    p_2q: float,
    p_readout: float,
    progress_every: int = 50,
) -> Tuple[float, float, float]:
    
    model.eval()
    predictions, targets = [], []
    total_batches = len(data_loader)
    eval_loop_start = datetime.now()
    with torch.no_grad():
        for batch_idx, (sents1, sents2, labels) in enumerate(data_loader):
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            raw_outputs = model(
                sents1, sents2,
                p_1q=p_1q, p_2q=p_2q, p_readout=p_readout,
            )

            if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                raw_outputs = torch.nan_to_num(
                    raw_outputs, nan=0.0, posinf=1.0, neginf=0.0
                )

            outputs = raw_outputs.detach().cpu().numpy().astype(np.float64)
            predictions.extend(outputs.tolist())
            targets.extend(labels.numpy().astype(np.float64).tolist())

            if progress_every > 0 and (batch_idx + 1) % progress_every == 0:
                elapsed = (datetime.now() - eval_loop_start).total_seconds()
                done = batch_idx + 1
                rate = done / max(1.0, elapsed)
                eta = (total_batches - done) / max(1e-9, rate)
                print(
                    f"  [progress] batch {done}/{total_batches}  "
                    f"elapsed={elapsed:.0f}s  rate={rate:.2f} batch/s  "
                    f"ETA={eta:.0f}s",
                    flush=True,
                )

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    mse = float(mean_squared_error(targets, predictions))
    pearson = safe_pearsonr(targets, predictions)
    spearman = safe_spearmanr(targets, predictions)
    return mse, pearson, spearman


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Per-gate-noise robustness evaluation of a QSTS sandwich-CB^3 "
            "W4 (9-qubit) checkpoint trained on SICK-R.  Implements "
            "Protocol B (clean training, noisy inference) with "
            "gate-dependent depolarizing-based noise modelling for Layer 2 "
            "of the hardware-related analysis (ablation across three error "
            "sources: single-qubit gate, two-qubit gate, ancilla readout)."
        )
    )
    p.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the .pt file produced by SICK-R_W4_HW.py.",
    )
    p.add_argument(
        "--data_file", type=str, required=True,
        help="Path to the SICK-R tab-separated dataset file (e.g. SICK.txt "
             "with TRAIN/TRIAL/TEST splits in column 'SemEval_set').  "
             "Used only to rebuild word_to_idx; no GloVe file is needed "
             "because the embedding tensor is loaded directly from the "
             "checkpoint.",
    )
    p.add_argument(
        "--p_1q", type=float, required=True,
        help="Single-qubit gate depolarizing error rate.  Inserted after "
             "EACH RZ and RY inside the 16 single_u blocks of qsts_circuit "
             "(48 channels total).  Heron-class representative value: 5e-4. "
             "Eagle-class: 2e-3.  Use 0.0 to disable.",
    )
    p.add_argument(
        "--p_2q", type=float, required=True,
        help="Two-qubit gate target error rate (NOT the local rate).  "
             "Internally converted to a local single-qubit rate "
             "p_local = 1 - sqrt(1 - p_2q), which is then applied "
             "independently to control and target after EACH CNOT "
             "(48 channels total = 24 CNOTs * 2 wires).  "
             "Heron-class representative value: 3e-3.  Eagle-class: 8e-3.  "
             "Use 0.0 to disable.",
    )
    p.add_argument(
        "--p_readout", type=float, required=True,
        help="Ancilla readout BitFlip probability, applied immediately "
             "before the terminal PauliZ measurement on wire 0.  "
             "Heron-class representative value: 1e-2.  Eagle-class: 1.5e-2. "
             "Use 0.0 to disable.",
    )
    p.add_argument(
        "--profile_label", type=str, default="custom",
        help="Free-form label for this configuration, used only in log "
             "filenames and console output (e.g. 'sanity', 'p1q_only', "
             "'p2q_only', 'readout_only', 'all_combined_heron', "
             "'all_combined_eagle').",
    )
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument(
        "--seed", type=int, default=SEED,
        help="W4 training seed used to select the matching checkpoint.  "
             "Acts here purely as a checkpoint selector and as the seed "
             "for set_seed(); the per-gate-noisy inference pass is itself "
             "deterministic (default.mixed performs exact density-matrix "
             "evolution, no sampling), so re-running with the same "
             "(checkpoint, p_1q, p_2q, p_readout) but a different --seed "
             "value will produce byte-identical Pearson numbers."
    )
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--max_test_samples", type=int, default=0,
        help="If > 0, truncate the SICK-R test set to the first N samples "
             "BEFORE building the DataLoader.  Intended for fast "
             "wall-clock-timing pilots (e.g. --max_test_samples 200 to "
             "estimate the cost of a full per-gate-noisy run).  Set to 0 "
             "(default) to use the full 4906-sample test set.  Note: the "
             "Pearson reported under truncation is NOT comparable to the "
             "full-set baseline -- this flag is for timing only."
    )
    p.add_argument(
        "--progress_every", type=int, default=50,
        help="Print a one-line stdout progress heartbeat every N batches "
             "during the test-set evaluation loop.  Set to 0 to disable.",
    )
    return p


def _validate_args(p_1q: float, p_2q: float, p_readout: float) -> None:
    for name, value in (("--p_1q", p_1q), ("--p_2q", p_2q), ("--p_readout", p_readout)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def main() -> None:
    args = build_argparser().parse_args()
    _validate_args(args.p_1q, args.p_2q, args.p_readout)
    set_seed(args.seed)

    device = torch.device("cpu")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = (
        f"QSTS-W4_SICK_PerGateNoise_{args.profile_label}_"
        f"p1q{args.p_1q:.4e}_p2q{args.p_2q:.4e}_pro{args.p_readout:.4e}_"
        f"seed{args.seed}_{run_stamp}.log"
    )
    log_path = log_dir / log_name

    use_noisy = (args.p_1q > 0.0) or (args.p_2q > 0.0) or (args.p_readout > 0.0)
    p_local_2q = _two_qubit_to_local_p(args.p_2q) if args.p_2q > 0.0 else 0.0

    header_lines = [
        f"=== Per-gate noisy-inference run started at {run_stamp} ===",
        f"  - protocol         : Protocol B (clean training, noisy inference)",
        f"  - noise model      : per-gate depolarizing + ancilla readout BitFlip",
        f"  - variant          : W4 (9 qubits, sandwich CB^3, 48 quantum params)",
        f"  - checkpoint       : {args.checkpoint_path}",
        f"  - data file        : {args.data_file}",
        f"  - device (eval)    : "
            f"{'default.mixed (density matrix)' if use_noisy else 'default.qubit (state vector, sanity check)'}",
        f"  - qubits           : {NUM_QUBITS} (1 ancilla + 4 + 4)",
        f"  - reduced dim      : {REDUCED_DIM}",
        f"  - quantum params   : {NUM_QUANTUM_PARAMS} (sandwich CB^3, independent A/B)",
        f"  - profile label    : {args.profile_label}",
        f"  - p_1q             : {args.p_1q:.4e}  (after each RZ and RY; 48 channels)",
        f"  - p_2q (target)    : {args.p_2q:.4e}  (per CNOT)",
        f"  - p_local_2q       : {p_local_2q:.4e}  "
            f"(= 1 - sqrt(1 - p_2q); per CNOT, applied to control AND target; 48 channels)",
        f"  - p_readout        : {args.p_readout:.4e}  (BitFlip on wire 0 before PauliZ)",
        f"  - noise location   : inside qsts_circuit (per gate); "
            f"NOT on ancilla H or CSWAP",
        f"  - max_test_samples : {args.max_test_samples if args.max_test_samples > 0 else 'full (4906)'}",
        f"  - seed             : {args.seed}",
        f"  - batch size       : {args.batch_size}",
        f"  - device (host)    : {device}",
    ]
    for line in header_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")


    train_data, val_data, test_data = load_data(args.data_file)
    msg = (
        f"Data sizes — TRAIN: {len(train_data)}, TRIAL: {len(val_data)}, "
        f"TEST: {len(test_data)}"
    )
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


    vocab = build_vocab([train_data])
    word_to_idx = build_word_to_idx(vocab)
    vocab_size = len(word_to_idx)
    msg = f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}"
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


    val_vocab = build_vocab([val_data])
    test_vocab = build_vocab([test_data])
    val_oov = val_vocab - vocab
    test_oov = test_vocab - vocab
    union_vocab = build_vocab([train_data, val_data, test_data])
    diag = (
        f"Total word types across train, validation, and test: {len(union_vocab)}\n"
        f"  · Val   set has {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>\n"
        f"  · Test  set has {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>"
    )
    print(diag)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(diag + "\n")


    placeholder_embeddings = np.zeros((vocab_size, EMBEDDING_DIM), dtype=np.float64)
    quantum_param_seed = args.seed + 2024
    model = QuantumNeuralNetwork(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=placeholder_embeddings,
        quantum_param_seed=quantum_param_seed,
    ).to(device)


    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)


    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    msg = (
        f"Loaded checkpoint OK.  Missing keys: {list(missing)}.  "
        f"Unexpected keys: {list(unexpected)}."
    )
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


    if args.max_test_samples > 0:
        test_data_used = test_data.iloc[: args.max_test_samples].reset_index(drop=True)
        msg = (
            f"[TIMING PILOT] Using truncated test set: "
            f"{len(test_data_used)} samples (full set has {len(test_data)})."
        )
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        test_data_used = test_data

    test_dataset = SentencePairDataset(test_data_used, word_to_idx)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )


    eval_start = datetime.now()
    test_mse, test_pearson, test_spearman = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        p_1q=args.p_1q,
        p_2q=args.p_2q,
        p_readout=args.p_readout,
        progress_every=args.progress_every,
    )
    eval_end = datetime.now()
    eval_seconds = (eval_end - eval_start).total_seconds()


    final_lines = [
        f"Final Testing (SICK-R test, n={len(test_data_used)}):",
        f"  profile_label              : {args.profile_label}",
        f"  p_1q                       : {args.p_1q:.4e}",
        f"  p_2q (target)              : {args.p_2q:.4e}",
        f"  p_local_2q (used in QNode) : {p_local_2q:.4e}",
        f"  p_readout                  : {args.p_readout:.4e}",
        f"  Test MSE                   : {test_mse:.6f}",
        f"  Test Pearson               : {test_pearson:.6f}",
        f"  Test Spearman              : {test_spearman:.6f}",
        f"  Inference wall-clock (sec) : {eval_seconds:.1f}",
    ]
    if args.max_test_samples > 0:
        full_n = len(test_data)
        used_n = len(test_data_used)
        ratio = full_n / max(1, used_n)
        eta_full = eval_seconds * ratio
        final_lines.append(
            f"  [TIMING PILOT] extrapolation : "
            f"{eval_seconds:.0f}s on {used_n} samples "
            f"--> ~{eta_full:.0f}s ({eta_full/3600.0:.2f}h) on full {full_n} samples"
        )
    if not use_noisy and args.max_test_samples == 0:


        expected = W4_BASELINE_PEARSON_BY_SEED.get(args.seed)
        if expected is None:
            final_lines.append(
                f"  Sanity check               : seed={args.seed} not in "
                f"W4_BASELINE_PEARSON_BY_SEED; expected baseline unknown."
            )
        else:
            delta = test_pearson - expected
            final_lines.append(
                f"  Sanity check               : expected W4 seed={args.seed} clean "
                f"Test Pearson = {expected:.6f} (per W4 baseline log)"
            )
            final_lines.append(
                f"  Sanity delta               : Δ = {delta:+.6f} "
                f"(|Δ| should be ≲ 1e-3; cross-simulator drift is fp64 round-off)"
            )
    for line in final_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
