

from typing import Dict, Iterable, List, Optional, Set, Tuple
import argparse
import copy
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
import torch.optim as optim
from pennylane.templates import AmplitudeEmbedding
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
NUM_QUBITS = 9
EMBEDDING_DIM = 300
REDUCED_DIM = 16
NUM_EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 15
INTERNAL_VAL_RATIO = 0.1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EPS = 1e-12
PROB_CLAMP_EPS = 1e-7
NUM_QUANTUM_PARAMS = 48


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


def load_glove_embeddings(
    glove_file: str,
    vocab: Set[str],
    embedding_dim: int,
) -> Tuple[Dict[str, int], np.ndarray]:
    clean_vocab = {w for w in vocab if w not in (PAD_TOKEN, UNK_TOKEN)}
    word_to_idx = {word: idx + 2 for idx, word in enumerate(sorted(clean_vocab))}
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[UNK_TOKEN] = 1

    embeddings = np.random.uniform(
        -0.25, 0.25, (len(word_to_idx), embedding_dim)
    ).astype(np.float64)
    embeddings[word_to_idx[PAD_TOKEN]] = 0.0

    hits = 0
    glove_path = Path(glove_file)
    if not glove_path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove_file}")

    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split()
            if len(values) != embedding_dim + 1:
                continue
            word = values[0]
            if word not in word_to_idx:
                continue
            try:
                vector = np.asarray(values[1:], dtype=np.float64)
            except ValueError:
                continue
            embeddings[word_to_idx[word]] = vector
            hits += 1

    vocab_size = len(word_to_idx) - 2
    oov = vocab_size - hits
    print(f"Vocabulary size: {vocab_size}")
    print(f"GloVe hits: {hits}")
    print(f"GloVe OOV: {oov}")
    if vocab_size > 0:
        print(f"GloVe hit rate: {hits / vocab_size:.4f}")
    else:
        print("GloVe hit rate: 0.0000")

    return word_to_idx, embeddings


_QQP_FIELD_ALIASES = {
    "question1":    ["question1", "sentence1"],
    "question2":    ["question2", "sentence2"],
    "is_duplicate": ["is_duplicate", "label"],
}


def _load_qqp_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"QQP file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        keep_default_na=False,
        na_values=[""],
        dtype=str,
        on_bad_lines="skip",
        engine="python",
        quoting=3,
    )


    rename_map: Dict[str, str] = {}
    for canonical, aliases in _QQP_FIELD_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    needed = {"question1", "question2", "is_duplicate"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"QQP file {path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}.  Accepted aliases per "
            f"canonical field: {_QQP_FIELD_ALIASES}."
        )

    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).reset_index(drop=True)
    df = df[
        df["is_duplicate"].astype(str).str.strip().isin({"0", "1"})
    ].reset_index(drop=True)
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    df["question1"] = df["question1"].astype(str)
    df["question2"] = df["question2"].astype(str)
    return df[["question1", "question2", "is_duplicate"]]


def stratified_subsample(
    df: pd.DataFrame,
    target_size: int,
    label_col: str,
    seed: int,
) -> pd.DataFrame:
    
    n_total = len(df)
    if target_size >= n_total:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    chunks = []
    classes = sorted(df[label_col].unique().tolist())
    cumulative = 0
    for k, cls in enumerate(classes):
        cls_df = df[df[label_col] == cls]
        cls_frac = len(cls_df) / n_total
        if k < len(classes) - 1:
            n_take = int(round(cls_frac * target_size))
        else:

            n_take = target_size - cumulative
        n_take = max(0, min(n_take, len(cls_df)))
        cumulative += n_take
        chosen = cls_df.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1)))
        chunks.append(chosen)

    out = pd.concat(chunks, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def stratified_split(
    df: pd.DataFrame,
    val_ratio: float,
    label_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = np.random.default_rng(seed)
    train_chunks, val_chunks = [], []
    classes = sorted(df[label_col].unique().tolist())
    for cls in classes:
        cls_df = df[df[label_col] == cls].sample(
            frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))
        ).reset_index(drop=True)
        n_val = int(round(len(cls_df) * val_ratio))
        n_val = max(1, min(n_val, len(cls_df) - 1))
        val_chunks.append(cls_df.iloc[:n_val])
        train_chunks.append(cls_df.iloc[n_val:])

    train_inner = pd.concat(train_chunks, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    val_inner = pd.concat(val_chunks, ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    return train_inner, val_inner


def build_vocab(data_sources: Iterable[pd.DataFrame]) -> Set[str]:
    
    vocab: Set[str] = set()
    for df in data_sources:
        for q1, q2 in zip(df["question1"].tolist(), df["question2"].tolist()):
            vocab.update(tokenize_text(q1))
            vocab.update(tokenize_text(q2))
    return vocab


class QQPPairDataset(Dataset):
    def __init__(self, data: pd.DataFrame, word_to_idx: Dict[str, int]):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.data = []

        for _, row in data.iterrows():
            sent1 = row["question1"]
            sent2 = row["question2"]

            label = float(int(row["is_duplicate"]))
            sent1_indices = self.sentence_to_indices(sent1)
            sent2_indices = self.sentence_to_indices(sent2)
            self.data.append((sent1_indices, sent2_indices, label))

    def sentence_to_indices(self, sentence: str) -> torch.Tensor:
        words = tokenize_text(sentence)
        if not words:
            words = [UNK_TOKEN]
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]
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


dev = qml.device("default.qubit", wires=NUM_QUBITS)


def single_u(params, wires=None):
    
    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def qsts_circuit(params, wires=None):
    


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


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_func(inp1, inp2, params):
    

    AmplitudeEmbedding(
        features=inp1,
        wires=range(1, NUM_QUBITS // 2 + 1),
        normalize=True,
        pad_with=0.0,
    )
    AmplitudeEmbedding(
        features=inp2,
        wires=range(NUM_QUBITS // 2 + 1, NUM_QUBITS),
        normalize=True,
        pad_with=0.0,
    )

    qml.Hadamard(wires=0)
    qsts_circuit(params, wires=range(1, NUM_QUBITS))
    for i in range(NUM_QUBITS // 2):
        qml.CSWAP(wires=[0, i + 1, i + 1 + NUM_QUBITS // 2])
    qml.Hadamard(wires=0)
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

        self.quantum = quantum_func

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

    def forward(self, x_text1: torch.Tensor, x_text2: torch.Tensor) -> torch.Tensor:
        embedded_text1 = self.embedding(x_text1)
        reduced_text1 = self.linear(embedded_text1)
        mask1 = x_text1 != 0
        text_avg1 = self.masked_mean(reduced_text1, mask1)

        embedded_text2 = self.embedding(x_text2)
        reduced_text2 = self.linear(embedded_text2)
        mask2 = x_text2 != 0
        text_avg2 = self.masked_mean(reduced_text2, mask2)

        outputs = []
        batch_size = x_text1.size(0)
        for i in range(batch_size):
            inp1 = self.prepare_quantum_input(text_avg1[i])
            inp2 = self.prepare_quantum_input(text_avg2[i])
            quantum_output = self.quantum(inp1, inp2, self.params).reshape(1)
            outputs.append(quantum_output)
        return torch.cat(outputs, dim=0)


def safe_accuracy(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    if targets.size == 0 or probs.size == 0:
        return 0.0
    preds = (probs >= threshold).astype(np.int64)
    return float(accuracy_score(targets.astype(np.int64), preds))


def safe_f1(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    if targets.size == 0 or probs.size == 0:
        return 0.0
    preds = (probs >= threshold).astype(np.int64)
    targets_int = targets.astype(np.int64)

    if (preds == 0).all() and (targets_int == 0).all():
        return 0.0
    try:
        return float(f1_score(targets_int, preds, pos_label=1, zero_division=0))
    except ValueError:
        return 0.0


def safe_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    if targets.size < 2 or probs.size < 2:
        return 0.5
    targets_int = targets.astype(np.int64)
    if len(np.unique(targets_int)) < 2:

        return 0.5
    try:
        return float(roc_auc_score(targets_int, probs))
    except ValueError:
        return 0.5


def _collect_outputs(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    
    model.eval()
    all_probs: List[float] = []
    all_targets: List[float] = []
    bce = nn.BCELoss(reduction="sum")
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for sents1, sents2, labels in data_loader:
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            labels = labels.to(device)

            raw_outputs = model(sents1, sents2)


            if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                raw_outputs = torch.nan_to_num(
                    raw_outputs, nan=0.5, posinf=1.0, neginf=0.0
                )


            probs = torch.clamp(
                raw_outputs, min=PROB_CLAMP_EPS, max=1.0 - PROB_CLAMP_EPS
            )

            loss_sum = bce(probs, labels)
            total_loss += float(loss_sum.item())
            total_n += labels.numel()

            all_probs.extend(probs.detach().cpu().numpy().astype(np.float64).tolist())
            all_targets.extend(labels.detach().cpu().numpy().astype(np.float64).tolist())

    probs_np = np.asarray(all_probs, dtype=np.float64)
    targets_np = np.asarray(all_targets, dtype=np.float64)
    bce_mean = total_loss / max(total_n, 1)
    return bce_mean, probs_np, targets_np


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
    
    bce_mean, probs_np, targets_np = _collect_outputs(model, data_loader, device)
    acc = safe_accuracy(targets_np, probs_np, threshold=0.5)
    f1 = safe_f1(targets_np, probs_np, threshold=0.5)
    auc = safe_auc(targets_np, probs_np)
    return bce_mean, acc, f1, auc


def find_best_f1_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    n_grid: int = 199,
) -> Tuple[float, float]:
    
    if targets.size == 0 or probs.size == 0:
        return 0.5, 0.0
    targets_int = targets.astype(np.int64)
    if len(np.unique(targets_int)) < 2:
        return 0.5, 0.0

    grid = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        preds = (probs >= t).astype(np.int64)
        try:
            f1 = float(f1_score(targets_int, preds, pos_label=1, zero_division=0))
        except ValueError:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    if best_f1 < 0.0:
        return 0.5, 0.0
    return best_t, best_f1


def evaluate_at_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    
    acc = safe_accuracy(targets, probs, threshold=threshold)
    f1 = safe_f1(targets, probs, threshold=threshold)
    return acc, f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    num_epochs: int,
    patience: int,
    log_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    grad_clip_max_norm: float = 1.0,
):
    
    best_val_f1 = float("-inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state: Optional[Dict] = None

    total_nan_batches = 0
    total_batches_seen = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        valid_batches_this_epoch = 0
        nan_batches_this_epoch = 0

        for sents1, sents2, labels in train_loader:
            total_batches_seen += 1
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            raw_outputs = model(sents1, sents2)


            outputs = torch.clamp(
                raw_outputs, min=PROB_CLAMP_EPS, max=1.0 - PROB_CLAMP_EPS
            )
            loss = criterion(outputs, labels)


            if (torch.isnan(loss) or torch.isinf(loss)
                    or torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any()):
                optimizer.zero_grad(set_to_none=True)
                nan_batches_this_epoch += 1
                total_nan_batches += 1
                continue

            loss.backward()


            bad_grad = False
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        bad_grad = True
                        break
            if bad_grad:
                optimizer.zero_grad(set_to_none=True)
                nan_batches_this_epoch += 1
                total_nan_batches += 1
                continue

            if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip_max_norm,
                )
            optimizer.step()
            total_loss += float(loss.item())
            valid_batches_this_epoch += 1


        if valid_batches_this_epoch == 0:
            msg = (
                f"Epoch {epoch + 1}/{num_epochs}: ALL {nan_batches_this_epoch} "
                f"batches produced NaN/Inf loss. Training cannot proceed — "
                f"aborting early and falling back to best-so-far checkpoint."
            )
            print(msg)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            break

        val_bce, val_acc, val_f1, val_auc = evaluate_model(model, val_loader, device)
        scheduler.step(val_f1)
        train_loss = total_loss / valid_batches_this_epoch

        current_lrs = ", ".join(f"{group['lr']:.6g}" for group in optimizer.param_groups)
        nan_tag = (
            f", NaN batches: {nan_batches_this_epoch}/"
            f"{nan_batches_this_epoch + valid_batches_this_epoch}"
            if nan_batches_this_epoch > 0 else ""
        )
        message = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {train_loss:.6f}, "
            f"Val BCE: {val_bce:.6f}, Val Acc: {val_acc:.6f}, "
            f"Val F1: {val_f1:.6f}, Val AUC: {val_auc:.6f}, "
            f"LRs: [{current_lrs}]{nan_tag}"
        )
        print(message)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            with log_path.open("a", encoding="utf-8") as f:
                f.write("Early stopping triggered.\n")
            break

    if best_model_state is None:
        raise RuntimeError(
            "No valid model state was saved during training. Either the very "
            "first epoch NaN'd out entirely, or training never ran an "
            "evaluation step."
        )

    model.load_state_dict(best_model_state)


    nan_ratio = (total_nan_batches / total_batches_seen) if total_batches_seen else 0.0
    summary_lines = [
        f"Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.6f}",
        f"Run health: {total_nan_batches}/{total_batches_seen} batches "
        f"produced NaN/Inf ({nan_ratio:.4%}).",
    ]
    if total_nan_batches > 0:
        if nan_ratio < 0.01:
            summary_lines.append(
                "    -> NaN ratio < 1%: run is clean, checkpoint trustworthy."
            )
        elif nan_ratio < 0.05:
            summary_lines.append(
                "    -> NaN ratio 1-5%: mild numerical instability, "
                "checkpoint still usable but report with caution."
            )
        else:
            summary_lines.append(
                "    -> NaN ratio >= 5%: significant instability, consider "
                "this seed as a stability outlier in the paper."
            )
    for line in summary_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")


def parse_train_size(value: str) -> int:
    
    if value is None:
        raise argparse.ArgumentTypeError("--train_size is required")
    v = value.strip().lower()
    if v == "all":
        return 10**9
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--train_size must be an integer or 'all', got '{value}'"
        )
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"--train_size must be positive, got {n}"
        )
    return n


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QSTS on QQP (9-qubit / 16-dim, sandwich CB ansatz).",
    )
    parser.add_argument(
        "--qqp_train_file", type=str, required=True,
        help="Path to GLUE QQP train.tsv.",
    )
    parser.add_argument(
        "--qqp_dev_file", type=str, required=True,
        help="Path to GLUE QQP dev.tsv (used as held-out test set).",
    )
    parser.add_argument(
        "--glove_file", type=str, required=True,
        help="Path to glove.840B.300d.txt.",
    )
    parser.add_argument(
        "--train_size", type=parse_train_size, required=True,
        help="Training subset size: 30000, 50000, 100000, or 'all'.",
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default="./cache_QQP",
                        help="Where to persist stratified subsamples for reuse.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--num_epochs", type=int, default=-1,
        help="Number of training epochs. -1 = auto (80, or 25 in 'all' mode).",
    )
    parser.add_argument(
        "--patience", type=int, default=-1,
        help="Early-stopping patience. -1 = auto (15, or 8 in 'all' mode).",
    )
    parser.add_argument(
        "--internal_val_ratio", type=float, default=INTERNAL_VAL_RATIO,
        help="Fraction of the train subset used for internal validation / "
             "early stopping.  (Strictly disjoint from the QQP dev test set.)",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Max gradient L2 norm for clipping. Set to 0 or negative to disable.",
    )
    return parser


def resolve_size_label(target_size: int, n_train_full: int) -> str:
    
    if target_size >= n_train_full:
        return "all"
    return str(target_size)


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    topology_name = "CB"
    optimizer_name = "adam_plateau"
    dataset_name = "QQP"
    variant_tag = "qsts"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    train_full_df = _load_qqp_tsv(Path(args.qqp_train_file))
    dev_full_df = _load_qqp_tsv(Path(args.qqp_dev_file))
    n_train_full = len(train_full_df)
    n_dev_full = len(dev_full_df)
    train_pos_frac = float(train_full_df["is_duplicate"].mean())
    dev_pos_frac = float(dev_full_df["is_duplicate"].mean())

    size_label = resolve_size_label(args.train_size, n_train_full)
    is_all_mode = (args.train_size >= n_train_full)


    if args.num_epochs == -1:
        num_epochs = 25 if is_all_mode else NUM_EPOCHS
    else:
        num_epochs = args.num_epochs
    if args.patience == -1:
        patience = 8 if is_all_mode else PATIENCE
    else:
        patience = args.patience


    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"QSTS_{dataset_name}_{topology_name}_{optimizer_name}_"
        f"{NUM_QUBITS}qubits_{variant_tag}_size{size_label}_"
        f"seed{args.seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"

    header = (
        f"=== Run started at {run_stamp} ===\n"
        f"variant={variant_tag}\n"
        f"  - dataset         : QQP (GLUE)\n"
        f"  - QQP train       : {n_train_full} pairs (pos frac {train_pos_frac:.4f})\n"
        f"  - QQP dev (test)  : {n_dev_full} pairs  (pos frac {dev_pos_frac:.4f})\n"
        f"  - train subset    : size={size_label}  (stratified, seed={args.seed})\n"
        f"  - inner val ratio : {args.internal_val_ratio} of the train subset (stratified)\n"
        f"  - embedding layer : frozen (requires_grad=False)\n"
        f"  - vocabulary      : (strictly inductive, label-free,\n"
        f"                       fixed across all training-size sweeps)\n"
        f"  - qubits          : {NUM_QUBITS} (1 ancilla + 4 + 4)\n"
        f"  - reduced dim     : {REDUCED_DIM}\n"
        f"  - ansatz          : sandwich  U(theta1,theta2) = S(theta2) . C_{topology_name}^3 . S(theta1)\n"
        f"  - entangling      : 3 layers of {topology_name} CNOTs per block (12 CNOTs/block,\n"
        f"                       matched to one AA layer on 4 qubits)\n"
        f"  - quantum params  : {NUM_QUANTUM_PARAMS} "
        f"(Block A: 12+12 for front/back S layers; Block B: 12+12; independent A/B)\n"
        f"  - task            : binary paraphrase identification\n"
        f"  - output          : SWAP-test overlap score in [0,1] used directly as duplicate score\n"
        f"  - loss            : BCELoss on clamped score (eps={PROB_CLAMP_EPS})\n"
        f"  - early stop      : val F1 (binary, pos_label=1) at threshold 0.5\n"
        f"  - eval metrics    : Accuracy, F1, AUC-ROC; reported at BOTH threshold 0.5\n"
        f"                       and at the F1-optimal threshold tuned on the internal val set\n"
        f"seed={args.seed}, batch_size={args.batch_size}, num_epochs={num_epochs}, "
        f"patience={patience}, grad_clip={args.grad_clip}, "
        f"internal_val_ratio={args.internal_val_ratio}, all_mode={is_all_mode}, "
        f"device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


    cache_path = (
        cache_dir
        / f"qqp_train_subsample_size{size_label}_seed{args.seed}_nfull{n_train_full}.tsv"
    )

    expected_size = n_train_full if is_all_mode else args.train_size

    if cache_path.is_file():
        train_subset = pd.read_csv(
            cache_path, sep="\t", keep_default_na=False, dtype=str, quoting=3,
        )
        train_subset["is_duplicate"] = train_subset["is_duplicate"].astype(int)


        if len(train_subset) != expected_size:
            raise RuntimeError(
                f"Cached training subsample at {cache_path} has "
                f"{len(train_subset)} rows but the current configuration "
                f"requires {expected_size} rows (train_size={size_label}, "
                f"seed={args.seed}, n_train_full={n_train_full}).  Refusing "
                f"to use a stale cache.  Delete the file and re-run to "
                f"regenerate."
            )
        msg = f"Loaded cached training subsample from {cache_path} ({len(train_subset)} rows)."
    else:
        train_subset = stratified_subsample(
            train_full_df,
            target_size=args.train_size,
            label_col="is_duplicate",
            seed=args.seed,
        )
        train_subset.to_csv(cache_path, sep="\t", index=False, quoting=3)
        msg = f"Wrote new training subsample to {cache_path} ({len(train_subset)} rows)."
    print(msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


    train_inner, val_inner = stratified_split(
        train_subset,
        val_ratio=args.internal_val_ratio,
        label_col="is_duplicate",
        seed=args.seed,
    )
    split_msg = (
        f"Train/Val(internal)/Test sizes: "
        f"{len(train_inner)} / {len(val_inner)} / {n_dev_full}.  "
        f"Inner-train pos frac: {float(train_inner['is_duplicate'].mean()):.4f}; "
        f"Inner-val pos frac:   {float(val_inner['is_duplicate'].mean()):.4f}; "
        f"Test (QQP dev) pos frac: {dev_pos_frac:.4f}."
    )
    print(split_msg)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(split_msg + "\n")


    vocab = build_vocab([train_full_df])
    word_to_idx, pretrained_embeddings = load_glove_embeddings(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)
    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}"
    )
    print(glove_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")


    train_dataset = QQPPairDataset(train_inner, word_to_idx)
    val_dataset = QQPPairDataset(val_inner, word_to_idx)
    test_dataset = QQPPairDataset(dev_full_df, word_to_idx)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )


    quantum_param_seed = args.seed + 2024

    model = QuantumNeuralNetwork(
        vocab_size,
        EMBEDDING_DIM,
        REDUCED_DIM,
        pretrained_embeddings,
        quantum_param_seed=quantum_param_seed,
    ).to(device)


    criterion = nn.BCELoss()

    optimizer = optim.Adam(
        [
            {"params": model.linear.parameters(), "lr": 1e-3, "weight_decay": 1e-5},
            {"params": [model.params],            "lr": 1e-3, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


    scheduler_patience = 3 if is_all_mode else 10
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
        threshold=1e-4,
        min_lr=[1e-5, 1e-5],
    )


    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=patience,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        device=device,
        grad_clip_max_norm=args.grad_clip,
    )


    val_bce_final, val_probs_final, val_targets_final = _collect_outputs(
        model, val_loader, device
    )
    test_bce, test_probs, test_targets = _collect_outputs(
        model, test_loader, device
    )


    test_acc_05, test_f1_05 = evaluate_at_threshold(test_targets, test_probs, 0.5)
    test_auc = safe_auc(test_targets, test_probs)


    best_thr_val, best_f1_val = find_best_f1_threshold(val_targets_final, val_probs_final)
    test_acc_tuned, test_f1_tuned = evaluate_at_threshold(
        test_targets, test_probs, best_thr_val
    )

    final_lines = [
        f"Final Testing (QQP dev, n={n_dev_full}):",
        f"  BCE                        : {test_bce:.6f}",
        f"  AUC-ROC (threshold-free)   : {test_auc:.6f}",
        f"  -- Protocol (a) threshold = 0.5  ({'GLUE-clean':>12s})",
        f"     Accuracy                : {test_acc_05:.6f}",
        f"     F1 (pos_label=1)        : {test_f1_05:.6f}",
        f"  -- Protocol (b) threshold tuned on internal val, applied to dev",
        f"     Best threshold on val   : {best_thr_val:.4f}  (val F1 at this thr: {best_f1_val:.6f})",
        f"     Accuracy on dev @ best  : {test_acc_tuned:.6f}",
        f"     F1 on dev @ best        : {test_f1_tuned:.6f}",
    ]
    for line in final_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in final_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
