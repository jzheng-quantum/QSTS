from typing import Dict, Iterable, List, Set, Tuple
import argparse
import copy
import hashlib
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from pennylane.templates import AmplitudeEmbedding
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


SEED = 42
NUM_QUBITS = 13
EMBEDDING_DIM = 300
REDUCED_DIM = 64
NUM_EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 25
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EPS = 1e-12


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


def load_glove_embeddings(glove_file: str, vocab: Set[str], embedding_dim: int) -> Tuple[Dict[str, int], np.ndarray]:
    clean_vocab = {w for w in vocab if w not in (PAD_TOKEN, UNK_TOKEN)}
    word_to_idx = {word: idx + 2 for idx, word in enumerate(sorted(clean_vocab))}
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[UNK_TOKEN] = 1

    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embedding_dim)).astype(np.float64)
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
    print(f"GloVe hit rate: {hits / vocab_size:.4f}" if vocab_size > 0 else "GloVe hit rate: 0.0000")

    return word_to_idx, embeddings


def load_jsonl(file_path: str) -> List[Dict]:
    
    data_path = Path(file_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    records: List[Dict] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def md5_file(file_path: str) -> str:
    
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sentence_for_key(sentence: str) -> str:
    
    return " ".join(str(sentence).strip().split())


def make_pair_key(
    row: Dict,
    normalize_order: bool = True,
) -> Tuple[str, str]:
    
    s1 = normalize_sentence_for_key(row["sentence1"])
    s2 = normalize_sentence_for_key(row["sentence2"])
    if normalize_order and s2 < s1:
        s1, s2 = s2, s1
    return s1, s2


def load_data(train_file: str, test_file: str) -> Tuple[List[Dict], List[Dict]]:
    
    train_records = load_jsonl(train_file)
    test_records = load_jsonl(test_file)
    return train_records, test_records


def load_data_loo(
    train_files: List[str],
    test_file: str,
    leave_out_test_file: str,
) -> Tuple[List[Dict], List[Dict], Dict]:
    


    cumulative_train_pattern = re.compile(
        r"STS-(13|14|15|16).*[/\\]train\.jsonl"
    )
    for path in train_files:
        path_str = str(path)
        if cumulative_train_pattern.search(path_str):
            raise ValueError(
                "Refusing to load a cumulative STS train.jsonl as an "
                "atomic LOO training source. The path\n"
                f"    {path_str}\n"
                "matches a SemEval-{13,14,15,16}/.../train.jsonl file. "
                "These files are not atomic; they already include data "
                "from earlier years' test sets. Pass atomic files only:\n"
                "  - STS-12/Data_STS_12/train.jsonl/train.jsonl  (atomic)\n"
                "  - STS-X/Data_STS_X/test.jsonl/test.jsonl       (atomic, X != target year)"
            )


    per_file_records: List[List[Dict]] = []
    per_file_meta: List[Tuple[str, int, str]] = []
    raw_total = 0
    for path in train_files:
        records = load_jsonl(path)
        per_file_records.append(records)
        per_file_meta.append((path, len(records), md5_file(path)))
        raw_total += len(records)


    after_intra_file_dedup = 0
    intra_dedup_records: List[List[Dict]] = []
    intra_file_conflict_pairs = 0
    for records in per_file_records:
        seen_in_file: Dict[Tuple[str, str], float] = {}
        deduped: List[Dict] = []
        for r in records:
            pair = make_pair_key(r, normalize_order=True)
            this_score = float(r["score"])
            if pair in seen_in_file:


                kept_score = seen_in_file[pair]
                if round(kept_score, 6) != round(this_score, 6):
                    intra_file_conflict_pairs += 1
                continue
            seen_in_file[pair] = this_score
            deduped.append(r)
        intra_dedup_records.append(deduped)
        after_intra_file_dedup += len(deduped)


    seen_pair_to_score: Dict[Tuple[str, str], float] = {}
    deduped_train: List[Dict] = []
    cross_file_conflict_pairs = 0
    for records in intra_dedup_records:
        for r in records:
            pair = make_pair_key(r, normalize_order=True)
            this_score = float(r["score"])
            if pair in seen_pair_to_score:
                kept_score = seen_pair_to_score[pair]
                if round(kept_score, 6) != round(this_score, 6):
                    cross_file_conflict_pairs += 1
                continue
            seen_pair_to_score[pair] = this_score
            deduped_train.append(r)
    after_cross_file_dedup = len(deduped_train)


    target_test_pairs: Set[Tuple[str, str]] = set()
    for r in load_jsonl(leave_out_test_file):
        target_test_pairs.add(make_pair_key(r, normalize_order=True))

    final_train: List[Dict] = []
    cross_year_contam_removed = 0
    for r in deduped_train:
        pair = make_pair_key(r, normalize_order=True)
        if pair in target_test_pairs:
            cross_year_contam_removed += 1
            continue
        final_train.append(r)


    test_records = load_jsonl(test_file)


    train_canonical_keys = {make_pair_key(r, normalize_order=True)
                            for r in final_train}
    test_canonical_keys = {make_pair_key(r, normalize_order=True)
                           for r in test_records}
    train_test_canonical_overlap = train_canonical_keys & test_canonical_keys
    if len(train_test_canonical_overlap) != 0:
        sample = list(train_test_canonical_overlap)[:5]
        raise AssertionError(
            f"LOO post-condition violated: |train ∩ test| (canonical key) "
            f"= {len(train_test_canonical_overlap)}, expected 0. "
            f"This usually means leave_out_test_file ({leave_out_test_file}) "
            f"differs from test_file ({test_file}) and the two files are "
            f"not equivalent under the canonical pair key. Sample colliding "
            f"pairs: {sample}"
        )

    stats = {
        "per_file_meta": per_file_meta,
        "raw_total": raw_total,
        "after_intra_file_dedup": after_intra_file_dedup,
        "intra_file_conflict_pairs": intra_file_conflict_pairs,
        "after_cross_file_dedup": after_cross_file_dedup,
        "cross_file_conflict_pairs": cross_file_conflict_pairs,
        "cross_year_contam_removed": cross_year_contam_removed,
        "final_train_size": len(final_train),
        "test_file": test_file,
        "test_md5": md5_file(test_file),
        "test_size": len(test_records),
        "leave_out_test_file": leave_out_test_file,
        "leave_out_test_md5": md5_file(leave_out_test_file),
        "train_test_canonical_overlap": len(train_test_canonical_overlap),
    }

    return final_train, test_records, stats


def format_loo_stats(stats: Dict, dataset_name: str) -> str:
    
    lines = []
    lines.append("=" * 75)
    lines.append(f"LOO training-set construction for {dataset_name}")
    lines.append("=" * 75)
    lines.append("")
    lines.append("Source atomic files (in order):")
    for path, n_rows, md5 in stats["per_file_meta"]:
        lines.append(f"  rows={n_rows:>6,}  md5={md5}  path={path}")
    lines.append("")
    lines.append(f"Pair key: canonical (whitespace-normalised + order-canonical), "
                 f"so (s1,s2) and (s2,s1) are treated as the same STS pair.")
    lines.append("")
    lines.append(f"Raw total rows (sum across all source files):  "
                 f"{stats['raw_total']:>6,}")


    intra_drop = stats["raw_total"] - stats["after_intra_file_dedup"]
    lines.append(f"After intra-file dedup:                        "
                 f"{stats['after_intra_file_dedup']:>6,}  "
                 f"({intra_drop:>4,} duplicate rows removed; of these, "
                 f"{stats['intra_file_conflict_pairs']:>3,} had a score "
                 f"conflicting with the kept row)")


    cross_drop = stats["after_intra_file_dedup"] - stats["after_cross_file_dedup"]
    lines.append(f"After cross-file dedup:                        "
                 f"{stats['after_cross_file_dedup']:>6,}  "
                 f"({cross_drop:>4,} cross-file duplicates removed; of "
                 f"these, {stats['cross_file_conflict_pairs']:>3,} had a "
                 f"score conflicting with the kept row)")
    lines.append(f"After LOO contamination filter (vs target test):  "
                 f"{stats['final_train_size']:>6,}  "
                 f"({stats['cross_year_contam_removed']:>4,} pairs in "
                 f"target test removed)")
    lines.append("")
    lines.append(f"Test file:                                     "
                 f"rows={stats['test_size']:>6,}  "
                 f"md5={stats['test_md5']}")
    lines.append(f"  path: {stats['test_file']}")
    lines.append(f"  NOTE: test file is loaded VERBATIM (no dedup) to "
                 f"preserve comparability")
    lines.append(f"  with prior published numbers (SimCSE/T5/E5/etc. all "
                 f"evaluate raw jsonl).")
    lines.append("")
    lines.append(f"Leave-out test file (used for contamination filter):")
    lines.append(f"  md5={stats['leave_out_test_md5']}")
    lines.append(f"  path: {stats['leave_out_test_file']}")
    if stats["leave_out_test_md5"] == stats["test_md5"]:
        lines.append(f"  (== test file, the standard LOO recipe)")
    else:
        lines.append(f"  (DIFFERENT from test file -- this is unusual; "
                     f"verify intent)")
    lines.append("")


    lines.append(f"Post-condition |train ∩ test| (canonical key):   "
                 f"{stats['train_test_canonical_overlap']:>6,}  (must be 0)")
    lines.append("")
    lines.append("=" * 75)
    return "\n".join(lines)


def build_vocab(data_sources: Iterable[Iterable[Dict]]) -> Set[str]:
    vocab: Set[str] = set()
    for records in data_sources:
        for item in records:
            vocab.update(tokenize_text(item["sentence1"]))
            vocab.update(tokenize_text(item["sentence2"]))
    return vocab


class SentencePairDataset(Dataset):
    

    def __init__(self, records: List[Dict], word_to_idx: Dict[str, int]):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.data = []

        for row in records:
            sent1 = row["sentence1"]
            sent2 = row["sentence2"]
            label = float(row["score"]) / 5.0
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


def compute_split_indices_from_records(
    records: List[Dict],
    val_ratio: float = 0.2,
    n_bins: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    
    labels = np.array([float(r["score"]) / 5.0 for r in records], dtype=np.float64)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    binned = discretizer.fit_transform(labels.reshape(-1, 1)).reshape(-1)

    rng = np.random.RandomState(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for bin_label in np.unique(binned):
        indices = np.where(binned == bin_label)[0]
        rng.shuffle(indices)
        split_point = int(len(indices) * val_ratio)
        val_indices.extend(indices[:split_point].tolist())
        train_indices.extend(indices[split_point:].tolist())
    return train_indices, val_indices


def stratified_train_val_split(
    train_dataset: SentencePairDataset,
    val_ratio: float = 0.2,
    n_bins: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    
    labels = np.array([item[2] for item in train_dataset], dtype=np.float64)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    binned = discretizer.fit_transform(labels.reshape(-1, 1)).reshape(-1)

    rng = np.random.RandomState(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for bin_label in np.unique(binned):
        indices = np.where(binned == bin_label)[0]
        rng.shuffle(indices)
        split_point = int(len(indices) * val_ratio)
        val_indices.extend(indices[:split_point].tolist())
        train_indices.extend(indices[split_point:].tolist())
    return train_indices, val_indices


def check_label_distribution(indices: List[int], dataset: Dataset) -> Dict[float, int]:
    bucket_labels = [round(dataset[idx][2], 4) for idx in indices]
    unique, counts = np.unique(bucket_labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


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
    single_u(params[12:15], wires=wires[4])
    single_u(params[15:18], wires=wires[5])


    qml.CNOT(wires=[wires[5], wires[0]])
    qml.CNOT(wires=[wires[4], wires[5]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.CNOT(wires=[wires[5], wires[0]])
    qml.CNOT(wires=[wires[4], wires[5]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.CNOT(wires=[wires[5], wires[0]])
    qml.CNOT(wires=[wires[4], wires[5]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.CNOT(wires=[wires[5], wires[0]])
    qml.CNOT(wires=[wires[4], wires[5]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[1]])

    qml.CNOT(wires=[wires[5], wires[0]])
    qml.CNOT(wires=[wires[4], wires[5]])
    qml.CNOT(wires=[wires[3], wires[4]])
    qml.CNOT(wires=[wires[2], wires[3]])
    qml.CNOT(wires=[wires[1], wires[2]])
    qml.CNOT(wires=[wires[0], wires[1]])


    single_u(params[18:21], wires=wires[0])
    single_u(params[21:24], wires=wires[1])
    single_u(params[24:27], wires=wires[2])
    single_u(params[27:30], wires=wires[3])
    single_u(params[30:33], wires=wires[4])
    single_u(params[33:36], wires=wires[5])


    single_u(params[36:39], wires=wires[6])
    single_u(params[39:42], wires=wires[7])
    single_u(params[42:45], wires=wires[8])
    single_u(params[45:48], wires=wires[9])
    single_u(params[48:51], wires=wires[10])
    single_u(params[51:54], wires=wires[11])


    qml.CNOT(wires=[wires[11], wires[6]])
    qml.CNOT(wires=[wires[10], wires[11]])
    qml.CNOT(wires=[wires[9],  wires[10]])
    qml.CNOT(wires=[wires[8],  wires[9]])
    qml.CNOT(wires=[wires[7],  wires[8]])
    qml.CNOT(wires=[wires[6],  wires[7]])

    qml.CNOT(wires=[wires[11], wires[6]])
    qml.CNOT(wires=[wires[10], wires[11]])
    qml.CNOT(wires=[wires[9],  wires[10]])
    qml.CNOT(wires=[wires[8],  wires[9]])
    qml.CNOT(wires=[wires[7],  wires[8]])
    qml.CNOT(wires=[wires[6],  wires[7]])

    qml.CNOT(wires=[wires[11], wires[6]])
    qml.CNOT(wires=[wires[10], wires[11]])
    qml.CNOT(wires=[wires[9],  wires[10]])
    qml.CNOT(wires=[wires[8],  wires[9]])
    qml.CNOT(wires=[wires[7],  wires[8]])
    qml.CNOT(wires=[wires[6],  wires[7]])

    qml.CNOT(wires=[wires[11], wires[6]])
    qml.CNOT(wires=[wires[10], wires[11]])
    qml.CNOT(wires=[wires[9],  wires[10]])
    qml.CNOT(wires=[wires[8],  wires[9]])
    qml.CNOT(wires=[wires[7],  wires[8]])
    qml.CNOT(wires=[wires[6],  wires[7]])

    qml.CNOT(wires=[wires[11], wires[6]])
    qml.CNOT(wires=[wires[10], wires[11]])
    qml.CNOT(wires=[wires[9],  wires[10]])
    qml.CNOT(wires=[wires[8],  wires[9]])
    qml.CNOT(wires=[wires[7],  wires[8]])
    qml.CNOT(wires=[wires[6],  wires[7]])


    single_u(params[54:57], wires=wires[6])
    single_u(params[57:60], wires=wires[7])
    single_u(params[60:63], wires=wires[8])
    single_u(params[63:66], wires=wires[9])
    single_u(params[66:69], wires=wires[10])
    single_u(params[69:72], wires=wires[11])


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
            torch.rand(72, generator=quantum_rng, dtype=torch.float64) * 2 * math.pi
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


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):


    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sents1, sents2, labels in data_loader:
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            raw_outputs = model(sents1, sents2)


            if torch.isnan(raw_outputs).any() or torch.isinf(raw_outputs).any():
                raw_outputs = torch.nan_to_num(
                    raw_outputs, nan=0.0, posinf=1.0, neginf=0.0
                )

            outputs = raw_outputs.detach().cpu().numpy().astype(np.float64)
            predictions.extend(outputs.tolist())
            targets.extend(labels.numpy().astype(np.float64).tolist())

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    mse = float(mean_squared_error(targets, predictions))
    pearson = safe_pearsonr(targets, predictions)
    spearman = safe_spearmanr(targets, predictions)
    return mse, pearson, spearman


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


    best_val_pearson = float("-inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None


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
            outputs = model(sents1, sents2)
            loss = criterion(outputs, labels)


            if (torch.isnan(loss) or torch.isinf(loss)
                    or torch.isnan(outputs).any() or torch.isinf(outputs).any()):
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

        val_mse, val_pearson, val_spearman = evaluate_model(model, val_loader, device)
        scheduler.step(val_pearson)
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
            f"Val MSE: {val_mse:.6f}, Val Pearson: {val_pearson:.6f}, "
            f"Val Spearman: {val_spearman:.6f}, "
            f"LRs: [{current_lrs}]{nan_tag}"
        )
        print(message)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
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
            "No valid model state was saved during training. "
            "Either the very first epoch NaN'd out entirely, or training "
            "never ran an evaluation step."
        )

    model.load_state_dict(best_model_state)


    nan_ratio = (total_nan_batches / total_batches_seen) if total_batches_seen else 0.0
    summary_lines = [
        f"Best epoch: {best_epoch}, Best Val Pearson: {best_val_pearson:.6f}",
        f"Run health: {total_nan_batches}/{total_batches_seen} batches "
        f"produced NaN/Inf ({nan_ratio:.4%}).",
    ]
    if total_nan_batches > 0:


        if nan_ratio < 0.01:
            summary_lines.append(
                "    → NaN ratio < 1%: run is clean, checkpoint trustworthy."
            )
        elif nan_ratio < 0.05:
            summary_lines.append(
                "    → NaN ratio 1-5%: mild numerical instability, "
                "checkpoint still usable but report with caution."
            )
        else:
            summary_lines.append(
                "    → NaN ratio >= 5%: significant instability, consider "
                "this seed as a stability outlier in the paper."
            )
    for line in summary_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QSTS evaluation on STS-12 through STS-16 (LOO protocol)."
    )
    parser.add_argument(
        "--glove_file",
        type=str,
        default="./glove.840B.300d.txt",
        help="Path to the GloVe 840B 300d text file.",
    )
    parser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more atomic STS jsonl files to merge as training data. "
            "For LOO protocol on STS-Y, this should be 5 atomic files: "
            "STS-12 train.jsonl + the four STS-X test.jsonl for X != Y. "
            "DO NOT pass STS-13/14/15/16 train.jsonl here -- those are "
            "already cumulative constructions of prior atomic files and "
            "would cause severe internal duplication. The loader strictly "
            "deduplicates by (sentence1, sentence2) and removes pairs "
            "appearing in --leave_out_test_file."
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help=(
            "Path to the target year's official test.jsonl. Loaded VERBATIM "
            "(no dedup) so that reported metrics are directly comparable "
            "with prior published numbers (SimCSE, Sentence-T5, E5, etc.)."
        ),
    )
    parser.add_argument(
        "--leave_out_test_file",
        type=str,
        required=True,
        help=(
            "Path to the target year's official test.jsonl, used to identify "
            "and remove any pairs that may have leaked into --train_files via "
            "cross-year SemEval test set sharing (e.g. D12_test ∩ D13_test = "
            "32 pairs). In the standard LOO recipe this equals --test_file, "
            "but it is exposed as a separate flag for full transparency."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="STS",
        help="Dataset tag for log/checkpoint filenames (e.g. STS-12, STS-14, STS-15, STS-16).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of the training set to carve out as a stratified validation set.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=5,
        help="Number of uniform bins used for the stratified train/val split.",
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max gradient L2 norm for clipping. Set to 0 or negative to disable.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)


    device = torch.device("cpu")
    topology_name = "CB"
    optimizer_name = "adam_plateau"
    dataset_name = args.dataset_name
    variant_tag = "qsts_LOO"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"QSTS_{dataset_name}_{topology_name}_{optimizer_name}_"
        f"{NUM_QUBITS}qubits_{variant_tag}_seed{args.seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"


    header = (
        f"=== Run started at {run_stamp} ===\n"
        f"dataset={dataset_name}\n"
        f"protocol=Leave-One-Year-Out (LOO) pooled STS\n"
        f"  - topology        : {topology_name}\n"
        f"  - optimizer       : {optimizer_name}\n"
        f"  - qubits          : {NUM_QUBITS}\n"
        f"  - quantum params  : 72\n"
        f"  - embedding       : pretrained GloVe (frozen)\n"
        f"  - protocol        : training set = atomic SemEval STS files\n"
        f"                      (D12_train + the four D_X_test for X != Y),\n"
        f"                      strictly deduplicated and with the target\n"
        f"                      year's test pairs removed; test set is the\n"
        f"                      target year's official test.jsonl, verbatim.\n"
        f"  - val split       : stratified {args.val_ratio:.2f} of training set, "
        f"KBins(n_bins={args.n_bins}, uniform)\n"
        f"seed={args.seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    train_records, test_records, loo_stats = load_data_loo(
        train_files=args.train_files,
        test_file=args.test_file,
        leave_out_test_file=args.leave_out_test_file,
    )


    loo_log_block = format_loo_stats(loo_stats, args.dataset_name)
    print(loo_log_block)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(loo_log_block + "\n")


    train_indices, val_indices = compute_split_indices_from_records(
        train_records,
        val_ratio=args.val_ratio,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    train_records_for_vocab = [train_records[i] for i in train_indices]
    val_records_holdout = [train_records[i] for i in val_indices]


    vocab = build_vocab([train_records_for_vocab])
    word_to_idx, pretrained_embeddings = load_glove_embeddings(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)

    val_vocab = build_vocab([val_records_holdout])
    test_vocab = build_vocab([test_records])
    val_oov = val_vocab - vocab
    test_oov = test_vocab - vocab
    union_vocab = build_vocab([train_records, test_records])

    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}\n"
        f"Total word types across training and test: {len(union_vocab)}\n"
        f"  - Validation set: {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)\n"
        f"  - Test set: {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")


    train_dataset = SentencePairDataset(train_records, word_to_idx)
    test_dataset = SentencePairDataset(test_records, word_to_idx)


    overlap = set(train_indices).intersection(val_indices)
    train_label_distribution = check_label_distribution(train_indices, train_dataset)
    val_label_distribution = check_label_distribution(val_indices, train_dataset)
    split_summary = (
        f"Total train+test size: {len(train_dataset) + len(test_dataset)}\n"
        f"  Training subset size : {len(train_indices)}\n"
        f"  Validation subset size: {len(val_indices)}\n"
        f"  Test set size        : {len(test_dataset)}\n"
        f"  Train/Val overlap    : {len(overlap)} (must be 0)\n"
        f"  Train label distribution: {train_label_distribution}\n"
        f"  Val   label distribution: {val_label_distribution}"
    )
    print(split_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(split_summary + "\n")


    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    quantum_param_seed = args.seed + 2024

    model = QuantumNeuralNetwork(
        vocab_size,
        EMBEDDING_DIM,
        REDUCED_DIM,
        pretrained_embeddings,
        quantum_param_seed=quantum_param_seed,
    ).to(device)
    criterion = nn.MSELoss()


    optimizer = optim.Adam(
        [
            {"params": model.linear.parameters(), "lr": 1e-3, "weight_decay": 1e-5},
            {"params": [model.params],            "lr": 1e-3, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
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
        num_epochs=args.num_epochs,
        patience=args.patience,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        device=device,
        grad_clip_max_norm=args.grad_clip,
    )

    test_mse, test_pearson, test_spearman = evaluate_model(model, test_loader, device)
    final_message = (
        f"Final Testing MSE: {test_mse:.6f}, "
        f"Pearson: {test_pearson:.6f}, Spearman: {test_spearman:.6f}"
    )
    print(final_message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(final_message + "\n")


if __name__ == "__main__":
    main()
