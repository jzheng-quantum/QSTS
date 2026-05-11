
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
import torch.nn.functional as F
import torch.optim as optim
from pennylane.templates import AmplitudeEmbedding
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


SEED = 42
EMBEDDING_DIM = 300
REDUCED_DIM = 64
NUM_QUBITS = 6
NUM_EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 25
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EPS = 1e-12


SUPPORTED_VARIANTS = ("qaoa", "hea")
DEFAULT_VARIANT = "qaoa"


_VARIANT_WEIGHTS_SHAPE = {
    "qaoa": (12,),
    "hea":  (12,),
}
_VARIANT_PARAM_COUNT = {
    "qaoa": 2 * NUM_QUBITS,
    "hea":  2 * NUM_QUBITS,
}


def _variant_num_params(variant: str) -> int:
    if variant not in _VARIANT_PARAM_COUNT:
        raise ValueError(
            f"Unknown variant '{variant}'. Supported: {SUPPORTED_VARIANTS}"
        )
    return _VARIANT_PARAM_COUNT[variant]


def _variant_weights_shape(variant: str):
    if variant not in _VARIANT_WEIGHTS_SHAPE:
        raise ValueError(
            f"Unknown variant '{variant}'. Supported: {SUPPORTED_VARIANTS}"
        )
    return _VARIANT_WEIGHTS_SHAPE[variant]

DEFAULT_SEEDS = [42]


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


def prepare_glove_hits_once(
    glove_file: str,
    vocab: Set[str],
    embedding_dim: int,
) -> Tuple[Dict[str, int], Dict[str, np.ndarray], int]:
    clean_vocab = {w for w in vocab if w not in (PAD_TOKEN, UNK_TOKEN)}
    word_to_idx = {word: idx + 2 for idx, word in enumerate(sorted(clean_vocab))}
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[UNK_TOKEN] = 1

    glove_path = Path(glove_file)
    if not glove_path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove_file}")

    hit_vectors: Dict[str, np.ndarray] = {}
    hits = 0
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
            hit_vectors[word] = vector
            hits += 1

    vocab_size = len(word_to_idx) - 2
    oov = vocab_size - len(hit_vectors)
    print(f"Vocabulary size: {vocab_size}")
    print(f"GloVe hits: {hits}")
    print(f"GloVe OOV: {oov}")
    print(
        f"GloVe hit rate: {hits / vocab_size:.4f}"
        if vocab_size > 0
        else "GloVe hit rate: 0.0000"
    )
    return word_to_idx, hit_vectors, hits


def build_embedding_matrix_for_seed(
    word_to_idx: Dict[str, int],
    hit_vectors: Dict[str, np.ndarray],
    embedding_dim: int,
) -> np.ndarray:
    
    embeddings = np.random.uniform(
        -0.25, 0.25, (len(word_to_idx), embedding_dim)
    ).astype(np.float64)
    embeddings[word_to_idx[PAD_TOKEN]] = 0.0
    for word, vec in hit_vectors.items():
        idx = word_to_idx.get(word)
        if idx is not None:
            embeddings[idx] = vec
    return embeddings


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
    

    def __init__(
        self,
        records: List[Dict],
        word_to_idx: Dict[str, int],
        score_scale: float = 5.0,
    ):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.score_scale = float(score_scale)
        self.data = []

        raw_scores = []
        for row in records:
            sent1 = row["sentence1"]
            sent2 = row["sentence2"]
            raw_score = float(row["score"])
            raw_scores.append(raw_score)
            label = raw_score / self.score_scale
            sent1_indices = self.sentence_to_indices(sent1)
            sent2_indices = self.sentence_to_indices(sent2)
            self.data.append((sent1_indices, sent2_indices, label))

        if raw_scores:
            self.raw_score_min = float(min(raw_scores))
            self.raw_score_max = float(max(raw_scores))
        else:
            self.raw_score_min = 0.0
            self.raw_score_max = 0.0

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
    score_scale: float = 5.0,
) -> Tuple[List[int], List[int]]:
    
    labels = np.array(
        [float(r["score"]) / float(score_scale) for r in records],
        dtype=np.float64,
    )
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


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_encoder_qaoa(amplitudes, params):
    
    AmplitudeEmbedding(
        features=amplitudes,
        wires=range(NUM_QUBITS),
        normalize=True,
        pad_with=0.0,
    )

    for q in range(NUM_QUBITS):
        qml.RY(params[q], wires=q)

    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RY(params[NUM_QUBITS + i], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])

    qml.CNOT(wires=[0, NUM_QUBITS - 1])
    qml.RY(params[2 * NUM_QUBITS - 1], wires=NUM_QUBITS - 1)
    qml.CNOT(wires=[0, NUM_QUBITS - 1])
    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_encoder_hea(amplitudes, params):
    
    AmplitudeEmbedding(
        features=amplitudes,
        wires=range(NUM_QUBITS),
        normalize=True,
        pad_with=0.0,
    )

    for q in range(NUM_QUBITS):
        qml.RZ(params[q], wires=q)

    for i in range(NUM_QUBITS - 1):
        qml.CRY(params[NUM_QUBITS + i], wires=[i, i + 1])

    qml.CRY(params[2 * NUM_QUBITS - 1], wires=[NUM_QUBITS - 1, 0])
    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]


_VARIANT_QNODE = {
    "qaoa": quantum_encoder_qaoa,
    "hea":  quantum_encoder_hea,
}


class QMLPSentenceEncoder(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
        quantum_param_seed: int,
        variant: str = DEFAULT_VARIANT,
    ):
        super().__init__()
        assert reduced_dim == 2 ** NUM_QUBITS, (
            f"reduced_dim ({reduced_dim}) must equal 2**NUM_QUBITS "
            f"({2 ** NUM_QUBITS}) for amplitude encoding."
        )
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Supported: {SUPPORTED_VARIANTS}"
            )
        self.variant = variant


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False


        self.linear = nn.Linear(embedding_dim, reduced_dim).double()


        weights_shape = _variant_weights_shape(variant)
        quantum_rng = torch.Generator()
        quantum_rng.manual_seed(int(quantum_param_seed))
        self.params = nn.Parameter(
            torch.rand(weights_shape, generator=quantum_rng, dtype=torch.float64)
            * 2.0 * math.pi
        )


        self.quantum = _VARIANT_QNODE[variant]

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

    def forward(self, x_text: torch.Tensor) -> torch.Tensor:
        
        embedded = self.embedding(x_text)
        reduced = self.linear(embedded)
        mask = x_text != 0
        sentence_vec = self.masked_mean(reduced, mask)

        outputs = []
        batch_size = x_text.size(0)
        for i in range(batch_size):
            amp = self.prepare_quantum_input(sentence_vec[i])

            z_exps = self.quantum(amp, self.params)
            if isinstance(z_exps, (list, tuple)):
                z_vec = torch.stack([z.reshape(()) for z in z_exps], dim=0)
            else:

                z_vec = z_exps.reshape(NUM_QUBITS)
            outputs.append(z_vec.reshape(1, NUM_QUBITS))
        return torch.cat(outputs, dim=0)


class QMLPSentenceSimilarity(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
        quantum_param_seed: int,
        variant: str = DEFAULT_VARIANT,
    ):
        super().__init__()
        self.encoder = QMLPSentenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            reduced_dim=reduced_dim,
            pretrained_embeddings=pretrained_embeddings,
            quantum_param_seed=quantum_param_seed,
            variant=variant,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        v1 = self.encoder(x1)
        v2 = self.encoder(x2)
        cos_sim = F.cosine_similarity(v1, v2, dim=1, eps=EPS)
        return (cos_sim + 1.0) / 2.0


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
            "Either the first epoch NaN'd out entirely, or no evaluation step ran."
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
                "this seed as a stability outlier."
            )
    for line in summary_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QMLP Hybrid (QAOA/HEA) baseline on STS-12..STS-16 under the LOO pooled STS protocol."
    )
    parser.add_argument("--glove_file", type=str,
                        default="./glove.840B.300d.txt")
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
            "cross-year SemEval test set sharing (e.g. D12_test n D13_test = "
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
        help="Fraction of TRAIN to carve out as a stratified validation set.",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=5,
        help="Number of uniform bins used for the stratified train/val split.",
    )
    parser.add_argument(
        "--score_scale",
        type=float,
        default=5.0,
        help=(
            "Divisor used to rescale the raw similarity score into [0, 1]. "
            "STS-12..STS-16 use a 0..5 scale (default 5.0)."
        ),
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"List of random seeds to run. Defaults to {DEFAULT_SEEDS}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shortcut for a single-seed run. Ignored if --seeds is given.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient L2 norm for clipping. "
                             "Set to 0 or negative to disable.")
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT,
                        choices=list(SUPPORTED_VARIANTS),
                        help=(
                            "Quantum ansatz variant: "
                            "'qaoa' (QAOA-style, 12 params), "
                            "'hea' (Hardware-Efficient Ansatz, 12 params). "
                            f"Default: {DEFAULT_VARIANT}."
                        ))
    return parser


def resolve_seeds(args) -> List[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return list(args.seeds)
    if args.seed is not None:
        return [int(args.seed)]
    return list(DEFAULT_SEEDS)


def run_single_seed(
    seed: int,
    *,
    train_records: List[Dict],
    test_records: List[Dict],
    hit_vectors: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    log_dir: Path,
    run_stamp: str,
    variant_tag: str,
    model_name: str,
    dataset_name: str,
    optimizer_name: str,
    loo_log_block: str = "",
    variant: str,
) -> Dict[str, float]:
    set_seed(seed)


    train_indices, val_indices = compute_split_indices_from_records(
        train_records,
        val_ratio=args.val_ratio,
        n_bins=args.n_bins,
        seed=seed,
        score_scale=args.score_scale,
    )
    train_records_for_vocab = [train_records[i] for i in train_indices]
    val_records_holdout = [train_records[i] for i in val_indices]

    seed_vocab = build_vocab([train_records_for_vocab])
    clean_vocab = {w for w in seed_vocab if w not in (PAD_TOKEN, UNK_TOKEN)}
    word_to_idx: Dict[str, int] = {
        word: idx + 2 for idx, word in enumerate(sorted(clean_vocab))
    }
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[UNK_TOKEN] = 1
    vocab_size = len(word_to_idx)

    pretrained_embeddings = build_embedding_matrix_for_seed(
        word_to_idx=word_to_idx,
        hit_vectors=hit_vectors,
        embedding_dim=EMBEDDING_DIM,
    )

    base_name = (
        f"{model_name}_{dataset_name}_{optimizer_name}_"
        f"{variant_tag}_seed{seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"


    if variant == "qaoa":
        ansatz_desc = (
            "QAOA-style (L=1): RY per qubit + "
            "(CNOT-RY-CNOT) chain on (i,i+1) for i=0..4 + "
            "(CNOT-RY-CNOT) long-range closure on (0,5). "
            "Structural deviation: inner RZ -> RY (see docstring)"
        )
    elif variant == "hea":
        ansatz_desc = (
            "Hardware-Efficient Ansatz (L=1): RZ per qubit + "
            "CRY(ctrl=i, tgt=i+1) chain for i=0..4 + "
            "CRY(ctrl=5, tgt=0) closure"
        )
    else:
        ansatz_desc = f"UNKNOWN VARIANT: {variant}"

    n_quantum_expected = _variant_num_params(variant)

    header = (
        f"=== Run started at {run_stamp} (seed={seed}) ===\n"
        f"model={model_name}\n"
        f"dataset={dataset_name}\n"
        f"protocol=Leave-One-Year-Out (LOO) pooled STS\n"
        f"variant={variant_tag}\n"
        f"  - embedding       : GloVe 840B 300d, frozen, padding_idx=0\n"
        f"  - vocabulary      : (inductive; OOV in val/test → <UNK>)\n"
        f"  - linear          : Linear({EMBEDDING_DIM}->{REDUCED_DIM}), trainable\n"
        f"  - quantum encoder : QMLP on {NUM_QUBITS} qubits (shared across sent_A/sent_B)\n"
        f"                      AmplitudeEmbedding(64->{NUM_QUBITS}q) +\n"
        f"                      {ansatz_desc}\n"
        f"  - quantum params  : {n_quantum_expected}  "
        f"(ansatz variant = '{variant}')\n"
        f"  - readout         : [<Z_0>, ..., <Z_{NUM_QUBITS - 1}>] -> "
        f"{NUM_QUBITS}-dim sentence embedding\n"
        f"  - similarity head : cosine_similarity(v1, v2)/2 + 0.5  (same as CNN baseline)\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


        if loo_log_block:
            f.write(loo_log_block + "\n")


    val_vocab = build_vocab([val_records_holdout])
    test_vocab = build_vocab([test_records])
    val_oov = val_vocab - seed_vocab
    test_oov = test_vocab - seed_vocab
    union_vocab = build_vocab([train_records, test_records])

    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}\n"
        f"Total word types across train, validation, and test: {len(union_vocab)}\n"
        f"  · Val  set has {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>\n"
        f"  · Test set has {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>"
    )
    print(glove_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")


    train_dataset = SentencePairDataset(
        train_records, word_to_idx, score_scale=args.score_scale
    )
    test_dataset = SentencePairDataset(
        test_records, word_to_idx, score_scale=args.score_scale
    )


    score_summary_lines = [
        f"Raw score statistics (before rescaling by /{args.score_scale}):",
        f"  TRAIN: min={train_dataset.raw_score_min:.4f}, max={train_dataset.raw_score_max:.4f}",
        f"  TEST : min={test_dataset.raw_score_min:.4f}, max={test_dataset.raw_score_max:.4f}",
    ]
    overall_max = max(train_dataset.raw_score_max, test_dataset.raw_score_max)
    overall_min = min(train_dataset.raw_score_min, test_dataset.raw_score_min)
    if overall_max > args.score_scale + 1e-6 or overall_min < -1e-6:
        score_summary_lines.append(
            f"  WARNING: raw scores fall outside [0, {args.score_scale}]. "
            f"Either --score_scale is misconfigured or the data file is unexpected."
        )
    for line in score_summary_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in score_summary_lines:
            f.write(line + "\n")


    overlap = set(train_indices).intersection(val_indices)
    train_label_distribution = check_label_distribution(train_indices, train_dataset)
    val_label_distribution = check_label_distribution(val_indices, train_dataset)
    split_summary = (
        f"Total train+test size : {len(train_dataset) + len(test_dataset)}\n"
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
    loader_generator.manual_seed(seed)

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


    quantum_param_seed = seed + 2024

    model = QMLPSentenceSimilarity(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=pretrained_embeddings,
        quantum_param_seed=quantum_param_seed,
        variant=variant,
    ).to(device)
    criterion = nn.MSELoss()


    n_linear = sum(p.numel() for p in model.encoder.linear.parameters()
                   if p.requires_grad)
    n_quantum = model.encoder.params.numel()
    pc_lines = [
        "Parameter breakdown:",
        f"  - Linear({EMBEDDING_DIM}->{REDUCED_DIM}) : {n_linear:,}",
        f"  - Quantum QMLP params        : {n_quantum:,}",
        f"  - Total trainable            : {n_linear + n_quantum:,}",
    ]
    for line in pc_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in pc_lines:
            f.write(line + "\n")


    optimizer = optim.Adam(
        [
            {"params": model.encoder.linear.parameters(),
             "lr": 1e-3, "weight_decay": 1e-5},
            {"params": [model.encoder.params],
             "lr": 1e-3, "weight_decay": 0.0},
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

    return {"test_mse": test_mse,
            "test_pearson": test_pearson,
            "test_spearman": test_spearman}


def main() -> None:
    args = build_argparser().parse_args()
    seeds = resolve_seeds(args)
    variant = args.variant

    device = torch.device("cpu")
    model_name = "QMLP_Hybrid"
    dataset_name = args.dataset_name
    optimizer_name = "adam_plateau"


    n_params_for_tag = _variant_num_params(variant)
    variant_tag = "qmlp"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


    train_records, test_records, loo_stats = load_data_loo(
        train_files=args.train_files,
        test_file=args.test_file,
        leave_out_test_file=args.leave_out_test_file,
    )


    loo_log_block = format_loo_stats(loo_stats, args.dataset_name)
    print(loo_log_block)


    super_vocab = build_vocab([train_records])
    _super_word_to_idx, hit_vectors, _hits = prepare_glove_hits_once(
        args.glove_file, super_vocab, EMBEDDING_DIM
    )
    print(
        f"  (above stats are for the LOO TRAIN pool only; per-seed "
        f"vocabularies are strict subsets of this pool — each seed's "
        f"80% TRAIN partition — and are reported per-seed below.)"
    )
    print(f"Selected ansatz variant: '{variant}' "
          f"({n_params_for_tag} quantum params)")

    all_results: List[Tuple[int, Dict[str, float]]] = []
    for idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"  Starting seed {seed}  ({idx + 1}/{len(seeds)})")
        print("=" * 72)
        result = run_single_seed(
            seed=seed,
            train_records=train_records,
            test_records=test_records,
            hit_vectors=hit_vectors,
            args=args,
            device=device,
            log_dir=log_dir,
            run_stamp=run_stamp,
            variant_tag=variant_tag,
            model_name=model_name,
            dataset_name=dataset_name,
            optimizer_name=optimizer_name,
            loo_log_block=loo_log_block,
            variant=variant,
        )
        all_results.append((seed, result))


    mses = np.array([r["test_mse"] for _, r in all_results], dtype=np.float64)
    pearsons = np.array([r["test_pearson"] for _, r in all_results], dtype=np.float64)
    spearmans = np.array([r["test_spearman"] for _, r in all_results], dtype=np.float64)

    def _mean_std(x: np.ndarray) -> Tuple[float, float]:
        if x.size <= 1:
            return float(x.mean()), 0.0
        return float(x.mean()), float(x.std(ddof=1))

    mse_mean, mse_std = _mean_std(mses)
    pr_mean, pr_std = _mean_std(pearsons)
    sp_mean, sp_std = _mean_std(spearmans)

    summary_lines: List[str] = []
    summary_lines.append("=" * 72)
    summary_lines.append(
        f"Multi-seed summary  |  model={model_name}  variant={variant_tag}  "
        f"run_stamp={run_stamp}"
    )
    summary_lines.append(f"Seeds ({len(seeds)}): {seeds}")
    summary_lines.append("-" * 72)
    summary_lines.append(
        f"{'seed':>10} | {'test MSE':>12} | {'test Pearson':>14} | {'test Spearman':>14}"
    )
    for seed, r in all_results:
        summary_lines.append(
            f"{seed:>10} | {r['test_mse']:>12.6f} | "
            f"{r['test_pearson']:>14.6f} | {r['test_spearman']:>14.6f}"
        )
    summary_lines.append("-" * 72)
    summary_lines.append(
        f"{'mean':>10} | {mse_mean:>12.6f} | {pr_mean:>14.6f} | {sp_mean:>14.6f}"
    )
    summary_lines.append(
        f"{'std':>10} | {mse_std:>12.6f} | {pr_std:>14.6f} | {sp_std:>14.6f}   (ddof=1)"
    )
    summary_lines.append("-" * 72)
    summary_lines.append(
        f"Summary: "
        f"MSE = {mse_mean:.4f} ± {mse_std:.4f},  "
        f"Pearson = {pr_mean:.4f} ± {pr_std:.4f},  "
        f"Spearman = {sp_mean:.4f} ± {sp_std:.4f}"
    )
    summary_lines.append("=" * 72)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)


    for seed, _ in all_results:
        per_seed_log = log_dir / (
            f"{model_name}_{dataset_name}_{optimizer_name}_"
            f"{variant_tag}_seed{seed}_{run_stamp}.log"
        )
        with per_seed_log.open("a", encoding="utf-8") as f:
            f.write("\n" + summary_text + "\n")

    summary_path = log_dir / (
        f"{model_name}_{dataset_name}_{optimizer_name}_"
        f"{variant_tag}_SUMMARY_{run_stamp}.log"
    )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
