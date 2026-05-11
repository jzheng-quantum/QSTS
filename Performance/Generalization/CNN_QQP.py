

from typing import Dict, Iterable, List, Optional, Set, Tuple
import argparse
import copy
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
EMBEDDING_DIM = 300
REDUCED_DIM = 16

CNN_KERNEL_SIZES = (3, 4, 5)
CNN_N_FILTERS = 48
NUM_EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 15
INTERNAL_VAL_RATIO = 0.1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EPS = 1e-12
PROB_CLAMP_EPS = 1e-7


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
        embeddings[word_to_idx[word]] = vec
    return embeddings


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


class CNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_filters: int,
        kernel_sizes: Tuple[int, ...],
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False
        self.linear = nn.Linear(embedding_dim, reduced_dim).double()
        self.n_filters = n_filters
        self.kernel_sizes = tuple(kernel_sizes)
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            built_in_pad = k // 2 if (k % 2 == 1) else 0
            self.convs.append(
                nn.Conv1d(
                    in_channels=reduced_dim,
                    out_channels=n_filters,
                    kernel_size=k,
                    padding=built_in_pad,
                ).double()
            )

        self.fc = nn.Linear(len(self.kernel_sizes) * n_filters, reduced_dim).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0)
        emb = self.embedding(x)
        emb = self.linear(emb)
        emb = emb * mask.unsqueeze(-1).to(dtype=emb.dtype)
        x_conv = emb.transpose(1, 2)

        neg_inf = torch.finfo(x_conv.dtype).min
        mask_t = mask.unsqueeze(1)

        pooled_branches: List[torch.Tensor] = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            if k % 2 == 0:
                x_in = F.pad(x_conv, (k // 2 - 1, k // 2))
            else:
                x_in = x_conv
            conv_out = F.relu(conv(x_in))

            conv_out = conv_out.masked_fill(~mask_t, neg_inf)
            pooled = conv_out.max(dim=2).values

            pooled = torch.where(
                torch.isfinite(pooled), pooled, torch.zeros_like(pooled)
            )
            pooled_branches.append(pooled)

        concat = torch.cat(pooled_branches, dim=1)
        return self.fc(concat)

    def linear_parameters(self):
        return self.linear.parameters()

    def cnn_parameters(self):
        params: List[torch.nn.Parameter] = []
        for conv in self.convs:
            params.extend(conv.parameters())
        params.extend(self.fc.parameters())
        return params

    @staticmethod
    def _numel(params) -> int:
        return sum(p.numel() for p in params)

    def count_parameters(self) -> Dict[str, int]:
        lin = self._numel(p for p in self.linear_parameters() if p.requires_grad)
        cnn = self._numel(p for p in self.cnn_parameters() if p.requires_grad)
        return {
            "embedding": 0,
            "linear": lin,
            "cnn": cnn,
            "total_trainable": lin + cnn,
        }


class SentenceSimilarityCNN(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_filters: int,
        kernel_sizes: Tuple[int, ...],
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.encoder = CNNEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            reduced_dim=reduced_dim,
            pretrained_embeddings=pretrained_embeddings,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        v1 = self.encoder(x1)
        v2 = self.encoder(x2)
        cos_sim = F.cosine_similarity(v1, v2, dim=1, eps=EPS)
        return (cos_sim + 1.0) / 2.0


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

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sents1, sents2, labels in train_loader:
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            raw_outputs = model(sents1, sents2)


            outputs = torch.clamp(
                raw_outputs, min=PROB_CLAMP_EPS, max=1.0 - PROB_CLAMP_EPS
            )
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Training loss became NaN or Inf.")
            loss.backward()
            if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=grad_clip_max_norm,
                )
            optimizer.step()
            total_loss += float(loss.item())

        val_bce, val_acc, val_f1, val_auc = evaluate_model(model, val_loader, device)
        scheduler.step(val_f1)
        train_loss = total_loss / len(train_loader)

        current_lrs = ", ".join(f"{group['lr']:.6g}" for group in optimizer.param_groups)
        message = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {train_loss:.6f}, "
            f"Val BCE: {val_bce:.6f}, Val Acc: {val_acc:.6f}, "
            f"Val F1: {val_f1:.6f}, Val AUC: {val_auc:.6f}, "
            f"LRs: [{current_lrs}]"
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
        raise RuntimeError("No valid model state was saved during training.")

    model.load_state_dict(best_model_state)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.6f}\n")


DEFAULT_SEEDS = [42]


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
        description="CNN baseline on QQP (16-dim, kernels 3/4/5, n_filters=48).",
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
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"List of random seeds. Defaults to {DEFAULT_SEEDS}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shortcut for a single-seed run. Equivalent to --seeds <seed>. "
                             "If --seeds is given, this is ignored.")
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


def resolve_seeds(args) -> List[int]:
    if args.seeds is not None and len(args.seeds) > 0:
        return list(args.seeds)
    if args.seed is not None:
        return [int(args.seed)]
    return list(DEFAULT_SEEDS)


def resolve_size_label(target_size: int, n_train_full: int) -> str:
    if target_size >= n_train_full:
        return "all"
    return str(target_size)


def run_single_seed(
    seed: int,
    *,
    train_inner: pd.DataFrame,
    val_inner: pd.DataFrame,
    test_data: pd.DataFrame,
    word_to_idx: Dict[str, int],
    hit_vectors: Dict[str, np.ndarray],
    vocab_size: int,
    glove_summary: str,
    args: argparse.Namespace,
    device: torch.device,
    log_dir: Path,
    run_stamp: str,
    variant_tag: str,
    model_name: str,
    dataset_name: str,
    optimizer_name: str,
    size_label: str,
    num_epochs: int,
    patience: int,
    is_all_mode: bool,
    n_dev_full: int,
    train_pos_frac: float,
    dev_pos_frac: float,
    n_train_full: int,
) -> Dict[str, float]:
    set_seed(seed)
    pretrained_embeddings = build_embedding_matrix_for_seed(
        word_to_idx=word_to_idx,
        hit_vectors=hit_vectors,
        embedding_dim=EMBEDDING_DIM,
    )

    base_name = (
        f"{model_name}_{dataset_name}_{optimizer_name}_"
        f"{variant_tag}_size{size_label}_seed{seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"

    header = (
        f"=== Run started at {run_stamp} (seed={seed}) ===\n"
        f"model={model_name}\n"
        f"variant={variant_tag}\n"
        f"  - dataset         : QQP (GLUE)\n"
        f"  - QQP train       : {n_train_full} pairs (pos frac {train_pos_frac:.4f})\n"
        f"  - QQP dev (test)  : {n_dev_full} pairs  (pos frac {dev_pos_frac:.4f})\n"
        f"  - train subset    : size={size_label}  (stratified, seed={seed})\n"
        f"  - inner val ratio : {args.internal_val_ratio} of the train subset (stratified)\n"
        f"  - embedding       : GloVe-840B-300d, frozen (requires_grad=False), padding_idx=0\n"
        f"  - vocabulary      : (strictly inductive, label-free,\n"
        f"                       fixed across all training-size sweeps)\n"
        f"  - linear          : Linear({EMBEDDING_DIM}->{REDUCED_DIM})\n"
        f"  - CNN             : parallel Conv1d branches "
        f"(kernels={list(CNN_KERNEL_SIZES)}, n_filters={CNN_N_FILTERS}) "
        f"+ ReLU + masked-max-pool + concat + Linear({len(CNN_KERNEL_SIZES)*CNN_N_FILTERS}->{REDUCED_DIM})\n"
        f"  - similarity head : cosine remapped to [0,1]\n"
        f"  - task            : binary paraphrase identification\n"
        f"  - loss            : BCELoss on clamped score (eps={PROB_CLAMP_EPS})\n"
        f"  - early stop      : val F1 (binary, pos_label=1) at threshold 0.5\n"
        f"  - eval metrics    : Accuracy, F1, AUC-ROC; reported at BOTH threshold 0.5\n"
        f"                       and at the F1-optimal threshold tuned on the internal val set\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={num_epochs}, "
        f"patience={patience}, grad_clip={args.grad_clip}, "
        f"internal_val_ratio={args.internal_val_ratio}, all_mode={is_all_mode}, "
        f"device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")

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

    train_dataset = QQPPairDataset(train_inner, word_to_idx)
    val_dataset = QQPPairDataset(val_inner, word_to_idx)
    test_dataset = QQPPairDataset(test_data, word_to_idx)

    print(f"Number of training samples:   {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of testing samples:    {len(test_dataset)}")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    model = SentenceSimilarityCNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        n_filters=CNN_N_FILTERS,
        kernel_sizes=CNN_KERNEL_SIZES,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)


    criterion = nn.BCELoss()

    pc = model.encoder.count_parameters()
    pc_lines = [
        "Parameter breakdown:",
        f"  - Linear({EMBEDDING_DIM}->{REDUCED_DIM}) : {pc['linear']:,}",
        f"  - CNN ({len(CNN_KERNEL_SIZES)} parallel Conv1d + Linear head) : {pc['cnn']:,}",
        f"  - Total trainable : {pc['total_trainable']:,}",
    ]
    for line in pc_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in pc_lines:
            f.write(line + "\n")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=1e-3,
        weight_decay=1e-5,
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
        min_lr=1e-5,
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
        f"Final Testing (QQP dev, n={len(test_targets)}):",
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

    return {
        "test_bce":           test_bce,
        "test_auc":           test_auc,
        "test_acc_05":        test_acc_05,
        "test_f1_05":         test_f1_05,
        "test_acc_tuned":     test_acc_tuned,
        "test_f1_tuned":      test_f1_tuned,
        "best_thr_val":       best_thr_val,
        "best_f1_val":        best_f1_val,
    }


def main() -> None:
    args = build_argparser().parse_args()
    seeds = resolve_seeds(args)

    device = torch.device("cpu")
    model_name = "CNN"
    dataset_name = "QQP"
    optimizer_name = "adam_plateau"
    variant_tag = "cnn"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


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


    vocab = build_vocab([train_full_df])
    word_to_idx, hit_vectors, _hits = prepare_glove_hits_once(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)
    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}"
    )
    print(glove_summary)


    expected_size = n_train_full if is_all_mode else args.train_size

    def _get_or_make_subsample(seed: int) -> pd.DataFrame:
        cache_path = (
            cache_dir
            / f"qqp_train_subsample_size{size_label}_seed{seed}_nfull{n_train_full}.tsv"
        )
        if cache_path.is_file():
            df = pd.read_csv(
                cache_path, sep="\t", keep_default_na=False, dtype=str, quoting=3,
            )
            df["is_duplicate"] = df["is_duplicate"].astype(int)
            if len(df) != expected_size:
                raise RuntimeError(
                    f"Cached training subsample at {cache_path} has "
                    f"{len(df)} rows but the current configuration "
                    f"requires {expected_size} rows (train_size={size_label}, "
                    f"seed={seed}, n_train_full={n_train_full}).  Refusing "
                    f"to use a stale cache.  Delete the file and re-run."
                )
            print(f"Loaded cached training subsample from {cache_path} ({len(df)} rows).")
        else:
            df = stratified_subsample(
                train_full_df,
                target_size=args.train_size,
                label_col="is_duplicate",
                seed=seed,
            )
            df.to_csv(cache_path, sep="\t", index=False, quoting=3)
            print(f"Wrote new training subsample to {cache_path} ({len(df)} rows).")
        return df


    all_results: List[Tuple[int, Dict[str, float]]] = []
    for idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"  Starting seed {seed}  ({idx + 1}/{len(seeds)})")
        print("=" * 72)

        train_subset = _get_or_make_subsample(seed)
        train_inner, val_inner = stratified_split(
            train_subset,
            val_ratio=args.internal_val_ratio,
            label_col="is_duplicate",
            seed=seed,
        )

        result = run_single_seed(
            seed=seed,
            train_inner=train_inner,
            val_inner=val_inner,
            test_data=dev_full_df,
            word_to_idx=word_to_idx,
            hit_vectors=hit_vectors,
            vocab_size=vocab_size,
            glove_summary=glove_summary,
            args=args,
            device=device,
            log_dir=log_dir,
            run_stamp=run_stamp,
            variant_tag=variant_tag,
            model_name=model_name,
            dataset_name=dataset_name,
            optimizer_name=optimizer_name,
            size_label=size_label,
            num_epochs=num_epochs,
            patience=patience,
            is_all_mode=is_all_mode,
            n_dev_full=n_dev_full,
            train_pos_frac=train_pos_frac,
            dev_pos_frac=dev_pos_frac,
            n_train_full=n_train_full,
        )
        all_results.append((seed, result))


    bces       = np.array([r["test_bce"]       for _, r in all_results], dtype=np.float64)
    aucs       = np.array([r["test_auc"]       for _, r in all_results], dtype=np.float64)
    accs_05    = np.array([r["test_acc_05"]    for _, r in all_results], dtype=np.float64)
    f1s_05     = np.array([r["test_f1_05"]     for _, r in all_results], dtype=np.float64)
    accs_tuned = np.array([r["test_acc_tuned"] for _, r in all_results], dtype=np.float64)
    f1s_tuned  = np.array([r["test_f1_tuned"]  for _, r in all_results], dtype=np.float64)

    def _mean_std(x: np.ndarray) -> Tuple[float, float]:
        if x.size <= 1:
            return float(x.mean()), 0.0
        return float(x.mean()), float(x.std(ddof=1))

    bce_mean,    bce_std    = _mean_std(bces)
    auc_mean,    auc_std    = _mean_std(aucs)
    acc05_mean,  acc05_std  = _mean_std(accs_05)
    f1_05_mean,  f1_05_std  = _mean_std(f1s_05)
    acct_mean,   acct_std   = _mean_std(accs_tuned)
    f1t_mean,    f1t_std    = _mean_std(f1s_tuned)

    summary_lines: List[str] = []
    summary_lines.append("=" * 88)
    summary_lines.append(
        f"Multi-seed summary  |  model={model_name}  variant={variant_tag}  "
        f"size={size_label}  run_stamp={run_stamp}"
    )
    summary_lines.append(f"Seeds ({len(seeds)}): {seeds}")
    summary_lines.append("-" * 88)
    summary_lines.append(
        f"{'seed':>6} | {'BCE':>10} | {'AUC':>10} | "
        f"{'Acc@0.5':>10} | {'F1@0.5':>10} | "
        f"{'Acc@best':>10} | {'F1@best':>10}"
    )
    for seed, r in all_results:
        summary_lines.append(
            f"{seed:>6} | {r['test_bce']:>10.6f} | {r['test_auc']:>10.6f} | "
            f"{r['test_acc_05']:>10.6f} | {r['test_f1_05']:>10.6f} | "
            f"{r['test_acc_tuned']:>10.6f} | {r['test_f1_tuned']:>10.6f}"
        )
    summary_lines.append("-" * 88)
    summary_lines.append(
        f"{'mean':>6} | {bce_mean:>10.6f} | {auc_mean:>10.6f} | "
        f"{acc05_mean:>10.6f} | {f1_05_mean:>10.6f} | "
        f"{acct_mean:>10.6f} | {f1t_mean:>10.6f}"
    )
    summary_lines.append(
        f"{'std':>6} | {bce_std:>10.6f} | {auc_std:>10.6f} | "
        f"{acc05_std:>10.6f} | {f1_05_std:>10.6f} | "
        f"{acct_std:>10.6f} | {f1t_std:>10.6f}   (ddof=1)"
    )
    summary_lines.append("-" * 88)
    summary_lines.append(
        f"Summary @ thr=0.5 :  Acc = {acc05_mean:.4f} ± {acc05_std:.4f},  "
        f"F1 = {f1_05_mean:.4f} ± {f1_05_std:.4f},  "
        f"AUC = {auc_mean:.4f} ± {auc_std:.4f}"
    )
    summary_lines.append(
        f"Summary @ tuned-thr: Acc = {acct_mean:.4f} ± {acct_std:.4f},  "
        f"F1 = {f1t_mean:.4f} ± {f1t_std:.4f}"
    )
    summary_lines.append("=" * 88)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    for seed, _ in all_results:
        per_seed_log = log_dir / (
            f"{model_name}_{dataset_name}_{optimizer_name}_"
            f"{variant_tag}_size{size_label}_seed{seed}_{run_stamp}.log"
        )
        with per_seed_log.open("a", encoding="utf-8") as f:
            f.write("\n" + summary_text + "\n")

    summary_path = log_dir / (
        f"{model_name}_{dataset_name}_{optimizer_name}_"
        f"{variant_tag}_size{size_label}_SUMMARY_{run_stamp}.log"
    )
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
