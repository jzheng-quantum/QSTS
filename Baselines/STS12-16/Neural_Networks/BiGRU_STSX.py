from typing import Dict, Iterable, List, Set, Tuple
import argparse
import copy
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


SEED = 42
EMBEDDING_DIM = 300
REDUCED_DIM = 64
GRU_HIDDEN_DIM = 64
GRU_NUM_LAYERS = 1
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
    print(
        f"GloVe hit rate: {hits / vocab_size:.4f}"
        if vocab_size > 0
        else "GloVe hit rate: 0.0000"
    )

    return word_to_idx, embeddings


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


class BiGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False

        self.linear = nn.Linear(embedding_dim, reduced_dim).double()

        self.gru = nn.GRU(
            input_size=reduced_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        ).double()

        self.num_directions = 2
        self.hidden_dim = gru_hidden_dim
        self.fc = nn.Linear(self.num_directions * gru_hidden_dim,
                            reduced_dim).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mask = (x != 0)


        emb = self.embedding(x)
        emb = self.linear(emb)
        emb = emb * mask.unsqueeze(-1).to(dtype=emb.dtype)


        lengths = mask.sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )


        _, h_n = self.gru(packed)


        h_n = h_n.view(
            self.gru.num_layers,
            self.num_directions,
            x.size(0),
            self.hidden_dim,
        )


        last_layer_h = h_n[-1]
        forward_last   = last_layer_h[0]
        backward_first = last_layer_h[1]

        combined = torch.cat([forward_last, backward_first], dim=-1)
        return self.fc(combined)

    def linear_parameters(self):
        return self.linear.parameters()

    def bigru_parameters(self):
        return list(self.gru.parameters()) + list(self.fc.parameters())

    @staticmethod
    def _numel(params) -> int:
        return sum(p.numel() for p in params)

    def count_parameters(self) -> Dict[str, int]:
        lin = self._numel(p for p in self.linear_parameters() if p.requires_grad)
        bigru = self._numel(p for p in self.bigru_parameters() if p.requires_grad)
        return {
            "embedding": 0,
            "linear": lin,
            "bigru": bigru,
            "total_trainable": lin + bigru,
        }


class SentenceSimilarityBiGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.encoder = BiGRUEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            reduced_dim=reduced_dim,
            pretrained_embeddings=pretrained_embeddings,
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
            outputs = model(sents1, sents2).detach().cpu().numpy().astype(np.float64)
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
    device: torch.device,
    checkpoint_path: Path,
    log_path: Path,
    grad_clip_max_norm: float = 1.0,
) -> None:
    best_val_pearson = -float("inf")
    best_model_state = None
    patience_counter = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for sents1, sents2, labels in train_loader:
            sents1 = sents1.to(device)
            sents2 = sents2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sents1, sents2)
            loss = criterion(outputs, labels)
            loss.backward()

            if grad_clip_max_norm is not None and grad_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=grad_clip_max_norm,
                )

            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        val_mse, val_pearson, val_spearman = evaluate_model(model, val_loader, device)

        current_lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
        scheduler.step(val_pearson)

        epoch_line = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {avg_loss:.6f}, "
            f"Val MSE: {val_mse:.6f}, "
            f"Val Pearson: {val_pearson:.6f}, "
            f"Val Spearman: {val_spearman:.6f}, "
            f"LRs: {current_lrs}"
        )
        print(epoch_line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(epoch_line + "\n")

        if (not np.isnan(val_pearson)) and val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stop_line = "Early stopping triggered."
            print(stop_line)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(stop_line + "\n")
            break

    if best_model_state is None:
        raise RuntimeError("No valid model state was saved during training.")

    model.load_state_dict(best_model_state)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}, Best Val Pearson: {best_val_pearson:.6f}\n")


DEFAULT_SEEDS = [42]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classical neural baseline on STS-12..STS-16 "
                    "under the Leave-One-Year-Out (LOO) pooled STS protocol."
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
                        help="List of random seeds to run. "
                             f"Defaults to {DEFAULT_SEEDS}.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shortcut for a single-seed run. Equivalent to --seeds <seed>. "
                             "If --seeds is given, this is ignored.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient L2 norm for clipping. Set to 0 or negative to disable.")
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

    header = (
        f"=== Run started at {run_stamp} (seed={seed}) ===\n"
        f"model={model_name}\n"
        f"dataset={dataset_name}\n"
        f"protocol=Leave-One-Year-Out (LOO) pooled STS\n"
        f"variant={variant_tag}\n"
        f"  - embedding       : GloVe, padding_idx=0\n"
        f"  - linear          : Linear({EMBEDDING_DIM}->{REDUCED_DIM})\n"
        f"  - BiGRU           : GRU(input={REDUCED_DIM}, hidden={GRU_HIDDEN_DIM}, "
        f"layers={GRU_NUM_LAYERS}, bidirectional=True, batch_first) + "
        f"pack_padded_sequence + concat(h_n forward, h_n backward) + Linear({2*GRU_HIDDEN_DIM}->{REDUCED_DIM})\n"
        f"  - similarity      : cosine then (cos+1)/2  -> [0, 1]\n"
        f"  - protocol        : training set = atomic SemEval STS files\n"
        f"                      (D12_train + the four D_X_test for X != Y),\n"
        f"                      strictly deduplicated and with the target\n"
        f"                      year's test pairs removed; test set is the\n"
        f"                      target year's official test.jsonl, verbatim.\n"
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
        f"Total word types across training and test: {len(union_vocab)}\n"
        f"  - Validation set: {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)\n"
        f"  - Test set: {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)"
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

    model = SentenceSimilarityBiGRU(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        gru_num_layers=GRU_NUM_LAYERS,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)
    criterion = nn.MSELoss()

    pc = model.encoder.count_parameters()
    pc_lines = [
        "Parameter breakdown:",
        f"  - Linear({EMBEDDING_DIM}->{REDUCED_DIM}) : {pc['linear']:,}",
        f"  - BiGRU (bidirectional GRU + Linear head) : {pc['bigru']:,}",
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
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
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
        device=device,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        grad_clip_max_norm=args.grad_clip,
    )

    test_mse, test_pearson, test_spearman = evaluate_model(model, test_loader, device)
    test_line = (
        f"Final Testing MSE: {test_mse:.6f}, "
        f"Pearson: {test_pearson:.6f}, "
        f"Spearman: {test_spearman:.6f}"
    )
    print(test_line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(test_line + "\n")

    return {"test_mse": test_mse,
            "test_pearson": test_pearson,
            "test_spearman": test_spearman}


def main() -> None:
    args = build_argparser().parse_args()
    seeds = resolve_seeds(args)

    device = torch.device("cpu")
    model_name = "BiGRU"
    dataset_name = args.dataset_name
    optimizer_name = "adam_plateau"
    variant_tag = (
        f"bigru{GRU_HIDDEN_DIM}_layers{GRU_NUM_LAYERS}_pack_LOO"
    )
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
        )
        all_results.append((seed, result))

    mses      = np.array([r["test_mse"]      for _, r in all_results], dtype=np.float64)
    pearsons  = np.array([r["test_pearson"]  for _, r in all_results], dtype=np.float64)
    spearmans = np.array([r["test_spearman"] for _, r in all_results], dtype=np.float64)

    def _mean_std(x: np.ndarray) -> Tuple[float, float]:
        if x.size <= 1:
            return float(x.mean()), 0.0
        return float(x.mean()), float(x.std(ddof=1))

    mse_mean,  mse_std  = _mean_std(mses)
    pr_mean,   pr_std   = _mean_std(pearsons)
    sp_mean,   sp_std   = _mean_std(spearmans)

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
