from typing import Dict, Iterable, List, Set, Tuple
import argparse
import copy
import math
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
EMBEDDING_DIM = 300
REDUCED_DIM = 64
ATTN_D_MODEL = 64
ATTN_NUM_HEADS = 1
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
        embeddings[word_to_idx[word]] = vec
    return embeddings


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


class SentencePairDataset(Dataset):
    def __init__(self, data: pd.DataFrame, word_to_idx: Dict[str, int]):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.data = []

        for _, row in data.iterrows():
            sent1 = row["sentence_A"]
            sent2 = row["sentence_B"]
            label = (float(row["relatedness_score"]) - 1.0) / 4.0
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


class SelfAttentionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        attn_d_model: int,
        attn_num_heads: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        if attn_num_heads != 1:
            raise NotImplementedError(
                f"This baseline only supports single-head attention "
                f"(got attn_num_heads={attn_num_heads})."
            )
        if attn_d_model != reduced_dim:
            raise ValueError(
                f"attn_d_model ({attn_d_model}) must equal reduced_dim "
                f"({reduced_dim})."
            )

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False

        self.linear = nn.Linear(embedding_dim, reduced_dim).double()

        self.d_model = attn_d_model

        self.W_Q = nn.Linear(attn_d_model, attn_d_model, bias=True).double()
        self.W_K = nn.Linear(attn_d_model, attn_d_model, bias=True).double()
        self.W_V = nn.Linear(attn_d_model, attn_d_model, bias=True).double()
        self.W_O = nn.Linear(attn_d_model, attn_d_model, bias=True).double()

        self._attn_scale = 1.0 / math.sqrt(attn_d_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask_f
        lengths = mask_f.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != 0)
        mask_f = mask.unsqueeze(-1)

        emb = self.embedding(x)
        emb = self.linear(emb)
        x0 = emb * mask_f.to(dtype=emb.dtype)

        q = self.W_Q(x0)
        k = self.W_K(x0)
        v = self.W_V(x0)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self._attn_scale

        key_mask = mask.unsqueeze(1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~key_mask, neg_inf)

        attn = F.softmax(scores, dim=-1)

        ctx = torch.matmul(attn, v)
        ctx = self.W_O(ctx)
        ctx = ctx * mask_f.to(dtype=ctx.dtype)

        pooled = self._masked_mean(ctx, mask)
        return pooled

    def linear_parameters(self):
        return self.linear.parameters()

    def attention_parameters(self):
        return (list(self.W_Q.parameters())
                + list(self.W_K.parameters())
                + list(self.W_V.parameters())
                + list(self.W_O.parameters()))

    @staticmethod
    def _numel(params) -> int:
        return sum(p.numel() for p in params)

    def count_parameters(self) -> Dict[str, int]:
        lin = self._numel(p for p in self.linear_parameters() if p.requires_grad)
        attn = self._numel(p for p in self.attention_parameters() if p.requires_grad)
        return {
            "embedding": 0,
            "linear": lin,
            "attention": attn,
            "total_trainable": lin + attn,
        }


class SentenceSimilaritySelfAttn(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        attn_d_model: int,
        attn_num_heads: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.encoder = SelfAttentionEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            attn_d_model=attn_d_model,
            attn_num_heads=attn_num_heads,
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
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("Training loss became NaN or Inf.")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_file", type=str,
                        default="./glove.840B.300d.txt")
    parser.add_argument("--data_file", type=str,
                        default="./SICK.txt")
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
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
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
) -> Dict[str, float]:
    set_seed(seed)
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
        f"variant={variant_tag}\n"
        f"  - embedding       : GloVe, padding_idx=0\n"
        f"  - linear          : Linear({EMBEDDING_DIM}->{REDUCED_DIM})\n"
        f"  - self-attention  : single-head, d_model={ATTN_D_MODEL}, "
        f"Q/K/V/W_O each Linear({ATTN_D_MODEL}->{ATTN_D_MODEL})\n"
        f"  - pooling         : masked mean over tokens\n"
        f"  - similarity      : cosine then (cos+1)/2  -> [0, 1]\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")

    train_dataset = SentencePairDataset(train_data, word_to_idx)
    val_dataset = SentencePairDataset(val_data, word_to_idx)
    test_dataset = SentencePairDataset(test_data, word_to_idx)

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

    model = SentenceSimilaritySelfAttn(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        attn_d_model=ATTN_D_MODEL,
        attn_num_heads=ATTN_NUM_HEADS,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)
    criterion = nn.MSELoss()

    pc = model.encoder.count_parameters()
    pc_lines = [
        "Parameter breakdown:",
        f"  - Linear({EMBEDDING_DIM}->{REDUCED_DIM}) : {pc['linear']:,}",
        f"  - Self-attention (Q/K/V/W_O) : {pc['attention']:,}",
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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
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
    model_name = "SANN"
    dataset_name = "SICK"
    optimizer_name = "adam_plateau"
    variant_tag = (
        "sann"
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_data(args.data_file)

    vocab = build_vocab([train_data])
    word_to_idx, hit_vectors, _hits = prepare_glove_hits_once(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}  ")

    val_vocab = build_vocab([val_data])
    test_vocab = build_vocab([test_data])
    val_oov = val_vocab - vocab
    test_oov = test_vocab - vocab
    union_vocab = build_vocab([train_data, val_data, test_data])

    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}\n"
        f"Total word types across train, validation, and test: {len(union_vocab)}\n"
        f"  - Validation set: {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)\n"
        f"  - Test set: {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)"
    )

    all_results: List[Tuple[int, Dict[str, float]]] = []
    for idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"  Starting seed {seed}  ({idx + 1}/{len(seeds)})")
        print("=" * 72)
        result = run_single_seed(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
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
