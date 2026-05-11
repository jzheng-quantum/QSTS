
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
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pennylane.templates import AmplitudeEmbedding
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


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


NUM_U_PARAMS = 15
NUM_QUANTUM_PARAMS = NUM_U_PARAMS


READOUT_DIM = NUM_QUBITS

DEFAULT_SEEDS = [0, 1, 2, 3, 42]


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


dev = qml.device("default.qubit", wires=NUM_QUBITS)


def _single_qubit_u(params3, wire):
    
    qml.RZ(params3[0], wires=wire)
    qml.RY(params3[1], wires=wire)
    qml.RZ(params3[2], wires=wire)


def _conv_U(params, wire_a, wire_b):
    

    _single_qubit_u(params[0:3], wire_a)
    _single_qubit_u(params[3:6], wire_b)

    qml.CNOT(wires=[wire_b, wire_a])

    qml.RZ(params[12], wires=wire_a)
    qml.RY(params[13], wires=wire_b)

    qml.CNOT(wires=[wire_a, wire_b])

    qml.RY(params[14], wires=wire_b)

    qml.CNOT(wires=[wire_b, wire_a])

    _single_qubit_u(params[6:9], wire_a)
    _single_qubit_u(params[9:12], wire_b)


def _qcnn_block(params, wires):
    
    assert len(wires) == 6, "QCNN block requires exactly 6 qubits."
    u_params = params[0:NUM_U_PARAMS]


    _conv_U(u_params, wires[0], wires[1])
    _conv_U(u_params, wires[2], wires[3])
    _conv_U(u_params, wires[4], wires[5])

    _conv_U(u_params, wires[1], wires[2])
    _conv_U(u_params, wires[3], wires[4])


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_encoder(amplitudes, params):
    
    AmplitudeEmbedding(
        features=amplitudes,
        wires=range(NUM_QUBITS),
        normalize=True,
        pad_with=0.0,
    )
    _qcnn_block(params, wires=list(range(NUM_QUBITS)))

    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]


class QCNNSentenceEncoder(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
        quantum_param_seed: int,
    ):
        super().__init__()
        assert reduced_dim == 2 ** NUM_QUBITS, (
            f"reduced_dim ({reduced_dim}) must equal 2**NUM_QUBITS "
            f"({2 ** NUM_QUBITS}) for amplitude encoding."
        )


        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False


        self.linear = nn.Linear(embedding_dim, reduced_dim).double()


        quantum_rng = torch.Generator()
        quantum_rng.manual_seed(int(quantum_param_seed))
        self.params = nn.Parameter(
            torch.rand(NUM_QUANTUM_PARAMS, generator=quantum_rng, dtype=torch.float64)
            * 2.0 * math.pi
        )


        self.quantum = quantum_encoder

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

                z_vec = z_exps.reshape(READOUT_DIM)
            outputs.append(z_vec.reshape(1, READOUT_DIM))
        return torch.cat(outputs, dim=0)


class QCNNSentenceSimilarity(nn.Module):
    

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        reduced_dim: int,
        pretrained_embeddings: np.ndarray,
        quantum_param_seed: int,
    ):
        super().__init__()
        self.encoder = QCNNSentenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            reduced_dim=reduced_dim,
            pretrained_embeddings=pretrained_embeddings,
            quantum_param_seed=quantum_param_seed,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_file", type=str,
                        default="./glove.840B.300d.txt")
    parser.add_argument("--data_file", type=str,
                        default="./SICK.txt")
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
        f"  - embedding       : GloVe 840B 300d, frozen, padding_idx=0\n"
        f"  - linear          : Linear({EMBEDDING_DIM}->{REDUCED_DIM}), trainable\n"
        f"  - quantum encoder : QCNN on {NUM_QUBITS} qubits (shared across sent_A/sent_B)\n"
        f"                      AmplitudeEmbedding(64->6q) +\n"
        f"                      Conv U layer (shared 15 params, 5 pairs, 3 CNOTs each)\n"
        f"  - quantum params  : {NUM_QUANTUM_PARAMS}  "
        f"({NUM_U_PARAMS} shared U params, conv-only)\n"
        f"  - readout         : [<Z_q>  for q in {list(range(NUM_QUBITS))}]  "
        f"-> {READOUT_DIM}-dim sentence embedding\n"
        f"  - similarity head : cosine_similarity(v1, v2)/2 + 0.5\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    train_vocab_diag = build_vocab([train_data])
    val_vocab_diag = build_vocab([val_data])
    test_vocab_diag = build_vocab([test_data])
    val_oov_diag = val_vocab_diag - train_vocab_diag
    test_oov_diag = test_vocab_diag - train_vocab_diag
    union_vocab_diag = build_vocab([train_data, val_data, test_data])

    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}\n"
        f"Total word types across train, validation, and test: {len(union_vocab_diag)}\n"
        f"  - Validation set: {len(val_vocab_diag)} word types; "
        f"{len(val_oov_diag)} ({100.0 * len(val_oov_diag) / max(1, len(val_vocab_diag)):.2f}%) "
        f"OOV (mapped to <UNK>)\n"
        f"  - Test set: {len(test_vocab_diag)} word types; "
        f"{len(test_oov_diag)} ({100.0 * len(test_oov_diag) / max(1, len(test_vocab_diag)):.2f}%) "
        f"OOV (mapped to <UNK>)"
    )
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


    quantum_param_seed = seed + 2024

    model = QCNNSentenceSimilarity(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        reduced_dim=REDUCED_DIM,
        pretrained_embeddings=pretrained_embeddings,
        quantum_param_seed=quantum_param_seed,
    ).to(device)
    criterion = nn.MSELoss()


    n_linear = sum(p.numel() for p in model.encoder.linear.parameters()
                   if p.requires_grad)
    n_quantum = model.encoder.params.numel()
    pc_lines = [
        "Parameter breakdown:",
        f"  - Linear({EMBEDDING_DIM}->{REDUCED_DIM}) : {n_linear:,}",
        f"  - Quantum QCNN params        : {n_quantum:,}  "
        f"({NUM_U_PARAMS} shared U, conv-only)",
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

    device = torch.device("cpu")
    model_name = "QCNN_Hybrid"
    dataset_name = "SICK"
    optimizer_name = "adam_plateau"
    variant_tag = f"qcnn_q{NUM_QUBITS}_sharedU15"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


    train_data, val_data, test_data = load_data(args.data_file)
    vocab = build_vocab([train_data])
    word_to_idx, hit_vectors, _hits = prepare_glove_hits_once(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)


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
    print(glove_summary)

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
