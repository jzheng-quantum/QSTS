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
import torch.optim as optim
from pennylane.templates import AmplitudeEmbedding
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
NUM_QUBITS = 9
EMBEDDING_DIM = 300
REDUCED_DIM = 16
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


def single_u(params, wires=None):
    qml.RZ(params[0], wires=wires)
    qml.RY(params[1], wires=wires)
    qml.RZ(params[2], wires=wires)


def qsts_circuit(params, wires=None):
    """W4 ablation: each sentence uses 4 qubits, CB entangler repeated x3 (n-1 = 3).
    
    Layout (48 params total):
      sentence A (wires[0..3], 24 params):
        front S layer  : params[0..11]   (4 single_u, 12 params)
        CB(4q)^3       : 3 layers of 4 CNOTs = 12 CNOTs, no params
        back  S layer  : params[12..23]  (4 single_u, 12 params)
      sentence B (wires[4..7], 24 params):
        front S layer  : params[24..35]
        CB(4q)^3       : 3 layers of 4 CNOTs
        back  S layer  : params[36..47]
    """
    # ---------- Sentence A on wires[0..3] ----------
    # Front S layer (12 params)
    single_u(params[0:3],   wires=wires[0])
    single_u(params[3:6],   wires=wires[1])
    single_u(params[6:9],   wires=wires[2])
    single_u(params[9:12],  wires=wires[3])

    # CB(4q)^3 entangling: 3 layers, each layer = 4 CNOTs (descending ring)
    for _ in range(3):
        qml.CNOT(wires=[wires[3], wires[0]])
        qml.CNOT(wires=[wires[2], wires[3]])
        qml.CNOT(wires=[wires[1], wires[2]])
        qml.CNOT(wires=[wires[0], wires[1]])

    # Back S layer (12 params)
    single_u(params[12:15], wires=wires[0])
    single_u(params[15:18], wires=wires[1])
    single_u(params[18:21], wires=wires[2])
    single_u(params[21:24], wires=wires[3])

    # ---------- Sentence B on wires[4..7] ----------
    # Front S layer (12 params)
    single_u(params[24:27], wires=wires[4])
    single_u(params[27:30], wires=wires[5])
    single_u(params[30:33], wires=wires[6])
    single_u(params[33:36], wires=wires[7])

    # CB(4q)^3 entangling
    for _ in range(3):
        qml.CNOT(wires=[wires[7], wires[4]])
        qml.CNOT(wires=[wires[6], wires[7]])
        qml.CNOT(wires=[wires[5], wires[6]])
        qml.CNOT(wires=[wires[4], wires[5]])

    # Back S layer (12 params)
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
            torch.rand(48, generator=quantum_rng, dtype=torch.float64) * 2 * math.pi
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove_file", type=str, default="./glove.840B.300d.txt")
    parser.add_argument("--data_file", type=str, default="./SICK.txt")
    parser.add_argument("--log_dir", type=str, default="./logs_W4_TrainVocab")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient L2 norm for clipping. Set to 0 or negative to disable.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)


    device = torch.device("cpu")
    topology_name = "CB"
    optimizer_name = "adam_plateau"
    dataset_name = "SICK"
    variant_tag = "W4_sandwich_CB3_4qubitsPerSentence_independentparams_frozen_trainvocab"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"QSTS-W4_{dataset_name}_{topology_name}_{optimizer_name}_"
        f"{NUM_QUBITS}qubits_{variant_tag}_seed{args.seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"


    header = (
        f"=== Run started at {run_stamp} ===\n"
        f"variant={variant_tag}\n"
        f"  - ablation        : WIDTH (n=4 qubits per sentence; total wires = 4+4+1 = 9)\n"
        f"  - reduced dim     : 16 (= 2^4, AmplitudeEmbedding)\n"
        f"  - embedding layer : FROZEN (requires_grad=False)\n"
        f"  - vocabulary      : TRAIN ONLY (inductive; OOV in val/test → <UNK>)\n"
        f"  - ansatz          : sandwich  U(θ₁,θ₂) = S(θ₂) · C_{topology_name}^3 · S(θ₁)  (params INDEPENDENT front/back)\n"
        f"  - entangling      : 3 layers of {topology_name} CNOTs per block (n-1 = 3; 4 CNOTs/layer; 12 CNOTs/block)\n"
        f"  - quantum params  : 48 (Block A: 12+12 for front/back S layers; Block B: 12+12; independent A/B)\n"
        f"seed={args.seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")

    train_data, val_data, test_data = load_data(args.data_file)


    vocab = build_vocab([train_data])
    word_to_idx, pretrained_embeddings = load_glove_embeddings(
        args.glove_file, vocab, EMBEDDING_DIM
    )
    vocab_size = len(word_to_idx)

    val_vocab = build_vocab([val_data])
    test_vocab = build_vocab([test_data])
    val_oov = val_vocab - vocab
    test_oov = test_vocab - vocab
    union_vocab = build_vocab([train_data, val_data, test_data])

    glove_summary = (
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}  "
        f"[built from TRAIN ONLY]\n"
        f"For comparison, full TRAIN+TRIAL+TEST union has {len(union_vocab)} word types\n"
        f"  · Val   set has {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>\n"
        f"  · Test  set has {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
        f"are OOV → mapped to <UNK>"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(glove_summary + "\n")

    train_dataset = SentencePairDataset(train_data, word_to_idx)
    val_dataset = SentencePairDataset(val_data, word_to_idx)
    test_dataset = SentencePairDataset(test_data, word_to_idx)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


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
