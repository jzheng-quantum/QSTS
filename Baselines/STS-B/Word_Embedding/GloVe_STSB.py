from typing import Dict, Iterable, List, Set, Tuple
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
EMBEDDING_DIM = 300
BATCH_SIZE = 16
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


def load_data(train_file: str, val_file: str, test_file: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)
    return train_data, val_data, test_data


def build_vocab(data_sources: Iterable[List[Dict]]) -> Set[str]:
    
    vocab: Set[str] = set()
    for split in data_sources:
        for record in split:
            vocab.update(tokenize_text(record["sentence1"]))
            vocab.update(tokenize_text(record["sentence2"]))
    return vocab


class SentencePairDataset(Dataset):
    

    def __init__(self, records: List[Dict], word_to_idx: Dict[str, int]):
        self.word_to_idx = word_to_idx
        self.unk_idx = word_to_idx[UNK_TOKEN]
        self.data = []

        for record in records:
            sent1 = record["sentence1"]
            sent2 = record["sentence2"]
            label = float(record["score"]) / 5.0
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


class GloVeMeanModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: np.ndarray,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0).double()
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.data[0].zero_()
        self.embedding.weight.requires_grad = False

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        x = x * mask
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / lengths

    def forward(self, x_text1: torch.Tensor, x_text2: torch.Tensor) -> torch.Tensor:
        embedded1 = self.embedding(x_text1)
        mask1 = x_text1 != 0
        avg1 = self.masked_mean(embedded1, mask1)

        embedded2 = self.embedding(x_text2)
        mask2 = x_text2 != 0
        avg2 = self.masked_mean(embedded2, mask2)


        cos_sim = F.cosine_similarity(avg1, avg2, dim=1, eps=EPS)
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


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glove_file",
        type=str,
        default="./glove.840B.300d.txt",
    )


    parser.add_argument(
        "--train_file",
        type=str,
        default="./train.jsonl",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="./validation.jsonl",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="./test.jsonl",
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    model_name = "GloVeMean"
    dataset_name = "STSB"
    variant_tag = "glove"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"{model_name}_{dataset_name}_{variant_tag}_seed{args.seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"

    header = (
        f"=== Run started at {run_stamp} ===\n"
        f"model={model_name} (parameter-free baseline)\n"
        f"  - dataset      : STS-B (3 separate JSONL splits, score in [0, 5], normalised by /5.0)\n"
        f"  - embeddings   : GloVe-840B-300d, float64\n"
        f"  - pooling      : masked mean over non-PAD tokens\n"
        f"  - similarity   : cosine, mapped from [-1, 1] to [0, 1]\n"
        f"seed={args.seed}, batch_size={args.batch_size}, device={device}\n"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    train_data, val_data, test_data = load_data(
        args.train_file, args.val_file, args.test_file
    )

    split_summary = (
        f"Split sizes — TRAIN: {len(train_data)}, "
        f"VALIDATION: {len(val_data)}, TEST: {len(test_data)}"
    )
    print(split_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(split_summary + "\n")


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
        f"Vocabulary size (incl. <PAD>,<UNK>): {vocab_size}\n"
        f"Total word types across train, validation, and test: {len(union_vocab)}\n"
        f"  - Validation set: {len(val_vocab)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_vocab)):.2f}%) "
        f"OOV (mapped to <UNK>)\n"
        f"  - Test set: {len(test_vocab)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_vocab)):.2f}%) "
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


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
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


    model = GloVeMeanModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)


    train_mse, train_pearson, train_spearman = evaluate_model(model, train_loader, device)
    val_mse,   val_pearson,   val_spearman   = evaluate_model(model, val_loader,   device)
    test_mse,  test_pearson,  test_spearman  = evaluate_model(model, test_loader,  device)

    train_message = (
        f"[diagnostic] Train MSE: {train_mse:.6f}, "
        f"Pearson: {train_pearson:.6f}, Spearman: {train_spearman:.6f}"
    )
    val_message = (
        f"[diagnostic] Val   MSE: {val_mse:.6f}, "
        f"Pearson: {val_pearson:.6f}, Spearman: {val_spearman:.6f}"
    )
    final_message = (
        f"Final Testing MSE: {test_mse:.6f}, "
        f"Pearson: {test_pearson:.6f}, Spearman: {test_spearman:.6f}"
    )
    print(train_message)
    print(val_message)
    print(final_message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(train_message + "\n")
        f.write(val_message + "\n")
        f.write(final_message + "\n")


if __name__ == "__main__":
    main()
