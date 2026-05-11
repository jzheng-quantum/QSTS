from typing import Iterable, List, Set, Tuple
import argparse
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 16

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


class SentencePairDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = [
            (
                str(row["sentence_A"]),
                str(row["sentence_B"]),
                (float(row["relatedness_score"]) - 1.0) / 4.0,
            )
            for _, row in data.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate_fn(batch):
    sents1, sents2, labels = zip(*batch)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    return list(sents1), list(sents2), labels_tensor



def prepare_bow_features(
    sentences: List[str],
    vectorizer: CountVectorizer,
) -> torch.Tensor:
    bow = vectorizer.transform(sentences)
    bow = normalize(bow, norm="l2", axis=1, copy=False)
    return torch.from_numpy(bow.toarray().astype(np.float64))



class BoWModel(nn.Module):
    def forward(
        self,
        x_text1: torch.Tensor,  
        x_text2: torch.Tensor,  
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(x_text1, x_text2, dim=1, eps=1e-12)
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
    vectorizer: CountVectorizer,
    device: torch.device,
):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sents1, sents2, labels in data_loader:
            feats1 = prepare_bow_features(sents1, vectorizer).to(device)
            feats2 = prepare_bow_features(sents2, vectorizer).to(device)
            outputs = model(feats1, feats2).detach().cpu().numpy().astype(np.float64)
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
        "--data_file",
        type=str,
        default="./SICK.txt",
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    model_name = "BoW"
    dataset_name = "SICK"
    variant_tag = "bow"
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
        f"  - features     : raw counts, L2-normalised per sentence\n"
        f"  - similarity   : cosine, mapped from [-1, 1] to [0, 1]\n"
        f"seed={args.seed}, batch_size={args.batch_size}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    train_data, val_data, test_data = load_data(args.data_file)


    vectorizer = CountVectorizer(
        tokenizer=tokenize_text,
        lowercase=False,
        token_pattern=None,
    )
    train_corpus = pd.concat(
        [train_data["sentence_A"], train_data["sentence_B"]],
        ignore_index=True,
    ).astype(str).tolist()
    vectorizer.fit(train_corpus)

    vocab_size = len(vectorizer.vocabulary_)


    train_token_set = build_vocab([train_data])
    val_token_set = build_vocab([val_data])
    test_token_set = build_vocab([test_data])
    val_oov = val_token_set - train_token_set
    test_oov = test_token_set - train_token_set
    union_token_set = build_vocab([train_data, val_data, test_data])

    vocab_summary = (
        f"Vocabulary size (BoW): {vocab_size}\n"
        f"Total word types across train, validation, and test: {len(union_token_set)}\n"
        f"  - Validation set: {len(val_token_set)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_token_set)):.2f}%) "
        f"OOV (dropped by CountVectorizer)\n"
        f"  - Test set: {len(test_token_set)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_token_set)):.2f}%) "
        f"OOV (dropped by CountVectorizer)"
    )
    print(vocab_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(vocab_summary + "\n")


    test_dataset = SentencePairDataset(test_data)
    print(f"Number of testing samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


    model = BoWModel().to(device)

    test_mse, test_pearson, test_spearman = evaluate_model(
        model, test_loader, vectorizer, device
    )
    final_message = (
        f"Final Testing MSE: {test_mse:.6f}, "
        f"Pearson: {test_pearson:.6f}, Spearman: {test_spearman:.6f}"
    )
    print(final_message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(final_message + "\n")


if __name__ == "__main__":
    main()
