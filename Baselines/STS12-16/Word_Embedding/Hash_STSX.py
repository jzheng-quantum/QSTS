from typing import Dict, Iterable, List, Set, Tuple
import argparse
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
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import DataLoader, Dataset, Subset


SEED = 42
BATCH_SIZE = 16
HASH_N_FEATURES = 4096  


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
    

    def __init__(self, records: List[Dict], score_scale: float = 5.0):
        self.score_scale = float(score_scale)
        self.data = [
            (
                str(row["sentence1"]),
                str(row["sentence2"]),
                float(row["score"]) / self.score_scale,
            )
            for row in records
        ]

        if records:
            raw_scores = [float(row["score"]) for row in records]
            self.raw_score_min = float(min(raw_scores))
            self.raw_score_max = float(max(raw_scores))
        else:
            self.raw_score_min = 0.0
            self.raw_score_max = 0.0

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
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    return list(sents1), list(sents2), labels_tensor


def prepare_hash_features(
    sentences: List[str],
    vectorizer: HashingVectorizer,
) -> torch.Tensor:
    hashed = vectorizer.transform(sentences)  
    return torch.from_numpy(hashed.toarray().astype(np.float64))

class HashingModel(nn.Module):
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
    vectorizer: HashingVectorizer,
    device: torch.device,
):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sents1, sents2, labels in data_loader:
            feats1 = prepare_hash_features(sents1, vectorizer).to(device)
            feats2 = prepare_hash_features(sents2, vectorizer).to(device)
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
    parser = argparse.ArgumentParser(
        description="Hashing (parameter-free) baseline on STS-12..STS-16 "
                    "under the Leave-One-Year-Out (LOO) pooled STS protocol."
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
        help="Dataset tag for log filenames (e.g. STS-12, STS-14, STS-15, STS-16).",
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--n_features",
        type=int,
        default=HASH_N_FEATURES,
        help="Number of hashing buckets. Must be a power of two.",
    )
    return parser


def _check_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if not _check_power_of_two(args.n_features):
        raise ValueError(
            f"--n_features must be a power of two (sklearn guidance for "
            f"HashingVectorizer); got {args.n_features}."
        )

    device = torch.device("cpu")
    model_name = "Hash"
    dataset_name = args.dataset_name
    variant_tag = f"hash_nfeat{args.n_features}_LOO"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = (
        f"{model_name}_{dataset_name}_LOO_{variant_tag}_seed{args.seed}_{run_stamp}"
    )
    log_path = log_dir / f"{base_name}.log"

    header = (
        f"=== Run started at {run_stamp} ===\n"
        f"model={model_name} (parameter-free baseline)\n"
        f"dataset={dataset_name}\n"
        f"protocol=Leave-One-Year-Out (LOO) pooled STS\n"
        f"variant={variant_tag}\n"
        f"  - features    : n_features={args.n_features} "
        f"(HashingVectorizer is stateless, no fit step)\n"
        f"  - similarity  : cosine, mapped from [-1, 1] to [0, 1]\n"
        f"  - protocol    : training set = atomic SemEval STS files\n"
        f"                  (D12_train + the four D_X_test for X != Y),\n"
        f"                  strictly deduplicated and with the target\n"
        f"                  year's test pairs removed; test set is the\n"
        f"                  target year's official test.jsonl, verbatim.\n"
        f"  - val split   : stratified {args.val_ratio:.2f} of training set, "
        f"KBins(n_bins={args.n_bins}, uniform)\n"
        f"  - score scale : raw label divided by {args.score_scale} "
        f"(STS-12..STS-16 use 0..5)\n"
        f"seed={args.seed}, batch_size={args.batch_size}, device={device}"
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

    train_dataset = SentencePairDataset(train_records, score_scale=args.score_scale)
    test_dataset = SentencePairDataset(test_records, score_scale=args.score_scale)


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


    train_indices, val_indices = compute_split_indices_from_records(
        train_records,
        val_ratio=args.val_ratio,
        n_bins=args.n_bins,
        seed=args.seed,
        score_scale=args.score_scale,
    )
    train_records_for_vocab = [train_records[i] for i in train_indices]
    val_records_holdout = [train_records[i] for i in val_indices]


    vectorizer = HashingVectorizer(
        n_features=args.n_features,
        tokenizer=tokenize_text,
        lowercase=False,
        token_pattern=None,
        alternate_sign=True,
        norm="l2",
        dtype=np.float64,
    )

    train_for_vocab_tokens = build_vocab([train_records_for_vocab])
    val_tokens = build_vocab([val_records_holdout])
    test_tokens = build_vocab([test_records])
    union_tokens = build_vocab([train_records, test_records])
    val_oov = val_tokens - train_for_vocab_tokens
    test_oov = test_tokens - train_for_vocab_tokens
    collision_ratio = len(train_for_vocab_tokens) / float(args.n_features)

    vocab_summary = (
        f"Unique tokens in training set: {len(train_for_vocab_tokens)}\n"
        f"Total word types across training and test: {len(union_tokens)}\n"
        f"  - Validation set: {len(val_tokens)} word types; "
        f"{len(val_oov)} ({100.0 * len(val_oov) / max(1, len(val_tokens)):.2f}%) "
        f"are unseen in training (HashingVectorizer is stateless, so they "
        f"are still hashed but to buckets the train-side did not occupy)\n"
        f"  - Test set: {len(test_tokens)} word types; "
        f"{len(test_oov)} ({100.0 * len(test_oov) / max(1, len(test_tokens)):.2f}%) "
        f"are unseen in training (same caveat as above)\n"
        f"Hash buckets (n_features):                 {args.n_features}\n"
        f"Tokens-per-bucket (collision pressure, train-side): {collision_ratio:.4f}"
    )
    print(vocab_summary)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(vocab_summary + "\n")


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

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
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


    model = HashingModel().to(device)

    train_mse, train_pearson, train_spearman = evaluate_model(
        model, train_loader, vectorizer, device
    )
    val_mse, val_pearson, val_spearman = evaluate_model(
        model, val_loader, vectorizer, device
    )
    test_mse, test_pearson, test_spearman = evaluate_model(
        model, test_loader, vectorizer, device
    )

    train_message = (
        f"[diagnostic] Train subset MSE: {train_mse:.6f}, "
        f"Pearson: {train_pearson:.6f}, Spearman: {train_spearman:.6f}"
    )
    val_message = (
        f"[diagnostic] Val   subset MSE: {val_mse:.6f}, "
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
