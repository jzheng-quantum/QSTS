

from typing import Dict, List, Set, Tuple
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup


SEED = 42
REDUCED_DIM = 64
ROBERTA_HIDDEN = 768
MAX_LEN = 128
NUM_EPOCHS = 20
BATCH_SIZE = 16
PATIENCE = 5


ROBERTA_LR = 5e-5
HEAD_LR = 1e-3
ROBERTA_WD = 0.01
HEAD_WD = 1e-5
WARMUP_RATIO = 0.10

EPS = 1e-12


POOLING_CHOICES = ("cls", "mean")
DEFAULT_POOLING = "mean"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


class SentencePairDataset(Dataset):
    

    def __init__(
        self,
        records: List[Dict],
        tokenizer: RobertaTokenizer,
        max_len: int,
        score_scale: float = 5.0,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.score_scale = float(score_scale)
        self.records: List[Tuple[str, str, float]] = []

        raw_scores = []
        for row in records:
            sent_a = str(row["sentence1"])
            sent_b = str(row["sentence2"])
            raw_score = float(row["score"])
            raw_scores.append(raw_score)
            label = raw_score / self.score_scale
            self.records.append((sent_a, sent_b, label))

        if raw_scores:
            self.raw_score_min = float(min(raw_scores))
            self.raw_score_max = float(max(raw_scores))
        else:
            self.raw_score_min = 0.0
            self.raw_score_max = 0.0

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, sentence: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int):
        sent_a, sent_b, label = self.records[idx]
        enc_a = self._encode(sent_a)
        enc_b = self._encode(sent_b)
        return {
            "input_ids_a":      enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b":      enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "label":            torch.tensor(label, dtype=torch.float32),
        }


def stratified_train_val_split(
    train_dataset: SentencePairDataset,
    val_ratio: float = 0.2,
    n_bins: int = 5,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    
    labels = np.array([r[2] for r in train_dataset.records], dtype=np.float64)
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
    bucket_labels = [round(dataset.records[idx][2], 4) for idx in indices]
    unique, counts = np.unique(bucket_labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def collate_fn(batch):
    input_ids_a      = torch.stack([b["input_ids_a"]      for b in batch])
    attention_mask_a = torch.stack([b["attention_mask_a"] for b in batch])
    input_ids_b      = torch.stack([b["input_ids_b"]      for b in batch])
    attention_mask_b = torch.stack([b["attention_mask_b"] for b in batch])
    labels           = torch.stack([b["label"]            for b in batch])
    return (input_ids_a, attention_mask_a,
            input_ids_b, attention_mask_b,
            labels)


class RobertaFineTuneEncoder(nn.Module):
    

    def __init__(self, roberta_model_path: str, reduced_dim: int,
                 pooling: str = DEFAULT_POOLING):
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(
                f"Unknown pooling {pooling!r}; "
                f"expected one of {POOLING_CHOICES}."
            )
        self.pooling = pooling

        self.roberta = RobertaModel.from_pretrained(roberta_model_path)
        for p in self.roberta.parameters():
            p.requires_grad = True


        self.linear = nn.Linear(ROBERTA_HIDDEN, reduced_dim)

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        


        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        masked_sum = (last_hidden_state * mask).sum(dim=1)
        valid_count = mask.sum(dim=1).clamp(min=1.0)
        return masked_sum / valid_count

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        roberta_out = self.roberta(input_ids=input_ids,
                                   attention_mask=attention_mask)
        last_hidden = roberta_out.last_hidden_state
        if self.pooling == "cls":
            sent_vec = last_hidden[:, 0, :]
        elif self.pooling == "mean":
            sent_vec = self._mean_pool(last_hidden, attention_mask)
        else:

            raise RuntimeError(f"Unknown pooling: {self.pooling}")
        out = self.linear(sent_vec)
        return out


    @staticmethod
    def _numel(params) -> int:
        return sum(p.numel() for p in params)

    def count_parameters(self) -> Dict[str, int]:
        roberta_total = self._numel(self.roberta.parameters())
        roberta_trainable = self._numel(p for p in self.roberta.parameters()
                                        if p.requires_grad)
        linear_trainable = self._numel(p for p in self.linear.parameters()
                                       if p.requires_grad)
        return {
            "roberta_total":     roberta_total,
            "roberta_trainable": roberta_trainable,
            "linear_trainable":  linear_trainable,
            "total_trainable":   roberta_trainable + linear_trainable,
        }


class SentenceSimilarityRoBERTa(nn.Module):
    

    def __init__(self, roberta_model_path: str, reduced_dim: int,
                 pooling: str = DEFAULT_POOLING):
        super().__init__()
        self.encoder = RobertaFineTuneEncoder(roberta_model_path, reduced_dim,
                                              pooling=pooling)

    def forward(self,
                input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
                input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor) -> torch.Tensor:
        v1 = self.encoder(input_ids_a, attention_mask_a)
        v2 = self.encoder(input_ids_b, attention_mask_b)
        cos_sim = F.cosine_similarity(v1, v2, dim=1, eps=EPS)
        return (cos_sim + 1.0) / 2.0


def build_param_groups(model: SentenceSimilarityRoBERTa) -> List[Dict]:
    
    no_decay_keywords = ("bias", "LayerNorm.weight", "LayerNorm.bias")

    roberta_decay_params: List[nn.Parameter] = []
    roberta_no_decay_params: List[nn.Parameter] = []
    head_decay_params: List[nn.Parameter] = []
    head_no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(kw in name for kw in no_decay_keywords)
        if name.startswith("encoder.roberta."):
            if is_no_decay:
                roberta_no_decay_params.append(param)
            else:
                roberta_decay_params.append(param)
        elif name.startswith("encoder.linear."):
            if is_no_decay:
                head_no_decay_params.append(param)
            else:
                head_decay_params.append(param)
        else:
            raise RuntimeError(f"Unexpected parameter path: {name}")

    param_groups: List[Dict] = []
    if roberta_decay_params:
        param_groups.append({
            "params": roberta_decay_params,
            "lr": ROBERTA_LR, "weight_decay": ROBERTA_WD,
            "name": "roberta_decay",
        })
    if roberta_no_decay_params:
        param_groups.append({
            "params": roberta_no_decay_params,
            "lr": ROBERTA_LR, "weight_decay": 0.0,
            "name": "roberta_no_decay",
        })
    if head_decay_params:
        param_groups.append({
            "params": head_decay_params,
            "lr": HEAD_LR, "weight_decay": HEAD_WD,
            "name": "head_decay",
        })
    if head_no_decay_params:
        param_groups.append({
            "params": head_no_decay_params,
            "lr": HEAD_LR, "weight_decay": 0.0,
            "name": "head_no_decay",
        })
    return param_groups


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


def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for (input_ids_a, attention_mask_a,
             input_ids_b, attention_mask_b,
             labels) in data_loader:
            input_ids_a      = input_ids_a.to(device)
            attention_mask_a = attention_mask_a.to(device)
            input_ids_b      = input_ids_b.to(device)
            attention_mask_b = attention_mask_b.to(device)

            outputs = model(input_ids_a, attention_mask_a,
                            input_ids_b, attention_mask_b)


            outputs = outputs.detach().cpu().numpy().astype(np.float64)
            predictions.extend(outputs.tolist())
            targets.extend(labels.numpy().astype(np.float64).tolist())

    predictions = np.asarray(predictions, dtype=np.float64)
    targets     = np.asarray(targets,     dtype=np.float64)

    mse      = float(mean_squared_error(targets, predictions))
    pearson  = safe_pearsonr(targets, predictions)
    spearman = safe_spearmanr(targets, predictions)
    return mse, pearson, spearman


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler,
                num_epochs: int,
                patience: int,
                log_path: Path,
                checkpoint_path: Path,
                device: torch.device,
                grad_clip_max_norm: float = 1.0):

    best_val_pearson = float("-inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for (input_ids_a, attention_mask_a,
             input_ids_b, attention_mask_b,
             labels) in train_loader:
            input_ids_a      = input_ids_a.to(device)
            attention_mask_a = attention_mask_a.to(device)
            input_ids_b      = input_ids_b.to(device)
            attention_mask_b = attention_mask_b.to(device)
            labels           = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids_a, attention_mask_a,
                            input_ids_b, attention_mask_b)
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


            scheduler.step()

            total_loss += float(loss.item())
            n_batches += 1

        val_mse, val_pearson, val_spearman = evaluate_model(model, val_loader, device)
        train_loss = total_loss / max(n_batches, 1)

        current_lrs = ", ".join(
            f"{group.get('name', f'g{i}')}={group['lr']:.3g}"
            for i, group in enumerate(optimizer.param_groups)
        )
        message = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {train_loss:.6f}, "
            f"Val MSE: {val_mse:.6f}, Val Pearson: {val_pearson:.6f}, "
            f"Val Spearman: {val_spearman:.6f}, "
            f"LRs: [{current_lrs}]"
        )
        print(message)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_epoch = epoch + 1


            best_model_state = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
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

    model.load_state_dict(best_model_state, strict=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}, "
                f"Best Val Pearson: {best_val_pearson:.6f}\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoBERTa-finetune baseline on STS-12..STS-16 under the LOO pooled STS protocol."
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
    parser.add_argument("--model_path", type=str,
                        default="./roberta-base")
    parser.add_argument("--log_dir",    type=str, default="./logs")
    parser.add_argument("--seeds",      type=int, nargs="+",
                        default=[42],
                        help="One or more random seeds. Multiple seeds produce a "
                             "mean +/- std summary across runs.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience",   type=int, default=PATIENCE)
    parser.add_argument("--max_len",    type=int, default=MAX_LEN)
    parser.add_argument("--grad_clip",  type=float, default=1.0,
                        help="Max gradient L2 norm for clipping. "
                             "Set to 0 or negative to disable.")
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO,
                        help="Fraction of total training steps to do linear "
                             "warmup from 0 to target lr.")
    parser.add_argument("--pooling",    type=str, default=DEFAULT_POOLING,
                        choices=list(POOLING_CHOICES),
                        help="Sentence-pooling strategy on top of fine-tuned RoBERTa. "
                             "'cls' = last_hidden_state[:, 0, :] (`<s>` token, "
                             "RoBERTa's analogue of BERT's [CLS]; Devlin et al. "
                             "2019 classification recipe carried over to RoBERTa). "
                             "'mean' = mean over non-padding tokens with "
                             "attention_mask (Sentence-BERT-style; Reimers & "
                             "Gurevych, 2019). Default: "
                             f"'{DEFAULT_POOLING}'.")
    return parser


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    seeds = list(args.seeds) if args.seeds else [SEED]
    seen = set()
    out: List[int] = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(int(s))
    return out


def run_single_seed(seed: int,
                    train_records: List[Dict],
                    test_records: List[Dict],
                    tokenizer: RobertaTokenizer,
                    args: argparse.Namespace,
                    device: torch.device,
                    log_dir: Path,
                    run_stamp: str,
                    variant_tag: str,
                    model_name: str,
                    dataset_name: str,
                    optimizer_name: str,
                    loo_log_block: str = "") -> Dict[str, float]:

    set_seed(seed)

    base_name = (f"{model_name}_{dataset_name}_{optimizer_name}_"
                 f"{variant_tag}_seed{seed}_{run_stamp}")
    log_path        = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"


    train_dataset = SentencePairDataset(
        train_records, tokenizer, args.max_len, score_scale=args.score_scale
    )
    test_dataset = SentencePairDataset(
        test_records, tokenizer, args.max_len, score_scale=args.score_scale
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


    train_indices, val_indices = stratified_train_val_split(
        train_dataset,
        val_ratio=args.val_ratio,
        n_bins=args.n_bins,
        seed=seed,
    )

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


    model = SentenceSimilarityRoBERTa(
        roberta_model_path=args.model_path,
        reduced_dim=REDUCED_DIM,
        pooling=args.pooling,
    ).to(device)
    criterion = nn.MSELoss()


    param_groups = build_param_groups(model)
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = max(1, int(round(args.warmup_ratio * total_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


    if args.pooling == "cls":
        pooling_desc = ("last_hidden_state[:, 0, :]  "
                        "(i.e. `<s>` token, RoBERTa's analogue of BERT's [CLS]; "
                        "Devlin et al. 2019 classification recipe)")
    elif args.pooling == "mean":
        pooling_desc = ("mean over non-padding tokens of last_hidden_state "
                        "(Sentence-BERT-style, masked by attention_mask)")
    else:
        pooling_desc = f"<unknown pooling: {args.pooling}>"


    pc = model.encoder.count_parameters()
    header = (
        f"=== Run started at {run_stamp} (seed={seed}) ===\n"
        f"model={model_name}\n"
        f"dataset={dataset_name}\n"
        f"protocol=Leave-One-Year-Out (LOO) pooled STS\n"
        f"variant={variant_tag}\n"
        f"  - tokenizer       : RobertaTokenizer (BPE), max_len={args.max_len}\n"
        f"  - encoder         : RoBERTa-base, fully fine-tuned (all params trainable)\n"
        f"  - pooling         : {args.pooling}\n"
        f"  - sentence vector : {pooling_desc}\n"
        f"  - head            : Linear({ROBERTA_HIDDEN}->{REDUCED_DIM}), trainable, fp32\n"
        f"  - similarity      : (cos(v1, v2) + 1) / 2   in [0, 1]\n"
        f"  - two-tower       : A and B encoded independently (no cross-attention)\n"
        f"  - optimizer       : AdamW, betas=(0.9, 0.999), eps=1e-8\n"
        f"  - RoBERTa lr         : {ROBERTA_LR}  (weight_decay={ROBERTA_WD}, "
        f"LayerNorm/bias excluded from wd)\n"
        f"  - head lr         : {HEAD_LR}  (weight_decay={HEAD_WD})\n"
        f"  - scheduler       : linear warmup + linear decay, "
        f"warmup_ratio={args.warmup_ratio}\n"
        f"  - total steps     : {total_steps}  (warmup {warmup_steps})\n"
        f"  - precision       : fp32 (RoBERTa native)\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}\n"
        f"Parameter breakdown:\n"
        f"  - RoBERTa (trainable)             : {pc['roberta_trainable']:,}\n"
        f"  - Linear({ROBERTA_HIDDEN}->{REDUCED_DIM}) : {pc['linear_trainable']:,}\n"
        f"  - Total trainable              : {pc['total_trainable']:,}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


        if loo_log_block:
            f.write(loo_log_block + "\n")


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

    return {"test_mse":      test_mse,
            "test_pearson":  test_pearson,
            "test_spearman": test_spearman}


def main() -> None:
    args = build_argparser().parse_args()
    seeds = resolve_seeds(args)

    device = torch.device("cpu")
    model_name     = "RoBERTa"
    dataset_name   = args.dataset_name
    optimizer_name = "adamw_warmup_linear"
    variant_tag    = f"finetune_{args.pooling}pool_LOO"
    run_stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


    train_records, test_records, loo_stats = load_data_loo(
        train_files=args.train_files,
        test_file=args.test_file,
        leave_out_test_file=args.leave_out_test_file,
    )


    loo_log_block = format_loo_stats(loo_stats, args.dataset_name)
    print(loo_log_block)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)


    all_results: List[Tuple[int, Dict[str, float]]] = []
    for idx, seed in enumerate(seeds):
        print("\n" + "=" * 72)
        print(f"  Starting seed {seed}  ({idx + 1}/{len(seeds)})")
        print("=" * 72)
        result = run_single_seed(
            seed=seed,
            train_records=train_records,
            test_records=test_records,
            tokenizer=tokenizer,
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

    mse_mean, mse_std = _mean_std(mses)
    pr_mean,  pr_std  = _mean_std(pearsons)
    sp_mean,  sp_std  = _mean_std(spearmans)

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
        f"MSE = {mse_mean:.4f} +/- {mse_std:.4f},  "
        f"Pearson = {pr_mean:.4f} +/- {pr_std:.4f},  "
        f"Spearman = {sp_mean:.4f} +/- {sp_std:.4f}"
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
