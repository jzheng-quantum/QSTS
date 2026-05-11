

import argparse
import os


os.environ.setdefault("HF_HUB_OFFLINE",       "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel


SIMCSE_HIDDEN      = 768
MAX_LEN            = 128
BATCH_SIZE         = 32
EPS                = 1e-12
PROB_CLAMP_EPS     = 1e-7


SIMCSE_VARIANTS    = ("unsup", "sup")
DEFAULT_SIMCSE_VARIANT = "sup"


def simcse_model_dir(base_dir: str, variant: str) -> str:
    
    if variant not in SIMCSE_VARIANTS:
        raise ValueError(
            f"Unknown SimCSE variant {variant!r}; "
            f"expected one of {SIMCSE_VARIANTS}."
        )
    return str(Path(base_dir) / f"{variant}-simcse-bert-base-uncased")


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
    df["question1"]    = df["question1"].astype(str)
    df["question2"]    = df["question2"].astype(str)
    return df[["question1", "question2", "is_duplicate"]]


class QQPSentencePairDataset(Dataset):
    

    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: List[Tuple[str, str, float]] = []
        for _, row in data.iterrows():
            sent_a = str(row["question1"])
            sent_b = str(row["question2"])
            label  = float(int(row["is_duplicate"]))
            self.records.append((sent_a, sent_b, label))

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, sentence: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
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
            "label":            torch.tensor(label, dtype=torch.float64),
        }


def collate_fn(batch):
    input_ids_a      = torch.stack([b["input_ids_a"]      for b in batch])
    attention_mask_a = torch.stack([b["attention_mask_a"] for b in batch])
    input_ids_b      = torch.stack([b["input_ids_b"]      for b in batch])
    attention_mask_b = torch.stack([b["attention_mask_b"] for b in batch])
    labels           = torch.stack([b["label"]            for b in batch])
    return (input_ids_a, attention_mask_a,
            input_ids_b, attention_mask_b,
            labels)


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


def compute_bce(probs: np.ndarray, targets: np.ndarray) -> float:
    
    if probs.size == 0 or targets.size == 0:
        return 0.0
    p = np.clip(probs.astype(np.float64), PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS)
    y = targets.astype(np.float64)
    bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(bce))


@torch.no_grad()
def evaluate_qqp_dev(model: BertModel,
                     loader: DataLoader,
                     device: torch.device,
                     simcse_variant: str,
                     log_path: Path = None) -> Dict[str, float]:
    
    if simcse_variant not in SIMCSE_VARIANTS:
        raise ValueError(
            f"Unknown simcse_variant {simcse_variant!r}; "
            f"expected one of {SIMCSE_VARIANTS}."
        )

    model.eval()
    probs_list:   List[float] = []
    targets_list: List[float] = []

    n_total = len(loader.dataset)
    n_done = 0
    last_pct = -1

    for (input_ids_a, attention_mask_a,
         input_ids_b, attention_mask_b,
         labels) in loader:
        input_ids_a      = input_ids_a.to(device)
        attention_mask_a = attention_mask_a.to(device)
        input_ids_b      = input_ids_b.to(device)
        attention_mask_b = attention_mask_b.to(device)

        out_a = model(input_ids=input_ids_a, attention_mask=attention_mask_a)
        out_b = model(input_ids=input_ids_b, attention_mask=attention_mask_b)


        if simcse_variant == "sup":


            v_a = out_a.pooler_output
            v_b = out_b.pooler_output
        elif simcse_variant == "unsup":


            v_a = out_a.last_hidden_state[:, 0, :]
            v_b = out_b.last_hidden_state[:, 0, :]
        else:

            raise RuntimeError(f"Unknown simcse_variant: {simcse_variant}")


        cos = F.cosine_similarity(v_a, v_b, dim=1, eps=EPS)
        sim = (cos + 1.0) / 2.0

        probs_list  .extend(sim.detach().cpu().numpy().astype(np.float64).tolist())
        targets_list.extend(labels.numpy().astype(np.float64).tolist())

        n_done += labels.numel()
        pct = (n_done * 100) // max(n_total, 1)
        if pct - last_pct >= 5:
            last_pct = pct
            msg = f"  [dev] {n_done}/{n_total} ({pct:>3d}%)"
            print(msg, flush=True)
            if log_path is not None:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

    probs_np   = np.asarray(probs_list,   dtype=np.float64)
    targets_np = np.asarray(targets_list, dtype=np.float64)

    bce      = compute_bce(probs_np, targets_np)
    auc      = safe_auc(targets_np, probs_np)
    accuracy = safe_accuracy(targets_np, probs_np, threshold=0.5)
    f1       = safe_f1(targets_np, probs_np, threshold=0.5)

    return {
        "n":        int(targets_np.size),
        "bce":      bce,
        "auc":      auc,
        "accuracy": accuracy,
        "f1":       f1,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qqp_dev_file", type=str,
                        required=True,
                        help="Path to the QQP dev TSV.  GLUE original or "
                             "HuggingFace header conventions both accepted.")
    parser.add_argument("--model_base_dir", type=str,
                        required=True,
                        help="Directory that CONTAINS the SimCSE variant dirs.")
    parser.add_argument("--simcse_variant", type=str,
                        default=DEFAULT_SIMCSE_VARIANT,
                        choices=list(SIMCSE_VARIANTS),
                        help=f"Which SimCSE checkpoint to load.  "
                             f"Default: '{DEFAULT_SIMCSE_VARIANT}'.")
    parser.add_argument("--log_dir",    type=str,
                        default="./logs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_len",    type=int, default=MAX_LEN)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cpu")

    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"SimCSE-{args.simcse_variant}"
    log_dir    = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


    pooler_short_tag = ("clspool" if args.simcse_variant == "sup"
                        else "clsbeforepoolerpool")

    base_name = (f"{model_name}_QQP_zeroshot_{pooler_short_tag}_"
                 f"maxlen{args.max_len}_thr0.5_{run_stamp}")
    log_path = log_dir / f"{base_name}.log"


    model_path = simcse_model_dir(args.model_base_dir, args.simcse_variant)
    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"SimCSE model directory not found: {model_path}"
        )

    if args.simcse_variant == "sup":
        pooling_label = "cls (with pooler MLP, official sup-SimCSE protocol)"
        pooling_desc  = ("pooler_output  (i.e. [CLS] token passed through "
                         "BERT's pretrained pooler Linear+Tanh; this MLP is "
                         "supervised by the contrastive loss in sup-SimCSE)")
    elif args.simcse_variant == "unsup":
        pooling_label = ("cls_before_pooler (without pooler MLP, official "
                         "unsup-SimCSE protocol)")
        pooling_desc  = ("last_hidden_state[:, 0, :]  (i.e. [CLS] token "
                         "BEFORE the pooler MLP; unsup-SimCSE drops the "
                         "pooler at inference, see --mlp_only_train)")
    else:
        pooling_label = f"<unknown simcse_variant: {args.simcse_variant}>"
        pooling_desc  = pooling_label

    header = (
        f"=== Zero-shot SimCSE evaluation on QQP ===\n"
        f"run_stamp={run_stamp}\n"
        f"model={model_name}\n"
        f"  - variant         : {args.simcse_variant} "
        f"(princeton-nlp/{args.simcse_variant}-simcse-bert-base-uncased)\n"
        f"  - model path      : {model_path}\n"
        f"  - architecture    : BertModel (BERT-base backbone, hidden_size={SIMCSE_HIDDEN})\n"
        f"  - tokenizer       : AutoTokenizer (BertTokenizerFast / WordPiece), "
        f"max_len={args.max_len}\n"
        f"  - vocabulary      : pretrained SimCSE WordPiece (frozen vocab)\n"
        f"  - input prefix    : (none) — SimCSE was not trained with prefixes\n"
        f"  - pooling         : {pooling_label}\n"
        f"  - sentence vector : {pooling_desc}\n"
        f"  - similarity      : (cos(v1, v2) + 1) / 2   in [0, 1], "
        f"computed via F.cosine_similarity (no L2 normalisation step)\n"
        f"  - decision rule   : duplicate iff score >= 0.5  "
        f"(Protocol-(a), GLUE-clean; matches QSTS-V4 main-table threshold)\n"
        f"  - protocol        : ZERO-SHOT (no trainable parameters, "
        f"no gradient updates, no random seed)\n"
        f"  - trainable params: 0\n"
        f"  - two-tower       : sentences encoded independently, cosine compared\n"
        f"  - dataset         : QQP (GLUE)\n"
        f"  - dev (test) file : {args.qqp_dev_file}\n"
        f"  - eval metrics    : BCE (clamped), AUC-ROC (threshold-free), "
        f"Accuracy, F1 (pos_label=1) at threshold 0.5\n"
        f"  - offline mode    : HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}, "
        f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}, "
        f"local_files_only=True\n"
        f"  - device          : {device}\n"
        f"  - batch_size      : {args.batch_size}  (inference only)"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    pc_lines = [
        "Parameter breakdown:",
        f"  - SimCSE-{args.simcse_variant} (frozen, non-trainable) : {total_params:,}",
        f"  - Trainable                  : 0",
    ]
    for line in pc_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in pc_lines:
            f.write(line + "\n")


    dev_path = Path(args.qqp_dev_file)
    print()
    print(f"Loading QQP dev from {dev_path} ...", flush=True)
    dev_full_df = _load_qqp_tsv(dev_path)
    n_dev_full   = len(dev_full_df)
    dev_pos_frac = float(dev_full_df["is_duplicate"].mean())

    data_lines = [
        f"  QQP dev (test)    : {n_dev_full} pairs  (pos frac {dev_pos_frac:.4f})",
    ]
    for line in data_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in data_lines:
            f.write(line + "\n")


    print()
    print(f"Evaluating QQP dev (n={n_dev_full}) zero-shot ...", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\nEvaluating QQP dev (n={n_dev_full}) zero-shot ...\n")

    dataset = QQPSentencePairDataset(dev_full_df, tokenizer, args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, collate_fn=collate_fn)

    t0 = datetime.now()
    metrics = evaluate_qqp_dev(model, loader, device,
                               simcse_variant=args.simcse_variant,
                               log_path=log_path)
    secs = (datetime.now() - t0).total_seconds()
    pairs_per_sec = n_dev_full / max(secs, 1e-6)


    final_lines: List[str] = []
    final_lines.append("")
    final_lines.append("=" * 72)
    final_lines.append(
        f"FINAL  |  model={model_name}  protocol=zero-shot  "
        f"variant={args.simcse_variant}  threshold=0.5"
    )
    final_lines.append(
        f"Trainable parameters: 0  "
        f"(no seeds needed: deterministic given checkpoint + dev set)"
    )
    final_lines.append(
        f"Wall-clock: {secs:.1f} s  ({pairs_per_sec:.1f} pairs/sec)"
    )
    final_lines.append("-" * 72)
    final_lines.append(
        f"Final Testing (QQP dev, n={metrics['n']}):"
    )
    final_lines.append(f"  BCE                        : {metrics['bce']:.6f}")
    final_lines.append(f"  AUC-ROC (threshold-free)   : {metrics['auc']:.6f}")
    final_lines.append(f"  -- Protocol (a) threshold = 0.5  ({'GLUE-clean':>12s})")
    final_lines.append(f"     Accuracy                : {metrics['accuracy']:.6f}")
    final_lines.append(f"     F1 (pos_label=1)        : {metrics['f1']:.6f}")
    final_lines.append("=" * 72)

    final_text = "\n".join(final_lines)
    print(final_text)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(final_text + "\n")


if __name__ == "__main__":
    main()
