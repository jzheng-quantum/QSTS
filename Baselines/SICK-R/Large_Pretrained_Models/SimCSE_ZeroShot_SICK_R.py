

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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel


SIMCSE_HIDDEN = 768
MAX_LEN = 128
BATCH_SIZE = 32
EPS = 1e-12

SIMCSE_VARIANTS = ("unsup", "sup")
DEFAULT_SIMCSE_VARIANT = "sup"


def simcse_model_dir(base_dir: str, variant: str) -> str:
    if variant not in SIMCSE_VARIANTS:
        raise ValueError(
            f"Unknown SimCSE variant {variant!r}; "
            f"expected one of {SIMCSE_VARIANTS}."
        )
    return str(Path(base_dir) / f"{variant}-simcse-bert-base-uncased")


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(file_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    data = pd.read_csv(data_path, sep="\t", keep_default_na=False)
    train_data = data[data["SemEval_set"] == "TRAIN"].reset_index(drop=True)
    val_data   = data[data["SemEval_set"] == "TRIAL"].reset_index(drop=True)
    test_data  = data[data["SemEval_set"] == "TEST"].reset_index(drop=True)
    return train_data, val_data, test_data


class SentencePairDataset(Dataset):
    

    def __init__(self, data: pd.DataFrame, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: List[Tuple[str, str, float]] = []
        for _, row in data.iterrows():
            sent_a = str(row["sentence_A"])
            sent_b = str(row["sentence_B"])

            label = (float(row["relatedness_score"]) - 1.0) / 4.0
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


@torch.no_grad()
def evaluate_split(model: BertModel,
                   loader: DataLoader,
                   device: torch.device,
                   simcse_variant: str) -> Dict[str, float]:
    
    if simcse_variant not in SIMCSE_VARIANTS:
        raise ValueError(
            f"Unknown simcse_variant {simcse_variant!r}; "
            f"expected one of {SIMCSE_VARIANTS}."
        )

    model.eval()
    preds, targs = [], []

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

        preds.extend(sim.detach().cpu().numpy().astype(np.float64).tolist())
        targs.extend(labels.numpy().astype(np.float64).tolist())

    preds = np.asarray(preds, dtype=np.float64)
    targs = np.asarray(targs, dtype=np.float64)

    return {
        "mse":      float(mean_squared_error(targs, preds)),
        "pearson":  safe_pearsonr(targs, preds),
        "spearman": safe_spearmanr(targs, preds),
        "n":        int(targs.size),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",  type=str,
                        default="./SICK.txt")
    parser.add_argument("--model_base_dir", type=str,
                        default="./",
                        help="Directory that contains the SimCSE variant dirs.")
    parser.add_argument("--simcse_variant", type=str,
                        default=DEFAULT_SIMCSE_VARIANT,
                        choices=list(SIMCSE_VARIANTS),
                        help=f"Which SimCSE checkpoint to load. "
                             f"Default: '{DEFAULT_SIMCSE_VARIANT}'.")
    parser.add_argument("--log_dir",    type=str, default="./logs")
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

    base_name = (f"{model_name}_SICK_zeroshot_{pooler_short_tag}_"
                 f"{run_stamp}")
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
        f"=== Zero-shot SimCSE evaluation on SICK-R ===\n"
        f"run_stamp={run_stamp}\n"
        f"model={model_name}\n"
        f"  - variant         : {args.simcse_variant} "
        f"(princeton-nlp/{args.simcse_variant}-simcse-bert-base-uncased)\n"
        f"  - model path      : {model_path}\n"
        f"  - tokenizer       : AutoTokenizer (BertTokenizerFast / WordPiece), "
        f"max_len={args.max_len}\n"
        f"  - pooling         : {pooling_label}\n"
        f"  - sentence vector : {pooling_desc}\n"
        f"  - similarity      : (cos(v1, v2) + 1) / 2   in [0, 1]\n"
        f"  - protocol        : zero-shot (no trainable parameters, "
        f"no gradient updates, no random seed)\n"
        f"  - trainable params: 0\n"
        f"  - two-tower       : sentences encoded independently, cosine compared\n"
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


    total_params = sum(p.numel() for p in model.parameters())
    pc_lines = [
        "Parameter breakdown:",
        f"  - SimCSE (frozen, non-trainable) : {total_params:,}",
        f"  - Trainable                      : 0",
    ]
    for line in pc_lines:
        print(line)
    with log_path.open("a", encoding="utf-8") as f:
        for line in pc_lines:
            f.write(line + "\n")


    train_data, val_data, test_data = load_data(args.data_file)
    splits: List[Tuple[str, pd.DataFrame]] = [
        ("TRAIN", train_data),
        ("TRIAL", val_data),
        ("TEST",  test_data),
    ]


    results: Dict[str, Dict[str, float]] = {}
    print()
    print("Evaluating splits...")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\nEvaluating splits...\n")

    for split_name, split_df in splits:
        dataset = SentencePairDataset(split_df, tokenizer, args.max_len)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
        metrics = evaluate_split(model, loader, device,
                                 simcse_variant=args.simcse_variant)
        results[split_name] = metrics

        line = (f"  [{split_name:>5}]  n={metrics['n']:>5}  "
                f"MSE={metrics['mse']:.6f}  "
                f"Pearson={metrics['pearson']:.6f}  "
                f"Spearman={metrics['spearman']:.6f}")
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


    final_lines: List[str] = []
    final_lines.append("")
    final_lines.append("=" * 72)
    final_lines.append(f"FINAL  |  model={model_name}  protocol=zero-shot")
    final_lines.append(f"Trainable parameters: 0  (no seeds needed: deterministic)")
    final_lines.append("-" * 72)
    final_lines.append(
        f"{'split':>10} | {'n':>6} | {'MSE':>10} | {'Pearson':>10} | {'Spearman':>10}"
    )
    for sp in ("TRAIN", "TRIAL", "TEST"):
        r = results[sp]
        final_lines.append(
            f"{sp:>10} | {r['n']:>6} | {r['mse']:>10.6f} | "
            f"{r['pearson']:>10.6f} | {r['spearman']:>10.6f}"
        )
    final_lines.append("-" * 72)
    test = results["TEST"]
    final_lines.append(
        f"Headline (TEST): "
        f"MSE = {test['mse']:.4f}, "
        f"Pearson = {test['pearson']:.4f}, "
        f"Spearman = {test['spearman']:.4f}"
    )
    final_lines.append("=" * 72)

    final_text = "\n".join(final_lines)
    print(final_text)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(final_text + "\n")


if __name__ == "__main__":
    main()
