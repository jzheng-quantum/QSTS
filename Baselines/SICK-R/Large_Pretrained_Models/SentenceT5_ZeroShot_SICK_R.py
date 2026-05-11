

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
from transformers import AutoTokenizer, T5EncoderModel


SENTENCE_T5_HIDDEN = 768
MAX_LEN = 128
BATCH_SIZE = 32
EPS = 1e-12


SENTENCE_T5_DIRNAME = "sentence-t5-base"


POOLING_CHOICES = ("mean", "cls")
DEFAULT_POOLING = "mean"


def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    masked_sum  = (last_hidden_state * mask).sum(dim=1)
    valid_count = mask.sum(dim=1).clamp(min=1.0)
    return masked_sum / valid_count


def sentence_t5_model_dir(base_dir: str) -> str:
    
    return str(Path(base_dir) / SENTENCE_T5_DIRNAME)


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
def evaluate_split(model: T5EncoderModel,
                   loader: DataLoader,
                   device: torch.device,
                   pooling: str = DEFAULT_POOLING) -> Dict[str, float]:
    
    if pooling not in POOLING_CHOICES:
        raise ValueError(
            f"Unknown pooling {pooling!r}; expected one of {POOLING_CHOICES}."
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
        last_a = out_a.last_hidden_state
        last_b = out_b.last_hidden_state

        if pooling == "cls":
            v_a = last_a[:, 0, :]
            v_b = last_b[:, 0, :]
        elif pooling == "mean":
            v_a = mean_pool(last_a, attention_mask_a)
            v_b = mean_pool(last_b, attention_mask_b)
        else:

            raise RuntimeError(f"Unknown pooling: {pooling}")

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
                        help="Directory that contains the sentence-t5-base "
                             "model directory. The resolved model path is "
                             f"<model_base_dir>/{SENTENCE_T5_DIRNAME}.")
    parser.add_argument("--pooling",    type=str,
                        default=DEFAULT_POOLING,
                        choices=list(POOLING_CHOICES),
                        help="Sentence-pooling strategy on top of the frozen "
                             "T5 encoder. 'mean' is the official Sentence-T5 "
                             "protocol (attention-mask-weighted mean over "
                             "tokens); 'cls' is provided for completeness "
                             "but is not semantically meaningful for T5 "
                             f"(no [CLS] token). Default: '{DEFAULT_POOLING}'.")
    parser.add_argument("--log_dir",    type=str, default="./logs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_len",    type=int, default=MAX_LEN)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cpu")

    run_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "SentenceT5-base"
    log_dir    = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_name = (f"{model_name}_SICK_zeroshot_{args.pooling}pool_"
                 f"{run_stamp}")
    log_path = log_dir / f"{base_name}.log"


    model_path = sentence_t5_model_dir(args.model_base_dir)
    if not Path(model_path).is_dir():
        raise FileNotFoundError(
            f"Sentence-T5 model directory not found: {model_path}.  "
            f"Expected layout: '{args.model_base_dir}/{SENTENCE_T5_DIRNAME}'."
        )


    if args.pooling == "mean":
        pooling_desc = ("attention-mask-weighted mean over token positions "
                        "(Sentence-T5's official protocol, Ni et al. 2022)")
    elif args.pooling == "cls":
        pooling_desc = ("last_hidden_state[:, 0, :]  "
                        "(first token; NOT semantically meaningful for T5, "
                        "provided for completeness only)")
    else:
        pooling_desc = f"<unknown pooling: {args.pooling}>"

    header = (
        f"=== Zero-shot Sentence-T5 evaluation on SICK-R ===\n"
        f"run_stamp={run_stamp}\n"
        f"model={model_name}\n"
        f"  - checkpoint      : sentence-transformers/sentence-t5-base "
        f"(Ni et al., ACL Findings 2022)\n"
        f"  - model path      : {model_path}\n"
        f"  - architecture    : T5EncoderModel (T5 encoder, hidden_size={SENTENCE_T5_HIDDEN})\n"
        f"  - tokenizer       : AutoTokenizer (T5TokenizerFast / SentencePiece), "
        f"max_len={args.max_len}\n"
        f"  - pooling         : {args.pooling}\n"
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
    model = T5EncoderModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


    total_params = sum(p.numel() for p in model.parameters())
    pc_lines = [
        "Parameter breakdown:",
        f"  - Sentence-T5 encoder (frozen, non-trainable) : {total_params:,}",
        f"  - Trainable                                   : 0",
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
        metrics = evaluate_split(model, loader, device, pooling=args.pooling)
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
    final_lines.append(f"FINAL  |  model={model_name}  protocol=zero-shot  "
                       f"pooling={args.pooling}")
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
