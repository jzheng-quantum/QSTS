

from typing import Dict, List, Tuple
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer


SEED = 42
REDUCED_DIM = 64
BERT_HIDDEN = 768
MAX_LEN = 128
NUM_EPOCHS = 80
BATCH_SIZE = 16
PATIENCE = 25
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


def load_data(train_file: str, val_file: str, test_file: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)
    return train_data, val_data, test_data


class SentencePairDataset(Dataset):
    

    def __init__(self, records: List[Dict], tokenizer: BertTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: List[Tuple[str, str, float]] = []


        for record in records:
            sent_a = str(record["sentence1"])
            sent_b = str(record["sentence2"])
            label = float(record["score"]) / 5.0
            self.records.append((sent_a, sent_b, label))

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


class BertFrozenEncoder(nn.Module):
    

    def __init__(self, bert_model_path: str, reduced_dim: int,
                 pooling: str = DEFAULT_POOLING):
        super().__init__()
        if pooling not in POOLING_CHOICES:
            raise ValueError(
                f"Unknown pooling {pooling!r}; "
                f"expected one of {POOLING_CHOICES}."
            )
        self.pooling = pooling

        self.bert = BertModel.from_pretrained(bert_model_path)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()


        self.linear = nn.Linear(BERT_HIDDEN, reduced_dim).double()

    def train(self, mode: bool = True):

        super().train(mode)
        self.bert.eval()
        return self

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
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
            last_hidden = bert_out.last_hidden_state
            if self.pooling == "cls":
                sent_vec_f32 = last_hidden[:, 0, :]
            elif self.pooling == "mean":
                sent_vec_f32 = self._mean_pool(last_hidden, attention_mask)
            else:

                raise RuntimeError(f"Unknown pooling: {self.pooling}")


        sent_vec = sent_vec_f32.detach().to(dtype=torch.float64)
        out = self.linear(sent_vec)
        return out


    @staticmethod
    def _numel(params) -> int:
        return sum(p.numel() for p in params)

    def count_parameters(self) -> Dict[str, int]:
        bert_total = self._numel(self.bert.parameters())
        bert_trainable = self._numel(p for p in self.bert.parameters()
                                     if p.requires_grad)
        linear_trainable = self._numel(p for p in self.linear.parameters()
                                       if p.requires_grad)
        return {
            "bert_total":       bert_total,
            "bert_trainable":   bert_trainable,
            "linear_trainable": linear_trainable,
            "total_trainable":  bert_trainable + linear_trainable,
        }


class SentenceSimilarityBERT(nn.Module):
    

    def __init__(self, bert_model_path: str, reduced_dim: int,
                 pooling: str = DEFAULT_POOLING):
        super().__init__()
        self.encoder = BertFrozenEncoder(bert_model_path, reduced_dim,
                                         pooling=pooling)

    def forward(self,
                input_ids_a: torch.Tensor, attention_mask_a: torch.Tensor,
                input_ids_b: torch.Tensor, attention_mask_b: torch.Tensor) -> torch.Tensor:
        v1 = self.encoder(input_ids_a, attention_mask_a)
        v2 = self.encoder(input_ids_b, attention_mask_b)
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
                optimizer: optim.Optimizer,
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
            total_loss += float(loss.item())
            n_batches += 1

        val_mse, val_pearson, val_spearman = evaluate_model(model, val_loader, device)
        scheduler.step(val_pearson)
        train_loss = total_loss / max(n_batches, 1)

        current_lrs = ", ".join(f"{group['lr']:.6g}"
                                for group in optimizer.param_groups)
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


            best_model_state = {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.requires_grad
            }
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


    missing, unexpected = model.load_state_dict(best_model_state, strict=False)


    if unexpected:
        raise RuntimeError(f"Unexpected keys when reloading best state: {unexpected}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}, "
                f"Best Val Pearson: {best_val_pearson:.6f}\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()


    parser.add_argument("--train_file", type=str,
                        default="./train.jsonl")
    parser.add_argument("--val_file",   type=str,
                        default="./validation.jsonl")
    parser.add_argument("--test_file",  type=str,
                        default="./test.jsonl")
    parser.add_argument("--model_path", type=str,
                        default="./bert-base-uncased")
    parser.add_argument("--log_dir",    type=str, default="./logs")
    parser.add_argument("--seeds",      type=int, nargs="+",
                        default=[0, 1, 2, 3, 42],
                        help="One or more seeds.  Multiple seeds produce a "
                             "mean +- std (ddof=1) summary across seeds.")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience",   type=int, default=PATIENCE)
    parser.add_argument("--max_len",    type=int, default=MAX_LEN)
    parser.add_argument("--grad_clip",  type=float, default=1.0,
                        help="Max gradient L2 norm for clipping. "
                             "Set to 0 or negative to disable.")
    parser.add_argument("--pooling",    type=str, default=DEFAULT_POOLING,
                        choices=list(POOLING_CHOICES),
                        help="Sentence-pooling strategy on top of frozen BERT. "
                             "'cls' = last_hidden_state[:, 0, :] ([CLS] token). "
                             "'mean' = mean over non-padding tokens with "
                             "attention_mask (Sentence-BERT-style).  Default: "
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
                    train_data: List[Dict],
                    val_data: List[Dict],
                    test_data: List[Dict],
                    tokenizer: BertTokenizer,
                    args: argparse.Namespace,
                    device: torch.device,
                    log_dir: Path,
                    run_stamp: str,
                    variant_tag: str,
                    model_name: str,
                    dataset_name: str,
                    optimizer_name: str) -> Dict[str, float]:

    set_seed(seed)

    base_name = (f"{model_name}_{dataset_name}_{optimizer_name}_"
                 f"{variant_tag}_seed{seed}_{run_stamp}")
    log_path        = log_dir / f"{base_name}.log"
    checkpoint_path = log_dir / f"{base_name}_best.pt"


    if args.pooling == "cls":
        pooling_desc = "last_hidden_state[:, 0, :]  (i.e. [CLS] token)"
    elif args.pooling == "mean":
        pooling_desc = ("mean over non-padding tokens of last_hidden_state "
                        "(Sentence-BERT-style, masked by attention_mask)")
    else:
        pooling_desc = f"<unknown pooling: {args.pooling}>"

    header = (
        f"=== Run started at {run_stamp} (seed={seed}) ===\n"
        f"model={model_name}\n"
        f"variant={variant_tag}\n"
        f"  - tokenizer       : BertTokenizer (WordPiece), max_len={args.max_len}\n"
        f"  - encoder         : BERT-base-uncased, frozen (requires_grad=False, eval mode)\n"
        f"  - pooling         : {args.pooling}\n"
        f"  - sentence vector : {pooling_desc}\n"
        f"  - head            : Linear({BERT_HIDDEN}->{REDUCED_DIM}), trainable, float64\n"
        f"  - similarity      : (cos(v1, v2) + 1) / 2   in [0, 1]\n"
        f"  - two-tower       : sentence A and B are encoded independently "
        f"(no BERT cross-attention)\n"
        f"seed={seed}, batch_size={args.batch_size}, num_epochs={args.num_epochs}, "
        f"patience={args.patience}, grad_clip={args.grad_clip}, device={device}"
    )
    print(header)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")


    train_dataset = SentencePairDataset(train_data, tokenizer, args.max_len)
    val_dataset   = SentencePairDataset(val_data,   tokenizer, args.max_len)
    test_dataset  = SentencePairDataset(test_data,  tokenizer, args.max_len)

    print(f"Number of training samples:   {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of testing samples:    {len(test_dataset)}")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              generator=loader_generator)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)


    model = SentenceSimilarityBERT(
        bert_model_path=args.model_path,
        reduced_dim=REDUCED_DIM,
        pooling=args.pooling,
    ).to(device)
    criterion = nn.MSELoss()

    pc = model.encoder.count_parameters()
    pc_lines = [
        "Parameter breakdown:",
        f"  - BERT (frozen, non-trainable) : {pc['bert_total']:,}",
        f"  - BERT trainable               : {pc['bert_trainable']:,}  "
        f"(expected 0 in frozen mode)",
        f"  - Linear({BERT_HIDDEN}->{REDUCED_DIM}) : {pc['linear_trainable']:,}",
        f"  - Total trainable              : {pc['total_trainable']:,}",
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
    model_name     = "BERT"
    dataset_name   = "STSB"
    optimizer_name = "adam_plateau"
    variant_tag    = f"frozen_{args.pooling}pool"
    run_stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)


    train_data, val_data, test_data = load_data(
        args.train_file, args.val_file, args.test_file
    )
    print(
        f"Split sizes — TRAIN: {len(train_data)}, "
        f"VALIDATION: {len(val_data)}, TEST: {len(test_data)}"
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_path)


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
            tokenizer=tokenizer,
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
