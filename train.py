from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model import GPTConfig, PoetryTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal Transformer on poems.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--max-iters", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def is_improved_loss(loss: float, best_loss: float, min_delta: float = 0.0) -> bool:
    return loss < best_loss - min_delta


def clone_state_dict_to_cpu(model: PoetryTransformer) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def serialize_training_args(args: argparse.Namespace) -> Dict[str, object]:
    serialized: Dict[str, object] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return torch.load(path, map_location="cpu")


def load_sequences(path: Path) -> List[torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return torch.load(path, map_location="cpu")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("data is too short for the chosen block_size")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_poem_batch(
    sequences: List[torch.Tensor],
    batch_size: int,
    block_size: int,
    pad_id: int,
    device: torch.device,
    sample_indices: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("no poem sequences were loaded")
    if sample_indices is None:
        sample_indices = torch.randint(len(sequences), (batch_size,))

    selected = []
    max_len = 1
    for index in sample_indices.tolist():
        seq = sequences[index]
        if len(seq) < 2:
            continue
        window = seq[: block_size + 1]
        selected.append(window)
        max_len = max(max_len, len(window) - 1)

    if not selected:
        raise ValueError("all selected poem sequences are too short")

    x = torch.full((len(selected), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(selected), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, seq in enumerate(selected):
        input_ids = seq[:-1]
        target_ids = seq[1:]
        x[row, : len(input_ids)] = input_ids
        y[row, : len(target_ids)] = target_ids
    return x.to(device), y.to(device)


def get_training_batch(
    data: torch.Tensor | List[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(data, list):
        return get_poem_batch(data, args.batch_size, args.block_size, pad_id, device)
    return get_batch(data, args.batch_size, args.block_size, device)


@torch.no_grad()
def estimate_loss(
    model: PoetryTransformer,
    train_data: torch.Tensor | List[torch.Tensor],
    val_data: torch.Tensor | List[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    pad_id: int,
) -> Dict[str, float]:
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_training_batch(data, args, device, pad_id)
            _, loss = model(x, y)
            split_losses[k] = loss.detach().cpu()
        losses[split] = float(split_losses.mean().item())
    model.train()
    return losses


def save_loss_curve(history: List[Dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [item["iter"] for item in history]
    train_losses = [item["train"] for item in history]
    val_losses = [item["val"] for item in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("cross entropy loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.data_dir / "vocab.json")
    pad_id = int(vocab.get("pad_id", 0))
    train_sequences_path = args.data_dir / "train_sequences.pt"
    val_sequences_path = args.data_dir / "val_sequences.pt"
    if train_sequences_path.exists() and val_sequences_path.exists():
        train_data = load_sequences(train_sequences_path)
        val_data = load_sequences(val_sequences_path)
        data_format = "poem_sequences"
    else:
        train_data = load_tensor(args.data_dir / "train_data.pt")
        val_data = load_tensor(args.data_dir / "val_data.pt")
        data_format = "sliding_tokens"

    config = GPTConfig(
        vocab_size=int(vocab["vocab_size"]),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = PoetryTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Parameters: {total_params / 1e6:.2f}M")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Data format: {data_format}")

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_iter = -1
    best_state_dict: Dict[str, torch.Tensor] | None = None
    no_improve_evals = 0
    pbar = tqdm(range(args.max_iters), desc="training")
    for iter_num in pbar:
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, device, pad_id)
            history.append({"iter": iter_num, **losses})
            pbar.set_postfix(train=f"{losses['train']:.3f}", val=f"{losses['val']:.3f}")
            if is_improved_loss(losses["val"], best_val_loss, args.min_delta):
                best_val_loss = losses["val"]
                best_iter = iter_num
                best_state_dict = clone_state_dict_to_cpu(model)
                no_improve_evals = 0
            else:
                no_improve_evals += 1
                if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                    print(
                        f"Early stopping at iter {iter_num}; "
                        f"best val loss {best_val_loss:.4f} at iter {best_iter}"
                    )
                    break

        x, y = get_training_batch(train_data, args, device, pad_id)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_losses = estimate_loss(model, train_data, val_data, args, device, pad_id)
    history.append({"iter": best_iter if best_iter >= 0 else args.max_iters, "selected": "best", **final_losses})

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "vocab_size": config.vocab_size,
        "loss_history": history,
        "best_iter": best_iter,
        "best_val_loss": best_val_loss,
        "selected_checkpoint": "best_val",
        "training_args": serialize_training_args(args),
        "data_format": data_format,
        "pad_id": pad_id,
        "ignore_index": IGNORE_INDEX,
    }
    ckpt_path = args.output_dir / "transformer_poetry.pth"
    torch.save(checkpoint, ckpt_path)

    curve_path = args.output_dir / "loss_curve.png"
    save_loss_curve(history, curve_path)

    print(f"Selected iter: {best_iter}")
    print(f"Selected train loss: {final_losses['train']:.4f}")
    print(f"Selected val loss: {final_losses['val']:.4f}")
    print(f"Saved checkpoint: {ckpt_path.resolve()}")
    print(f"Saved loss curve: {curve_path.resolve()}")


if __name__ == "__main__":
    main()
