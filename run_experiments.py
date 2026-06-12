from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from model import GPTConfig, PoetryTransformer
from train import (
    estimate_loss,
    get_batch,
    load_tensor,
    load_vocab,
    resolve_device,
    set_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled hyperparameter experiments.")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "experiments")
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def experiment_grid() -> List[Dict[str, object]]:
    return [
        {
            "name": "baseline",
            "changed_param": "baseline",
            "n_layer": 2,
            "n_head": 4,
            "d_model": 96,
            "d_ff": 384,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "name": "fewer_heads",
            "changed_param": "n_head=2",
            "n_layer": 2,
            "n_head": 2,
            "d_model": 96,
            "d_ff": 384,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "name": "wider_ffn",
            "changed_param": "d_ff=768",
            "n_layer": 2,
            "n_head": 4,
            "d_model": 96,
            "d_ff": 768,
            "dropout": 0.1,
            "learning_rate": 3e-4,
        },
        {
            "name": "higher_dropout",
            "changed_param": "dropout=0.2",
            "n_layer": 2,
            "n_head": 4,
            "d_model": 96,
            "d_ff": 384,
            "dropout": 0.2,
            "learning_rate": 3e-4,
        },
        {
            "name": "higher_lr",
            "changed_param": "lr=1e-3",
            "n_layer": 2,
            "n_head": 4,
            "d_model": 96,
            "d_ff": 384,
            "dropout": 0.1,
            "learning_rate": 1e-3,
        },
    ]


def train_one(
    exp: Dict[str, object],
    vocab_size: int,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=int(exp["n_layer"]),
        n_head=int(exp["n_head"]),
        d_model=int(exp["d_model"]),
        d_ff=int(exp["d_ff"]),
        dropout=float(exp["dropout"]),
    )
    model = PoetryTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(exp["learning_rate"]),
        weight_decay=0.01,
    )

    start = time.time()
    history = []
    for iter_num in range(args.max_iters):
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, device)
            history.append({"iter": iter_num, **losses})
            print(
                f"{exp['name']} iter {iter_num}: "
                f"train={losses['train']:.4f}, val={losses['val']:.4f}"
            )

        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_losses = estimate_loss(model, train_data, val_data, args, device)
    elapsed = time.time() - start
    params = sum(p.numel() for p in model.parameters())

    result = {
        "name": exp["name"],
        "changed_param": exp["changed_param"],
        "n_layer": exp["n_layer"],
        "n_head": exp["n_head"],
        "d_model": exp["d_model"],
        "d_ff": exp["d_ff"],
        "dropout": exp["dropout"],
        "learning_rate": exp["learning_rate"],
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "params_m": round(params / 1e6, 3),
        "final_train_loss": round(final_losses["train"], 4),
        "final_val_loss": round(final_losses["val"], 4),
        "seconds": round(elapsed, 1),
        "loss_history": history + [{"iter": args.max_iters, **final_losses}],
    }
    return result


def write_csv(results: List[Dict[str, object]], path: Path) -> None:
    fields = [
        "name",
        "changed_param",
        "n_layer",
        "n_head",
        "d_model",
        "d_ff",
        "dropout",
        "learning_rate",
        "max_iters",
        "batch_size",
        "block_size",
        "params_m",
        "final_train_loss",
        "final_val_loss",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row[field] for field in fields})


def write_markdown(results: List[Dict[str, object]], path: Path) -> None:
    best = min(results, key=lambda row: float(row["final_val_loss"]))
    lines = [
        "# Hyperparameter Experiment Results",
        "",
        "| experiment | changed param | params(M) | train loss | val loss | conclusion |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        conclusion = "best validation loss" if row["name"] == best["name"] else ""
        lines.append(
            "| {name} | {changed_param} | {params_m} | {final_train_loss} | "
            "{final_val_loss} | {conclusion} |".format(**row, conclusion=conclusion)
        )

    lines.extend(
        [
            "",
            "## Brief Analysis",
            "",
            (
                f"In this run, `{best['name']}` reached the lowest validation loss "
                f"({best['final_val_loss']}). Because all experiments keep the same "
                "data split, batch size, block size, seed, and training iterations, "
                "the comparison isolates the listed hyperparameter change."
            ),
            "",
            "Report wording in Chinese:",
            "",
            (
                f"在本轮控制变量实验中，验证集 loss 最低的是 `{best['name']}` "
                f"({best['final_val_loss']})。实验保持数据划分、batch size、block size、"
                "随机种子和训练迭代数一致，只改变表中的单个超参数，因此可以比较该超参数"
                "对收敛速度和验证集表现的影响。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(args.data_dir / "vocab.json")
    train_data = load_tensor(args.data_dir / "train_data.pt")
    val_data = load_tensor(args.data_dir / "val_data.pt")
    vocab_size = int(vocab["vocab_size"])

    print(f"Device: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data)}, val tokens: {len(val_data)}")

    results = []
    for exp in experiment_grid():
        print(f"\nRunning experiment: {exp['name']} ({exp['changed_param']})")
        set_seed(args.seed)
        results.append(train_one(exp, vocab_size, train_data, val_data, args, device))

    json_path = args.output_dir / "hyperparam_results.json"
    csv_path = args.output_dir / "hyperparam_results.csv"
    md_path = args.output_dir / "hyperparam_results.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, csv_path)
    write_markdown(results, md_path)

    print(f"\nSaved: {json_path.resolve()}")
    print(f"Saved: {csv_path.resolve()}")
    print(f"Saved: {md_path.resolve()}")


if __name__ == "__main__":
    main()
