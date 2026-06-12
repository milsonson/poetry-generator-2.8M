from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


DEFAULT_DATASET = "Lifan-Z/Chinese-poetries-txt"
SCRIPT_DIR = Path(__file__).resolve().parent
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare character-level poetry data.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-txt", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.strip().replace("\ufeff", "")
    replacements = {
        ",": "，",
        ".": "。",
        "?": "？",
        "!": "！",
        ";": "；",
        ":": "：",
        "(": "（",
        ")": "）",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text)


def flatten_text_value(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [flatten_text_value(item) for item in value]
        parts = [part for part in parts if part]
        if parts:
            return "".join(parts)
    return None


def extract_text(row: Any) -> str:
    if isinstance(row, dict):
        preferred = ["text", "content", "poem", "poetry", "sentence"]
        for key in preferred:
            if key in row:
                value = flatten_text_value(row[key])
                if value:
                    return value

        candidates = []
        for value in row.values():
            text = flatten_text_value(value)
            if text:
                candidates.append(text)
        if not candidates:
            return ""
        return max(candidates, key=lambda s: (len(CHINESE_RE.findall(s)), len(s)))

    value = flatten_text_value(row)
    return value or ""


def read_local_poems(path: Path, sample_size: int, seed: int) -> List[str]:
    poems = [
        normalize_text(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if normalize_text(line)
    ]
    poems = [poem for poem in poems if CHINESE_RE.search(poem)]
    random.Random(seed).shuffle(poems)
    if sample_size > 0:
        poems = poems[:sample_size]
    return poems


def load_hf_poems(args: argparse.Namespace) -> List[str]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        loaded = load_dataset(args.dataset_name, split=args.split)
    except Exception:
        loaded = load_dataset(args.dataset_name)

    if isinstance(loaded, DatasetDict):
        if args.split not in loaded:
            available = ", ".join(loaded.keys())
            raise ValueError(f"split '{args.split}' not found; available splits: {available}")
        dataset = loaded[args.split]
    else:
        dataset = loaded

    if args.sample_size > 0 and len(dataset) > args.sample_size:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.sample_size))

    poems = []
    for row in dataset:
        poem = normalize_text(extract_text(row))
        if poem and CHINESE_RE.search(poem):
            poems.append(poem)
    return poems


def encode_text(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    ids = [stoi[ch] for ch in text]
    return torch.tensor(ids, dtype=torch.long)


def build_vocab(poems: List[str]) -> Dict[str, object]:
    chars = sorted(set("".join(poems)))
    itos = SPECIAL_TOKENS + [ch for ch in chars if ch not in SPECIAL_TOKENS]
    stoi = {ch: i for i, ch in enumerate(itos)}
    return {"stoi": stoi, "itos": itos, "vocab_size": len(itos)}


def encode_poem(poem: str, stoi: Dict[str, int]) -> torch.Tensor:
    ids = [stoi[ch] for ch in poem]
    ids.append(stoi[EOS_TOKEN])
    return torch.tensor(ids, dtype=torch.long)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_poem_dataset(args: argparse.Namespace) -> None:
    if not 0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio should be in (0, 0.5)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.local_txt:
        poems = read_local_poems(Path(args.local_txt), args.sample_size, args.seed)
    else:
        poems = load_hf_poems(args)

    if not poems:
        raise RuntimeError("No valid poems were loaded. Check the dataset or --local-txt path.")

    vocab = build_vocab(poems)
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    sequences = [encode_poem(poem, stoi) for poem in poems]

    split_idx = int(len(sequences) * (1.0 - args.val_ratio))
    if len(sequences) > 1:
        split_idx = min(max(split_idx, 1), len(sequences) - 1)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:] or sequences[-1:]
    train_data = torch.cat(train_sequences) if train_sequences else torch.empty(0, dtype=torch.long)
    val_data = torch.cat(val_sequences) if val_sequences else torch.empty(0, dtype=torch.long)

    (args.output_dir / "poetry.txt").write_text(
        "\n".join(poems) + "\n",
        encoding="utf-8",
    )
    save_json(
        args.output_dir / "vocab.json",
        {
            "stoi": stoi,
            "itos": itos,
            "vocab_size": len(itos),
            "sample_size": len(poems),
            "pad_token": PAD_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_id": stoi[PAD_TOKEN],
            "eos_id": stoi[EOS_TOKEN],
            "data_format": "poem_sequences",
        },
    )
    torch.save(train_data, args.output_dir / "train_data.pt")
    torch.save(val_data, args.output_dir / "val_data.pt")
    torch.save(train_sequences, args.output_dir / "train_sequences.pt")
    torch.save(val_sequences, args.output_dir / "val_sequences.pt")

    print(f"Loaded poems: {len(poems)}")
    print(f"Total characters: {sum(len(poem) for poem in poems)}")
    print(f"Vocab size: {len(itos)}")
    print(f"Train poems: {len(train_sequences)}")
    print(f"Val poems: {len(val_sequences)}")
    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens: {len(val_data)}")
    print(f"Output dir: {args.output_dir.resolve()}")


def main() -> None:
    prepare_poem_dataset(parse_args())


if __name__ == "__main__":
    main()
