from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IGNORE = [
    ".git/*",
    ".github/*",
    ".pytest_cache/*",
    ".venv/*",
    "__pycache__/*",
    "backups/*",
    "data_100k_form/*",
    "data_100k_raw/*",
    "model_100k_form/*",
    "poet_style_data/raw/*",
    "poet_style_data/processed/*",
    "poet_style_data/processed_vocab_compatible/*",
    "poet_style_data/tools/*",
    "poet_style_data/training_full/*",
    "poet_style_models/*/transformer_poetry.pth",
    "poetry.txt",
    "poetry_cleaned.txt",
    "poetry_form_labeled.txt",
    "train_data.pt",
    "train_sequences.pt",
    "val_data.pt",
    "val_sequences.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update the Hugging Face Docker Space.")
    parser.add_argument("--repo-id", default="milsonson/poetry-generator-2.8M")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=ROOT,
        ignore_patterns=DEFAULT_IGNORE,
        commit_message="Deploy real poetry generator",
    )
    print(f"Space updated: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
