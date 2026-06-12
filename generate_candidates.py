from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from generate import generate_text, load_model, load_vocab, resolve_device
from poem_scorer import select_best_candidate


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple poem candidates and rerank them.")
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "transformer_poetry.pth")
    parser.add_argument("--vocab", type=Path, default=SCRIPT_DIR / "vocab.json")
    parser.add_argument("--start", type=str, default="春")
    parser.add_argument("--form", default="七言绝句")
    parser.add_argument("--poet", default="")
    parser.add_argument("--theme", default="")
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Print full ranking as JSON.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stoi, itos = load_vocab(args.vocab)
    model = load_model(args.checkpoint, device)
    candidate_count = max(1, args.candidates)
    theme = args.theme or args.start

    candidates: list[str] = []
    for index in range(candidate_count):
        seed = args.seed + index
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        candidates.append(
            generate_text(
                model,
                stoi,
                itos,
                device,
                start=args.start,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                form=args.form,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                adaptive_temperature=True,
            )
        )

    selected = select_best_candidate(candidates, form=args.form, theme=theme, poet=args.poet)
    if args.json:
        payload = {
            "selected": selected.text,
            "score": {
                "rank_score": selected.score.rank_score,
                "parts": selected.score.parts,
                "reasons": selected.score.reasons,
                "warnings": selected.score.warnings,
            },
            "ranked": [
                {
                    "rank": rank,
                    "text": item.text,
                    "rank_score": item.score.rank_score,
                    "parts": item.score.parts,
                    "reasons": item.score.reasons,
                    "warnings": item.score.warnings,
                }
                for rank, item in enumerate(selected.ranked, start=1)
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(selected.text)
    print()
    print(f"rank_score: {selected.score.rank_score}")
    if selected.score.reasons:
        print("reasons:", "；".join(selected.score.reasons[:5]))
    if selected.score.warnings:
        print("warnings:", "；".join(selected.score.warnings[:5]))


if __name__ == "__main__":
    main()
