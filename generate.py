from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from generation_forms import (
    FORM_OPTIONS,
    allowed_content_token_ids,
    chinese_chars,
    format_text_by_form,
    get_form_spec,
    strip_form_tokens,
)
from model import (
    GPTConfig,
    PoetryTransformer,
    adaptive_temperature_from_logits,
    apply_repetition_penalty,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate poems with a trained Transformer.")
    parser.add_argument("--checkpoint", type=Path, default=SCRIPT_DIR / "transformer_poetry.pth")
    parser.add_argument("--vocab", type=Path, default=SCRIPT_DIR / "vocab.json")
    parser.add_argument("--start", type=str, default="春")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--form", default="七言绝句", choices=FORM_OPTIONS)
    parser.add_argument("--repetition-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--adaptive-temperature", dest="adaptive_temperature", action="store_true", default=True)
    parser.add_argument("--no-adaptive-temperature", dest="adaptive_temperature", action="store_false")
    parser.add_argument("--target-entropy", type=float, default=0.55)
    parser.add_argument("--temperature-strength", type=float, default=0.65)
    parser.add_argument("--min-temperature", type=float, default=0.55)
    parser.add_argument("--max-temperature", type=float, default=1.35)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_vocab(path: Path) -> Tuple[Dict[str, int], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return vocab["stoi"], vocab["itos"]


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    missing = sorted({ch for ch in text if ch not in stoi})
    if missing:
        missing_str = "".join(missing)
        raise ValueError(f"characters not found in vocab: {missing_str}")
    return [stoi[ch] for ch in text]


def decode(ids: List[int], itos: List[str]) -> str:
    return "".join(itos[i] for i in ids)


def load_model(checkpoint_path: Path, device: torch.device) -> PoetryTransformer:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found. Run train.py first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = GPTConfig(**checkpoint["config"])
    model = PoetryTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def append_due_form_punctuation(idx: torch.Tensor, itos: List[str], form: str) -> torch.Tensor:
    spec = get_form_spec(form)
    if spec is None:
        return idx

    text = decode(idx[0].tolist(), itos)
    chars_seen = len(chinese_chars(text))
    boundaries: List[int] = []
    total = 0
    for length in spec.sentence_lengths:
        total += length
        boundaries.append(total)

    if chars_seen not in boundaries:
        return idx

    last_token = itos[int(idx[0, -1].item())]
    if last_token in "，。":
        return idx

    sentence_index = boundaries.index(chars_seen)
    punct = "，" if sentence_index % 2 == 0 else "。"
    try:
        punct_id = itos.index(punct)
    except ValueError:
        return idx
    punct_tensor = torch.tensor([[punct_id]], dtype=torch.long, device=idx.device)
    return torch.cat((idx, punct_tensor), dim=1)


@torch.no_grad()
def generate_structured(
    model: PoetryTransformer,
    idx: torch.Tensor,
    itos: List[str],
    form: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.5,
    repetition_window: int = 64,
    adaptive_temperature: bool = False,
    target_entropy: float = 0.55,
    temperature_strength: float = 0.65,
    min_temperature: float = 0.55,
    max_temperature: float = 1.35,
) -> torch.Tensor:
    spec = get_form_spec(form)
    if spec is None:
        return model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            adaptive_temperature=adaptive_temperature,
            target_entropy=target_entropy,
            temperature_strength=temperature_strength,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
        )

    allowed_ids = allowed_content_token_ids(itos)
    if not allowed_ids:
        raise ValueError("vocab has no Chinese content tokens for structured generation")

    visible = strip_form_tokens(decode(idx[0].tolist(), itos))
    target_chars = sum(spec.sentence_lengths)
    remaining = max(0, target_chars - len(chinese_chars(visible)))
    steps = min(remaining, max_new_tokens)
    allowed = torch.tensor(allowed_ids, dtype=torch.long, device=idx.device)

    for _ in range(steps):
        idx = append_due_form_punctuation(idx, itos, form)
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        logits = apply_repetition_penalty(
            logits,
            idx,
            penalty=repetition_penalty,
            window=repetition_window,
        )

        masked = torch.full_like(logits, float("-inf"))
        masked.index_copy_(1, allowed, logits.index_select(1, allowed))
        logits = masked

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            current_temperature = temperature
            if adaptive_temperature:
                current_temperature = adaptive_temperature_from_logits(
                    logits,
                    base_temperature=temperature,
                    target_entropy=target_entropy,
                    strength=temperature_strength,
                    min_temperature=min_temperature,
                    max_temperature=max_temperature,
                )
            logits = logits / current_temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    idx = append_due_form_punctuation(idx, itos, form)
    return idx


def build_prompt(start: str, form: str) -> str:
    spec = get_form_spec(form)
    if spec is None:
        return start
    return f"{spec.token}{start}"


def render_generated_text(ids: List[int], itos: List[str], form: str) -> str:
    raw = decode(ids, itos)
    if get_form_spec(form) is None:
        return raw
    return format_text_by_form(raw, form)


def generate_text(
    model: PoetryTransformer,
    stoi: Dict[str, int],
    itos: List[str],
    device: torch.device,
    start: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    form: str = "七言绝句",
    repetition_penalty: float = 1.5,
    repetition_window: int = 64,
    adaptive_temperature: bool = False,
    target_entropy: float = 0.55,
    temperature_strength: float = 0.65,
    min_temperature: float = 0.55,
    max_temperature: float = 1.35,
) -> str:
    prompt = build_prompt(start, form)
    start_ids = encode(prompt, stoi)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    out = generate_structured(
        model,
        idx,
        itos,
        form=form,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        adaptive_temperature=adaptive_temperature,
        target_entropy=target_entropy,
        temperature_strength=temperature_strength,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
    )
    return render_generated_text(out[0].tolist(), itos, form)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    stoi, itos = load_vocab(args.vocab)
    model = load_model(args.checkpoint, device)

    text = generate_text(
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
        adaptive_temperature=args.adaptive_temperature,
        target_entropy=args.target_entropy,
        temperature_strength=args.temperature_strength,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
