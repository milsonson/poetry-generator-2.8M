from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
BRACKET_TITLE_RE = re.compile(r"^《[^》]{1,40}》")
LEADING_AUTHOR_RE = re.compile(r"^[\u4e00-\u9fff]{2,6}[：:]")
NOISE_LINE_RE = re.compile(r"(作者|题解|注释|赏析|译文|年代|朝代|来源|http|www\.)", re.I)


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
        "[": "【",
        "]": "】",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[ \t\r\f\v]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chinese_char_count(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def clean_poem(raw: str) -> str:
    text = normalize_text(raw)
    text = BRACKET_TITLE_RE.sub("", text).strip()
    text = LEADING_AUTHOR_RE.sub("", text).strip()
    lines = []
    for line in text.splitlines() or [text]:
        line = line.strip()
        if not line or NOISE_LINE_RE.search(line):
            continue
        line = re.sub(r"[^\u4e00-\u9fff，。？！；：、《》（）【】]", "", line)
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"([，。？！；：])\1+", r"\1", text)
    return text.strip()


def clean_poems(
    rows: Iterable[str],
    min_chars: int = 8,
    max_chars: int = 256,
) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for row in rows:
        poem = clean_poem(row)
        count = chinese_char_count(poem)
        if count < min_chars or count > max_chars:
            continue
        if poem in seen:
            continue
        seen.add(poem)
        cleaned.append(poem)
    return cleaned


def read_poem_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    return [line.strip() for line in text.splitlines() if line.strip()]


def write_stats(path: Path, original_count: int, poems: list[str]) -> None:
    lengths = [chinese_char_count(poem) for poem in poems]
    payload = {
        "original_count": original_count,
        "cleaned_count": len(poems),
        "removed_count": original_count - len(poems),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a Chinese poetry text dataset.")
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "poetry.txt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "poetry_cleaned.txt")
    parser.add_argument("--stats", type=Path, default=SCRIPT_DIR / "poetry_cleaned_stats.json")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_poem_blocks(args.input)
    poems = clean_poems(rows, min_chars=args.min_chars, max_chars=args.max_chars)
    args.output.write_text("\n".join(poems) + "\n", encoding="utf-8")
    write_stats(args.stats, len(rows), poems)
    print(f"Original poems/lines: {len(rows)}")
    print(f"Cleaned poems: {len(poems)}")
    print(f"Output: {args.output.resolve()}")
    print(f"Stats: {args.stats.resolve()}")


if __name__ == "__main__":
    main()
