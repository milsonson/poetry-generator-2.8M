from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from generation_forms import FORM_TOKENS, build_labeled_line, detect_poem_form


SCRIPT_DIR = Path(__file__).resolve().parent


def label_poems(poems: list[str]) -> tuple[list[str], Counter[str]]:
    labeled: list[str] = []
    stats: Counter[str] = Counter()
    for poem in poems:
        form = detect_poem_form(poem)
        if not form:
            stats["其他"] += 1
            continue
        line = build_labeled_line(poem)
        if line is None:
            stats["其他"] += 1
            continue
        stats[form] += 1
        labeled.append(line)
    return labeled, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label cleaned poems with form tokens.")
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "poetry_cleaned.txt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "poetry_form_labeled.txt")
    parser.add_argument("--stats", type=Path, default=SCRIPT_DIR / "form_stats.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    poems = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    labeled, stats = label_poems(poems)
    args.output.write_text("\n".join(labeled) + "\n", encoding="utf-8")
    payload = {
        "input_count": len(poems),
        "labeled_count": len(labeled),
        "dropped_count": len(poems) - len(labeled),
        "form_counts": dict(stats),
        "form_tokens": FORM_TOKENS,
    }
    args.stats.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Input poems: {len(poems)}")
    print(f"Labeled poems: {len(labeled)}")
    print(f"Dropped poems: {len(poems) - len(labeled)}")
    for form, count in stats.most_common():
        print(f"{form}: {count}")
    print(f"Output: {args.output.resolve()}")
    print(f"Stats: {args.stats.resolve()}")


if __name__ == "__main__":
    main()
