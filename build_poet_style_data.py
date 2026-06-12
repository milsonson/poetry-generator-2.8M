from __future__ import annotations

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "poet_style_data"
DEFAULT_RAW_ROOT = DEFAULT_DATA_DIR / "raw" / "chinese-poetry-npm" / "package" / "chinese-poetry"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "processed"

DEFAULT_POET_ALIASES = {
    "柳永": ["柳永"],
    "苏轼": ["苏轼", "蘇軾"],
    "李白": ["李白"],
    "杜甫": ["杜甫"],
    "李贺": ["李贺", "李賀"],
}

SOURCE_PATTERNS = [
    ("唐诗", "json/poet.tang.*.json"),
    ("宋诗", "json/poet.song.*.json"),
    ("宋词", "ci/ci.song.*.json"),
]

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
NOISE_RE = re.compile(r"(作者|注释|译文|赏析|题解|来源|http|www\.)", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build poet-conditioned style data from chinese-poetry JSON files."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--poets",
        nargs="*",
        default=list(DEFAULT_POET_ALIASES),
        help="Canonical poet names to export. Defaults to the project target poets.",
    )
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--script",
        choices=["simplified", "original"],
        default="simplified",
        help="Output script. Defaults to simplified Chinese for fine-tuning consistency.",
    )
    parser.add_argument(
        "--prefix-template",
        default="作者{name}",
        help="Prefix placed before every poem. Keep to current vocab characters for fine-tuning.",
    )
    parser.add_argument(
        "--prefix-separator",
        default="。",
        help="Separator between author prefix and poem text. The corpus keeps one record per line.",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Prepend poem titles/rhythmic names to training text. Metadata is always kept in JSONL.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="Optional vocab.json. If set, records containing out-of-vocab characters are dropped.",
    )
    return parser.parse_args()


def load_converter(script: str) -> Callable[[str], str]:
    if script == "original":
        return lambda text: text
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise RuntimeError(
            "Simplified output requires OpenCC. Run with the helper environment:\n"
            "poet_style_data/tools/opencc_venv/bin/python build_poet_style_data.py"
        ) from exc

    converter = OpenCC("t2s")
    def convert_until_stable(text: str) -> str:
        for _ in range(4):
            converted = converter.convert(text)
            if converted == text:
                return converted
            text = converted
        return text

    return convert_until_stable


def natural_key(path: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]


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
    text = re.sub(r"\n+", "", text)
    text = re.sub(r"([，。？！；：])\1+", r"\1", text)
    return text.strip()


def remove_parenthetical_notes(text: str) -> str:
    patterns = [
        r"（[^（）]{0,160}）",
        r"【[^【】]{0,160}】",
        r"\([^()]{0,160}\)",
        r"\[[^\[\]]{0,160}\]",
    ]
    changed = True
    while changed:
        before = text
        for pattern in patterns:
            text = re.sub(pattern, "", text)
        changed = text != before
    return text


def clean_poem(
    paragraphs: Any,
    title: str = "",
    include_title: bool = False,
    convert: Callable[[str], str] | None = None,
) -> str:
    if isinstance(paragraphs, list):
        text = "".join(str(item) for item in paragraphs)
    else:
        text = str(paragraphs or "")
    text = normalize_text(text)
    text = remove_parenthetical_notes(text)
    if convert is not None:
        text = convert(text)
    if not text or NOISE_RE.search(text):
        return ""

    # Keep only CJK characters and common poetry punctuation. This removes
    # OCR markers, editorial brackets, and malformed control glyphs.
    text = re.sub(r"[^\u4e00-\u9fff，。？！；：、]", "", text)
    text = re.sub(r"([，。？！；：])\1+", r"\1", text)
    text = text.strip("，。？！；：")

    if include_title and title:
        title = normalize_text(title)
        title = convert(title) if convert is not None else title
        title = re.sub(r"[^\u4e00-\u9fff，。？！；：、]", "", title)
        if title and not text.startswith(title) and CHINESE_RE.search(title):
            text = f"《{title}》{text}"
    return text


def chinese_char_count(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def load_poet_aliases(poets: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for poet in poets:
        names = DEFAULT_POET_ALIASES.get(poet, [poet])
        for name in names:
            aliases[name] = poet
    return aliases


def iter_records(
    raw_root: Path,
    alias_to_poet: dict[str, str],
    include_title: bool = False,
    convert: Callable[[str], str] | None = None,
) -> Iterable[dict[str, Any]]:
    for source, pattern in SOURCE_PATTERNS:
        files = sorted(glob.glob(str(raw_root / pattern)), key=natural_key)
        for file_path in files:
            data = json.loads(Path(file_path).read_text(encoding="utf-8"))
            for row in data:
                author = str(row.get("author") or "").strip()
                poet = alias_to_poet.get(author)
                if not poet:
                    continue
                poem = clean_poem(
                    row.get("paragraphs"),
                    title=str(row.get("title") or row.get("rhythmic") or "").strip(),
                    include_title=include_title,
                    convert=convert,
                )
                if not poem:
                    continue
                source_author_original = author
                title = str(row.get("title") or row.get("rhythmic") or "").strip()
                title = normalize_text(title) if title else ""
                title = convert(title) if convert is not None else title
                yield {
                    "poet": poet,
                    "source_author": convert(author) if convert is not None else author,
                    "source_author_original": source_author_original,
                    "source": source,
                    "title": title,
                    "text": poem,
                    "chars": chinese_char_count(poem),
                }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"raw data root not found: {raw_root}")
    if args.min_chars <= 0 or args.max_chars <= args.min_chars:
        raise ValueError("--max-chars must be greater than --min-chars")

    output_dir = args.output_dir.resolve()
    per_poet_dir = output_dir / "per_poet"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_poet_dir.mkdir(parents=True, exist_ok=True)

    alias_to_poet = load_poet_aliases(args.poets)
    convert = load_converter(args.script)
    allowed_chars: set[str] | None = None
    if args.vocab:
        vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
        allowed_chars = set(vocab["stoi"])

    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []
    dropped = Counter()

    for record in iter_records(
        raw_root,
        alias_to_poet,
        include_title=args.include_title,
        convert=convert,
    ):
        if record["chars"] < args.min_chars:
            dropped["too_short"] += 1
            continue
        if record["chars"] > args.max_chars:
            dropped["too_long"] += 1
            continue
        key = (record["poet"], record["text"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if allowed_chars is not None:
            prefixed = (
                f"{args.prefix_template.format(name=record['poet'])}"
                f"{args.prefix_separator}{record['text']}\n"
            )
            if any(ch not in allowed_chars for ch in prefixed):
                dropped["out_of_vocab"] += 1
                continue
        seen.add(key)
        records.append(record)

    random.Random(args.seed).shuffle(records)

    by_poet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    char_totals = Counter()
    for record in records:
        by_poet[record["poet"]].append(record)
        by_source[record["poet"]][record["source"]] += 1
        char_totals[record["poet"]] += int(record["chars"])

    jsonl_path = output_dir / "poet_style_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    corpus_lines: list[str] = []
    for record in records:
        prefix = args.prefix_template.format(name=record["poet"])
        corpus_lines.append(f"{prefix}{args.prefix_separator}{record['text']}")
    corpus_path = output_dir / "poet_style_corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    for poet, poet_records in sorted(by_poet.items()):
        lines = [
            f"{args.prefix_template.format(name=poet)}{args.prefix_separator}{record['text']}"
            for record in poet_records
        ]
        (per_poet_dir / f"{poet}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = {
        "poets": {
            poet: {
                "works": len(poet_records),
                "chars": char_totals[poet],
                "sources": dict(by_source[poet]),
            }
            for poet, poet_records in sorted(by_poet.items())
        },
        "total_works": len(records),
        "dropped": dict(dropped),
        "raw_root": str(raw_root),
        "prefix_template": args.prefix_template,
        "prefix_separator": args.prefix_separator,
        "include_title": args.include_title,
        "script": args.script,
        "vocab": str(args.vocab.resolve()) if args.vocab else None,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
    }
    save_json(output_dir / "poet_style_stats.json", stats)

    print(f"Records: {len(records)}")
    for poet, info in stats["poets"].items():
        print(f"{poet}: {info['works']} works, {info['chars']} chars, {info['sources']}")
    print(f"Dropped: {dict(dropped)}")
    print(f"Corpus: {corpus_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"Stats: {output_dir / 'poet_style_stats.json'}")


if __name__ == "__main__":
    main()
