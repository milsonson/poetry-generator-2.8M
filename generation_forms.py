from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
PUNCT_RE = re.compile(r"[，。？！；：]")
FORM_TOKENS = {
    "五言绝句": "㊀",
    "七言绝句": "㊁",
    "五言律诗": "㊂",
    "七言律诗": "㊃",
}
TOKEN_TO_FORM = {token: form for form, token in FORM_TOKENS.items()}
FORM_TEMPLATES = {
    "五言绝句": (5, 5, 5, 5),
    "七言绝句": (7, 7, 7, 7),
    "五言律诗": (5, 5, 5, 5, 5, 5, 5, 5),
    "七言律诗": (7, 7, 7, 7, 7, 7, 7, 7),
}
FORM_OPTIONS = list(FORM_TEMPLATES.keys())


@dataclass(frozen=True)
class FormSpec:
    name: str
    token: str
    sentence_lengths: tuple[int, ...]


def chinese_chars(text: str) -> list[str]:
    return CHINESE_RE.findall(text)


def sentence_lengths(text: str) -> tuple[int, ...]:
    parts = [part for part in PUNCT_RE.split(strip_form_tokens(text)) if part]
    return tuple(len(chinese_chars(part)) for part in parts if CHINESE_RE.search(part))


def detect_poem_form(text: str) -> Optional[str]:
    lengths = sentence_lengths(text)
    for form, template in FORM_TEMPLATES.items():
        if lengths == template:
            return form
    return None


def strip_form_tokens(text: str) -> str:
    return "".join(ch for ch in text if ch not in TOKEN_TO_FORM)


def build_labeled_line(text: str) -> Optional[str]:
    form = detect_poem_form(text)
    if not form:
        return None
    return f"{FORM_TOKENS[form]}{strip_form_tokens(text)}"


def get_form_spec(form: str) -> Optional[FormSpec]:
    if form in ("", "自由生成", "free", None):
        return None
    if form not in FORM_TEMPLATES:
        raise ValueError(f"unsupported form: {form}")
    return FormSpec(
        name=form,
        token=FORM_TOKENS[form],
        sentence_lengths=FORM_TEMPLATES[form],
    )


def format_text_by_form(text: str, form: str) -> str:
    spec = get_form_spec(form)
    if spec is None:
        return strip_form_tokens(text)

    chars = chinese_chars(strip_form_tokens(text))
    lines: list[str] = []
    cursor = 0
    for index, length in enumerate(spec.sentence_lengths):
        sentence = "".join(chars[cursor : cursor + length])
        cursor += length
        if not sentence:
            break
        punct = "，" if index % 2 == 0 else "。"
        lines.append(sentence + punct)

    couplets = []
    for i in range(0, len(lines), 2):
        couplets.append("".join(lines[i : i + 2]))
    return "\n".join(couplets)


def allowed_content_token_ids(itos: Iterable[str]) -> list[int]:
    return [
        idx
        for idx, token in enumerate(itos)
        if len(token) == 1 and CHINESE_RE.fullmatch(token) is not None
    ]
