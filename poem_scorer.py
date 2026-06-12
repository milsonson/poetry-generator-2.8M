from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from generation_forms import FORM_TEMPLATES, PUNCT_RE, chinese_chars, sentence_lengths

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover - only used when optional dependency is absent.
    Style = None
    lazy_pinyin = None


SCRIPT_DIR = Path(__file__).resolve().parent

RHYME_GROUPS = {
    "an": "安岸暗按案山删关还环寰湾弯闲间艰颜寒韩看残餐兰栏阑干竿肝端酸欢宽官冠观鸾峦滩难丹单",
    "ang": "昂肮邦帮苍藏长场常昌窗床霜光黄王望忘亡方芳房妨防唐堂塘梁凉良量阳杨央香乡湘章张掌上尚商伤裳庄装妆江郎廊荒",
    "en": "恩门们闻文分纷坟汾尘臣辰晨人仁身神真珍邻贫频民巾津秦亲银因",
    "in": "云君群春新心林临深沉金今音阴寻",
    "eng": "生声城成明名清情晴轻倾平评萍兵惊京经行横更耕僧层曾登灯藤能仍",
    "ong": "风空中东同铜桐红洪鸿公宫穷融",
    "ai": "来开台苔才哀埃怀淮乖斋佳街阶皆偕鞋排",
    "ei": "微飞归辉晖挥违围肥非霏",
    "i": "西溪低啼齐迷泥题蹄稀衣依机枝时知诗池迟离移疑宜奇期旗",
    "ao": "高豪毫涛陶桃逃遥摇桥娇霄宵萧条朝潮招昭烧",
    "ou": "秋流留楼舟州愁游幽优忧休收头投侯喉谋浮",
    "u": "无吴吾湖孤姑苏书疏虚居鱼余渔初除如庐炉途图都",
}

CHAR_TO_RHYME: dict[str, str] = {}
for group, chars in RHYME_GROUPS.items():
    for char in chars:
        CHAR_TO_RHYME.setdefault(char, group)

RHYME_GROUP_OVERRIDES = {
    # "还" is a common polyphonic character. Treat it conservatively in the
    # weak scorer so it does not make an otherwise mismatched rhyme look clean.
    "还": "ai",
}

MANUAL_SYLLABLES = {
    "新": "xin",
    "心": "xin",
    "山": "shan",
    "关": "guan",
    "间": "jian",
    "还": "hai",
}

FINAL_GROUPS = {
    "an": {"an", "ian", "uan"},
    "ang": {"ang", "iang", "uang"},
    "ai": {"ai", "uai"},
    "ei": {"ei", "uei", "ui"},
    "ao": {"ao", "iao"},
    "ou": {"ou", "iou", "iu"},
    "en": {"en"},
    "in": {"in"},
    "un": {"un", "uen"},
    "eng": {"eng", "ing"},
    "ong": {"ong", "iong"},
    "i": {"i"},
    "u": {"u"},
    "v": {"v", "ve", "ue"},
}

FINAL_TO_GROUP = {
    final: group for group, finals in FINAL_GROUPS.items() for final in finals
}

THEME_WORDS = {
    "月": set("月夜霜露云天清影桂寒秋"),
    "春": set("春花风柳莺草雨桃杏芳"),
    "山": set("山云石松泉寺峰林岩"),
    "水": set("水江河湖溪波舟渔浪"),
    "酒": set("酒杯醉酌樽客醒吟"),
    "风": set("风云雨柳尘帆雁"),
}

STYLE_WORDS = {
    "李白": set("月酒天云长安青君仙剑山水舟"),
    "杜甫": set("江客故园白发兵乱老村国风尘"),
    "李贺": set("金玉冷鬼血宫秋露马龙寒"),
}

BAD_PATTERNS = [
    "三百度",
    "六十分",
    "三十二五",
    "七十一",
    "情情",
    "照照",
    "树树",
    "诗家诗句",
    "客里客人",
    "何处何妨",
    "何言是说",
    "即是贫",
    "公山路",
]

OBJECT_CHARS = set("山水月云花风雪江舟酒人客天树草鸟松柳烟雨霜露日星河湖溪石林峰寺门城楼")
ACTION_CHARS = set("来去归入出落照吹满看闻听问寻开过起飞行坐卧吟笑语")
FUNCTION_CHARS = set("不无有何谁莫更未已自相为与是如可应将但却又")
NUMBER_CHARS = set("一二三四五六七八九十百千万")
SAFE_REDUP_CHARS = set("萧飕依纷茫泠潇悠悠")


def load_char_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for name in ["poetry_form_labeled.txt", "poetry_cleaned.txt", "poetry.txt"]:
        path = SCRIPT_DIR / name
        if path.exists():
            counts.update(chinese_chars(path.read_text(encoding="utf-8")))
            if counts:
                break
    return counts


CHAR_COUNTS = load_char_counts()


@dataclass(frozen=True)
class PoemScore:
    rank_score: float
    parts: dict[str, float]
    reasons: list[str]
    warnings: list[str]
    rejected: bool = False

    @property
    def total(self) -> float:
        return self.rank_score


@dataclass(frozen=True)
class RankedCandidate:
    text: str
    score: PoemScore


@dataclass(frozen=True)
class SelectionResult:
    text: str
    score: PoemScore
    ranked: list[RankedCandidate]


def poem_lines(text: str) -> list[str]:
    normalized = text.replace("\n", "")
    return [part for part in PUNCT_RE.split(normalized) if chinese_chars(part)]


def pinyin_syllable(char: str) -> Optional[str]:
    if lazy_pinyin is None:
        return MANUAL_SYLLABLES.get(char)
    syllables = lazy_pinyin(char, errors="ignore")
    if not syllables:
        return MANUAL_SYLLABLES.get(char)
    return syllables[0]


def pinyin_final_group(char: str) -> Optional[str]:
    if lazy_pinyin is None or Style is None:
        return None
    finals = lazy_pinyin(char, style=Style.FINALS, strict=False, errors="ignore")
    if not finals:
        return None
    final = finals[0]
    return FINAL_TO_GROUP.get(final, final or None)


def pinyin_tone(char: str) -> Optional[int]:
    if lazy_pinyin is not None and Style is not None and hasattr(Style, "TONE3"):
        syllables = lazy_pinyin(
            char,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        if syllables:
            syllable = syllables[0]
            if syllable and syllable[-1].isdigit():
                return int(syllable[-1])
    return None


def tone_class(char: str) -> Optional[str]:
    tone = pinyin_tone(char)
    if tone in (1, 2):
        return "ping"
    if tone in (3, 4):
        return "ze"
    return None


def rhyme_group(char: str) -> Optional[str]:
    if char in RHYME_GROUP_OVERRIDES:
        return RHYME_GROUP_OVERRIDES[char]
    return pinyin_final_group(char) or CHAR_TO_RHYME.get(char)


def format_check(text: str, form: str) -> tuple[bool, list[str]]:
    if form not in FORM_TEMPLATES:
        return True, []
    actual = sentence_lengths(text)
    expected = FORM_TEMPLATES[form]
    if actual == expected:
        return True, ["格式通过"]
    return False, [f"格式不符: expected {expected}, got {actual}"]


def rhyme_bonus(text: str, form: str) -> tuple[float, list[str], list[str]]:
    if form not in FORM_TEMPLATES:
        return 0.0, [], ["自由生成不检查押韵"]
    lines = poem_lines(text)
    rhyme_indices = [1, 3] if len(FORM_TEMPLATES[form]) == 4 else [1, 3, 5, 7]
    endings = [lines[i][-1] for i in rhyme_indices if i < len(lines) and lines[i]]
    groups = [rhyme_group(char) for char in endings]
    syllables = [pinyin_syllable(char) for char in endings]
    known = [group for group in groups if group]
    if len(endings) < len(rhyme_indices):
        return -2.0, [], ["韵脚不足"]
    if len(known) < 2:
        return -1.0, [], [f"韵脚无法可靠识别: {''.join(endings)}"]

    counts = Counter(known)
    best_count = counts.most_common(1)[0][1]
    if best_count == len(groups):
        warnings = [f"押韵为普通话近似: {''.join(endings)}"]
        if len(set(endings)) < len(endings):
            warnings.append(f"重复韵脚字: {''.join(endings)}")
            return 1.0, ["同韵但重复"], warnings

        known_syllables = [syllable for syllable in syllables if syllable]
        if len(known_syllables) == len(syllables) and len(set(known_syllables)) < len(syllables):
            warnings.append(f"同声韵脚: {''.join(endings)}")
            return 2.0, ["同韵但同声"], warnings

        return 6.0, ["近似押韵"], warnings
    if best_count >= max(2, len(groups) - 1):
        return 2.0, ["部分近似押韵"], [f"韵脚不完全一致: {''.join(endings)}"]
    return -2.0, [], [f"韵脚分散: {''.join(endings)}"]


def rhyme_line_indices(form: str) -> list[int]:
    return [1, 3] if len(FORM_TEMPLATES[form]) == 4 else [1, 3, 5, 7]


def expected_line_final_tone(form: str, line_index: int) -> Optional[str]:
    if form not in FORM_TEMPLATES:
        return None
    return "ping" if line_index in rhyme_line_indices(form) else "ze"


def tail_tone_score(line: str, form: str, line_index: int) -> tuple[float, list[str], list[str]]:
    chars = chinese_chars(line)
    if not chars:
        return 0.0, [], []

    expected_final = expected_line_final_tone(form, line_index)
    classes = [tone_class(char) for char in chars[-3:]]
    known = [cls for cls in classes if cls is not None]
    final_char = chars[-1]
    final_class = classes[-1] if classes else None
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    if expected_final and final_class:
        if final_class == expected_final:
            score += 1.2 if expected_final == "ping" else 0.5
        else:
            if expected_final == "ping":
                score -= 2.4
                warnings.append(f"韵脚偏仄: {final_char}")
            else:
                score -= 0.5
                warnings.append(f"非韵句平收: {final_char}")

    if len(classes) >= 2 and classes[-2] and final_class:
        expected_penultimate = "ze" if expected_final == "ping" else "ping"
        if classes[-2] == expected_penultimate:
            score += 0.4
        elif classes[-2] == final_class:
            score -= 0.5
            if len(warnings) < 6:
                warnings.append(f"末二字平仄偏黏: {line}")

    if len(known) == 3:
        if len(set(known)) == 1:
            score -= 0.9
            if len(warnings) < 6:
                warnings.append(f"结尾三字平仄单调: {line}")
        elif known[-1] != known[-2]:
            score += 0.2

    if score > 0:
        reasons.append("尾字平仄较顺")
    return score, reasons, warnings


def tone_cadence_bonus(text: str, form: str) -> tuple[float, list[str], list[str]]:
    if form not in FORM_TEMPLATES:
        return 0.0, [], []
    lines = poem_lines(text)
    endings: list[tuple[int, str, Optional[str]]] = []
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    for index, line in enumerate(lines):
        chars = chinese_chars(line)
        if not chars:
            continue
        char = chars[-1]
        endings.append((index, char, tone_class(char)))
        line_score, line_reasons, line_warnings = tail_tone_score(line, form, index)
        score += line_score
        reasons.extend(line_reasons)
        warnings.extend(line_warnings)

    ending_classes = {index: cls for index, _char, cls in endings if cls is not None}
    for first in range(0, len(lines), 2):
        second = first + 1
        if first not in ending_classes or second not in ending_classes:
            continue
        pair = (ending_classes[first], ending_classes[second])
        if pair == ("ze", "ping"):
            score += 0.7
        else:
            score -= 0.8
            if len(warnings) < 6:
                warnings.append(f"联句收束平仄弱: {first + 1}-{second + 1}")

    if score > 0:
        reasons.append("平仄收束较顺")
    return round(max(-8.0, min(4.0, score)), 3), list(dict.fromkeys(reasons)), warnings[:6]


def repetition_penalty(text: str) -> tuple[float, list[str]]:
    chars = chinese_chars(text)
    penalty = 0.0
    warnings: list[str] = []
    for a, b in zip(chars, chars[1:]):
        if a == b and a not in SAFE_REDUP_CHARS:
            penalty += 5.0
            warnings.append(f"连续重复字: {a}{b}")
    for n, weight in [(2, 2.0), (3, 5.0)]:
        grams = ["".join(chars[i : i + n]) for i in range(max(0, len(chars) - n + 1))]
        for gram, count in Counter(grams).items():
            if count > 1:
                penalty += weight * (count - 1)
                if len(warnings) < 8:
                    warnings.append(f"重复{n}字片段: {gram}")
    for pattern in BAD_PATTERNS:
        if pattern in text:
            penalty += 6.0
            warnings.append(f"弱句模式: {pattern}")
    return -min(18.0, penalty), warnings


def rare_char_penalty(text: str) -> tuple[float, list[str]]:
    if not CHAR_COUNTS:
        return 0.0, []
    penalty = 0.0
    warnings: list[str] = []
    for char in chinese_chars(text):
        count = CHAR_COUNTS.get(char, 0)
        if count <= 1:
            penalty += 6.0
            warnings.append(f"极低频字: {char}")
        elif count <= 3:
            penalty += 2.0
            warnings.append(f"低频字: {char}")
    return -min(18.0, penalty), warnings[:8]


def weak_syntax_penalty(text: str) -> tuple[float, list[str]]:
    penalty = 0.0
    warnings: list[str] = []
    for line in poem_lines(text):
        chars = chinese_chars(line)
        if not chars:
            continue
        function_count = sum(1 for char in chars if char in FUNCTION_CHARS)
        number_count = sum(1 for char in chars if char in NUMBER_CHARS)
        object_count = sum(1 for char in chars if char in OBJECT_CHARS)
        action_count = sum(1 for char in chars if char in ACTION_CHARS)
        if function_count >= 4:
            penalty += 3.0
            warnings.append(f"虚字偏密: {line}")
        if number_count >= 3:
            penalty += 4.0
            warnings.append(f"数字偏密: {line}")
        if object_count == 0 and action_count == 0:
            penalty += 3.0
            warnings.append(f"句法支点弱: {line}")
    return -min(12.0, penalty), warnings[:6]


def theme_bonus(text: str, theme: str = "") -> tuple[float, list[str]]:
    if not theme:
        return 0.0, []
    words = THEME_WORDS.get(theme, {theme})
    lines = poem_lines(text)
    hit_lines = sum(1 for line in lines if set(chinese_chars(line)) & words)
    chars = set(chinese_chars(text))
    bonus = 0.0
    reasons: list[str] = []
    if theme in chars:
        bonus += 2.0
        reasons.append("主题字出现")
    if hit_lines >= 2:
        bonus += 2.0
        reasons.append("主题意象分布")
    return min(4.0, bonus), reasons


def style_bonus(text: str, poet: str = "") -> tuple[float, list[str]]:
    if not poet or poet not in STYLE_WORDS:
        return 0.0, []
    hits = set(chinese_chars(text)) & STYLE_WORDS[poet]
    if not hits:
        return 0.0, []
    return min(3.0, len(hits) * 0.8), [f"{poet}词场轻微命中"]


def score_poem(text: str, form: str, theme: str = "", poet: str = "") -> PoemScore:
    ok, format_messages = format_check(text, form)
    if not ok:
        return PoemScore(
            rank_score=-999.0,
            parts={"format": -999.0},
            reasons=[],
            warnings=format_messages,
            rejected=True,
        )

    r_bonus, r_reasons, r_warnings = rhyme_bonus(text, form)
    rep_penalty, rep_warnings = repetition_penalty(text)
    rare_penalty, rare_warnings = rare_char_penalty(text)
    syntax_penalty, syntax_warnings = weak_syntax_penalty(text)
    tone_bonus, tone_reasons, tone_warnings = tone_cadence_bonus(text, form)
    t_bonus, t_reasons = theme_bonus(text, theme)
    s_bonus, s_reasons = style_bonus(text, poet)

    parts = {
        "rhyme": r_bonus,
        "theme": t_bonus,
        "style": s_bonus,
        "repetition": rep_penalty,
        "rare_chars": rare_penalty,
        "weak_syntax": syntax_penalty,
        "tone": tone_bonus,
    }
    rank_score = round(sum(parts.values()), 3)
    reasons = [*format_messages, *r_reasons, *tone_reasons, *t_reasons, *s_reasons]
    warnings = [*r_warnings, *tone_warnings, *rep_warnings, *rare_warnings, *syntax_warnings]
    return PoemScore(rank_score=rank_score, parts=parts, reasons=reasons, warnings=warnings)


def select_best_candidate(
    candidates: Iterable[str],
    form: str,
    theme: str = "",
    poet: str = "",
) -> SelectionResult:
    ranked = [
        RankedCandidate(text=text, score=score_poem(text, form=form, theme=theme, poet=poet))
        for text in candidates
    ]
    ranked.sort(key=lambda item: (item.score.rejected, -item.score.rank_score))
    if not ranked:
        raise ValueError("no candidates to select from")
    return SelectionResult(text=ranked[0].text, score=ranked[0].score, ranked=ranked)
