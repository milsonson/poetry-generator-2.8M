from generation_forms import (
    FORM_TOKENS,
    build_labeled_line,
    detect_poem_form,
    format_text_by_form,
    strip_form_tokens,
)


def test_detect_poem_form_recognizes_near_style_forms():
    assert detect_poem_form("春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。") == "五言绝句"
    assert detect_poem_form("轻波拍岸琉璃碧，落日衔山玳瑁红。一曲渔歌人不会，芦花飞起渡头空。") == "七言绝句"


def test_build_labeled_line_prefixes_single_character_form_token():
    poem = "轻波拍岸琉璃碧，落日衔山玳瑁红。一曲渔歌人不会，芦花飞起渡头空。"

    labeled = build_labeled_line(poem)

    assert labeled.startswith(FORM_TOKENS["七言绝句"])
    assert strip_form_tokens(labeled) == poem


def test_format_text_by_form_enforces_sentence_lengths_and_punctuation():
    text = "春山云气动夜雨满江寒故国人何在孤灯照客船"

    assert format_text_by_form(text, "五言绝句") == "春山云气动，夜雨满江寒。\n故国人何在，孤灯照客船。"
