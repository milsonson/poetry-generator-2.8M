from clean_poetry_data import clean_poem, clean_poems


def test_clean_poem_removes_author_title_noise_and_normalizes_punctuation():
    raw = "《静夜思》 李白：床前明月光, 疑是地上霜。"

    assert clean_poem(raw) == "床前明月光，疑是地上霜。"


def test_clean_poems_filters_short_non_chinese_and_duplicates():
    rows = [
        "春眠不觉晓，处处闻啼鸟。",
        "abc123",
        "春眠不觉晓，处处闻啼鸟。",
        "太短",
        "夜来风雨声，花落知多少。",
    ]

    assert clean_poems(rows, min_chars=8, max_chars=40) == [
        "春眠不觉晓，处处闻啼鸟。",
        "夜来风雨声，花落知多少。",
    ]
