import poem_scorer
from poem_scorer import rhyme_group, score_poem, select_best_candidate


def test_rhyme_group_merges_only_near_rhyme_variants():
    assert rhyme_group("山") == rhyme_group("间")
    assert rhyme_group("寒") == rhyme_group("看")
    assert rhyme_group("关") == rhyme_group("间")
    assert rhyme_group("间") != rhyme_group("还")
    assert rhyme_group("春") != rhyme_group("时")


def test_rhyme_is_a_small_bonus_not_a_dominant_score():
    rhymed = "春风吹柳色，明月照江山。\n白鸟归云外，清霜满故关。"
    unrhymed = "春风吹柳色，明月照江山。\n白鸟归云外，清霜满东楼。"

    rhymed_score = score_poem(rhymed, form="五言绝句", theme="月")
    unrhymed_score = score_poem(unrhymed, form="五言绝句", theme="月")

    assert rhymed_score.rank_score > unrhymed_score.rank_score
    assert 0 < rhymed_score.parts["rhyme"] <= 6


def test_same_sound_rhyme_is_weaker_than_same_rhyme_with_different_sounds():
    same_sound = "春风吹柳色，明月照清新。\n白鸟归云外，寒花入客心。"
    different_sound = "春风吹柳色，明月照江山。\n白鸟归云外，清霜满故关。"

    same_sound_score = score_poem(same_sound, form="五言绝句", theme="月")
    different_sound_score = score_poem(different_sound, form="五言绝句", theme="月")

    assert rhyme_group("新") == rhyme_group("心")
    assert 0 < same_sound_score.parts["rhyme"] < different_sound_score.parts["rhyme"]
    assert any("同声" in warning for warning in same_sound_score.warnings)


def test_mismatched_rhyme_does_not_beat_clean_rhyme():
    wrong_rhyme = "月白黄金一笑人，风尘闲认碧云间。\n山林日暮秋声处，只在梅枝花满还。"
    clean_rhyme = "月白黄金一笑人，风尘闲认碧云山。\n山林日暮秋声处，只在梅枝花满关。"

    wrong_score = score_poem(wrong_rhyme, form="七言绝句", theme="月")
    clean_score = score_poem(clean_rhyme, form="七言绝句", theme="月")

    assert clean_score.parts["rhyme"] > wrong_score.parts["rhyme"]
    assert clean_score.rank_score > wrong_score.rank_score


def test_pingze_cadence_uses_line_position_and_tail_tones_without_rejecting(monkeypatch):
    tones = {
        "甲": 3,
        "乙": 1,
        "丙": 4,
        "丁": 2,
        "戊": 1,
        "己": 4,
        "庚": 3,
        "辛": 4,
    }
    monkeypatch.setattr(poem_scorer, "pinyin_tone", lambda char: tones.get(char))
    clean = "春风吹柳甲，明月照江乙。\n白鸟归云丙，清霜入故丁。"
    heavy_oblique = "春风吹柳戊，明月照江己。\n白鸟归云庚，清霜入故辛。"

    clean_score = score_poem(clean, form="五言绝句", theme="月")
    oblique_score = score_poem(heavy_oblique, form="五言绝句", theme="月")

    assert oblique_score.rejected is False
    assert clean_score.parts["tone"] > oblique_score.parts["tone"]
    assert clean_score.rank_score > oblique_score.rank_score
    assert any("韵脚偏仄" in warning for warning in oblique_score.warnings)


def test_missing_tone_dependency_does_not_use_manual_tone_table(monkeypatch):
    monkeypatch.setattr(poem_scorer, "lazy_pinyin", None)
    monkeypatch.setattr(poem_scorer, "Style", None)

    assert poem_scorer.pinyin_tone("山") is None


def test_pingze_cadence_looks_at_tail_tones_not_only_final_char(monkeypatch):
    tones = {
        "甲": 1,
        "乙": 2,
        "丙": 1,
        "丁": 2,
        "戊": 4,
        "己": 2,
        "庚": 4,
        "辛": 4,
        "壬": 3,
        "癸": 1,
        "子": 1,
        "丑": 2,
        "寅": 1,
        "卯": 2,
        "辰": 4,
        "巳": 2,
        "午": 1,
        "未": 2,
        "申": 3,
        "酉": 1,
    }
    monkeypatch.setattr(poem_scorer, "pinyin_tone", lambda char: tones.get(char))
    varied_tail = "甲乙丙丁戊，己庚辛壬癸。\n子丑寅卯辰，巳午未申酉。"
    monotone_tail = "甲乙庚辛戊，己庚子丑癸。\n子丑庚辛辰，巳午子丑酉。"

    varied_score = score_poem(varied_tail, form="五言绝句")
    monotone_score = score_poem(monotone_tail, form="五言绝句")

    assert varied_score.parts["tone"] > monotone_score.parts["tone"]
    assert any("结尾三字平仄单调" in warning for warning in monotone_score.warnings)


def test_weird_char_and_strong_repetition_are_ranked_down():
    clean = "春风吹柳色，明月照江山。\n白鸟归云外，清霜满故关。"
    weird = "月鍜便将闲作隠，不如一片更难云。\n何言是说公山路，莫为诗夫即是贫。"
    repeated = "春风春风起，明月照江山。\n春风春风在，清霜满故关。"

    clean_score = score_poem(clean, form="五言绝句", theme="月")
    weird_score = score_poem(weird, form="七言绝句", theme="月")
    repeated_score = score_poem(repeated, form="五言绝句", theme="月")

    assert clean_score.rank_score > weird_score.rank_score
    assert clean_score.rank_score > repeated_score.rank_score
    assert weird_score.warnings
    assert repeated_score.warnings


def test_select_best_candidate_uses_weak_ranking_without_claiming_quality_score():
    candidates = [
        "月鍜便将闲作隠，不如一片更难云。\n何言是说公山路，莫为诗夫即是贫。",
        "春风吹柳色，明月照江山。\n白鸟归云外，清霜满故关。",
    ]

    selected = select_best_candidate(candidates, form="五言绝句", theme="月")

    assert selected.text == candidates[1]
    assert selected.score.rank_score == max(item.score.rank_score for item in selected.ranked)
    assert selected.score.total == selected.score.rank_score
