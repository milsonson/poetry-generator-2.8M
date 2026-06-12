import sys

from generate import parse_args


def test_generation_defaults_use_conservative_sampling(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py"])

    args = parse_args()

    assert args.form == "七言绝句"
    assert args.temperature == 0.6
    assert args.top_k == 40
    assert args.repetition_penalty == 1.5
    assert args.adaptive_temperature is True


def test_generation_defaults_can_disable_adaptive_temperature(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--no-adaptive-temperature"])

    args = parse_args()

    assert args.adaptive_temperature is False
