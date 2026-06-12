import torch

from model import adaptive_temperature_from_logits, apply_repetition_penalty


def test_apply_repetition_penalty_lowers_recent_repeated_token_logits():
    logits = torch.tensor([[0.0, 4.0, 3.0, 2.0]])
    generated = torch.tensor([[1, 2, 1]])

    adjusted = apply_repetition_penalty(
        logits.clone(),
        generated,
        penalty=1.5,
        window=8,
        exempt_token_ids=None,
    )

    assert adjusted[0, 1] < logits[0, 1]
    assert adjusted[0, 2] < logits[0, 2]
    assert adjusted[0, 3] == logits[0, 3]
    assert adjusted[0, 1] < adjusted[0, 2]


def test_adaptive_temperature_reduces_high_entropy_and_raises_low_entropy():
    flat_logits = torch.zeros(1, 8)
    sharp_logits = torch.tensor([[10.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]])

    assert adaptive_temperature_from_logits(flat_logits, base_temperature=0.9) < 0.9
    assert adaptive_temperature_from_logits(sharp_logits, base_temperature=0.9) > 0.9
