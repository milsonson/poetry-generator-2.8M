from types import SimpleNamespace

import torch

from generate import append_due_form_punctuation, generate_structured


def test_append_due_form_punctuation_adds_expected_comma_after_first_sentence():
    itos = ["㊀", "春", "山", "云", "气", "动", "，", "。", "人"]
    idx = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)

    updated = append_due_form_punctuation(idx, itos, "五言绝句")

    assert updated.tolist()[0][-1] == itos.index("，")


def test_generate_structured_feeds_punctuation_back_into_context():
    class FakeModel:
        config = SimpleNamespace(block_size=32)

        def __init__(self):
            self.seen_inputs = []

        def generate(self, *args, **kwargs):
            raise AssertionError("structured forms should not use free generation")

        def __call__(self, idx):
            self.seen_inputs.append(idx.detach().cpu())
            logits = torch.full((1, idx.shape[1], 9), -1000.0, device=idx.device)
            logits[:, -1, 8] = 1000.0
            return logits, None

    itos = ["㊀", "春", "山", "云", "气", "动", "，", "。", "人"]
    idx = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
    model = FakeModel()

    generate_structured(model, idx, itos, form="五言绝句", max_new_tokens=1, temperature=0)

    assert model.seen_inputs[0].tolist()[0][-1] == itos.index("，")
