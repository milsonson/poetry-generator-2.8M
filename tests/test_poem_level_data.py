import argparse

import torch

from prepare_data import EOS_TOKEN, PAD_TOKEN, prepare_poem_dataset
from train import IGNORE_INDEX, get_poem_batch


def test_prepare_poem_dataset_saves_one_sequence_per_poem(tmp_path):
    source = tmp_path / "poems.txt"
    source.write_text(
        "㊁春风吹过古江岸，夜雨初晴入远山。\n"
        "㊀山中一夜雨，树杪百重泉。\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        local_txt=str(source),
        sample_size=0,
        seed=42,
        val_ratio=0.4,
        output_dir=tmp_path,
        dataset_name="unused",
        split="train",
    )

    prepare_poem_dataset(args)

    vocab = torch.load(tmp_path / "train_sequences.pt", map_location="cpu")
    val = torch.load(tmp_path / "val_sequences.pt", map_location="cpu")
    meta = (tmp_path / "vocab.json").read_text(encoding="utf-8")

    assert len(vocab) == 1
    assert len(val) == 1
    assert PAD_TOKEN in meta
    assert EOS_TOKEN in meta


def test_get_poem_batch_keeps_poem_start_and_ignores_padding():
    sequences = [
        torch.tensor([5, 6, 7, 8], dtype=torch.long),
        torch.tensor([9, 10], dtype=torch.long),
    ]

    x, y = get_poem_batch(
        sequences,
        batch_size=2,
        block_size=8,
        pad_id=0,
        device=torch.device("cpu"),
        sample_indices=torch.tensor([0, 1]),
    )

    assert x.tolist()[0] == [5, 6, 7]
    assert y.tolist()[0] == [6, 7, 8]
    assert x.tolist()[1] == [9, 0, 0]
    assert y.tolist()[1] == [10, IGNORE_INDEX, IGNORE_INDEX]
