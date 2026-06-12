import sys
from pathlib import Path

from train import is_improved_loss, parse_args, serialize_training_args


def test_training_defaults_use_original_compact_model_size(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = parse_args()

    assert args.max_iters == 8000
    assert args.n_layer == 4
    assert args.n_head == 4
    assert args.d_model == 128
    assert args.d_ff == 512
    assert args.learning_rate == 6e-4
    assert args.eval_interval == 200
    assert args.early_stop_patience == 5


def test_is_improved_loss_respects_min_delta():
    assert is_improved_loss(4.9, best_loss=5.0, min_delta=0.01)
    assert not is_improved_loss(4.995, best_loss=5.0, min_delta=0.01)


def test_serialize_training_args_converts_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--data-dir", "/tmp/data"])

    serialized = serialize_training_args(parse_args())

    assert serialized["data_dir"] == "/tmp/data"
    assert isinstance(serialized["output_dir"], str)
    assert not any(isinstance(value, Path) for value in serialized.values())
