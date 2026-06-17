from pathlib import Path

from generation_forms import FORM_OPTIONS
from web_app import format_stream_event, parse_args, parse_generation_settings


HTML = Path(__file__).resolve().parents[1] / "static" / "index.html"


def test_gui_keeps_generation_controls_without_samples_or_curve():
    html = HTML.read_text(encoding="utf-8")

    assert "自由生成" not in html
    assert 'id="topk"' in html
    assert 'id="repetitionPenalty"' in html
    assert 'value="1.5"' in html
    assert 'id="repetitionWindow"' in html
    assert 'id="candidates"' in html
    assert 'id="adaptiveTemperature"' in html
    assert 'id="adaptiveTemperature" name="adaptiveTemperature" type="checkbox" checked' in html
    assert "generation_samples.txt" not in html
    assert "loss_curve.png" not in html


def test_gui_is_poetry_only_and_exposes_progress_and_top_n():
    html = HTML.read_text(encoding="utf-8")

    assert "宋词" not in html
    assert "词牌" not in html
    assert 'id="genreCi"' not in html
    assert 'id="cipai"' not in html
    assert 'id="topN"' in html
    assert 'id="candidates" name="candidates" type="number" min="1" max="50" value="50"' in html
    assert 'id="progressText"' in html
    assert 'id="progressFill"' in html
    assert 'id="rankedList"' in html
    assert "/api/generate_stream" in html
    assert "正在生成第" in html


def test_generation_settings_clamp_top_n_to_candidate_count():
    settings = parse_generation_settings(
        {
            "start": "月",
            "form": "七言律诗",
            "candidates": 8,
            "top_n": 20,
            "temperature": 0.7,
        }
    )

    assert settings["start"] == "月"
    assert settings["form"] == "七言律诗"
    assert settings["candidates"] == 8
    assert settings["top_n"] == 8
    assert settings["temperature"] == 0.7


def test_stream_event_is_newline_delimited_json():
    event = format_stream_event({"event": "progress", "current": 2, "total": 5})

    assert event.endswith(b"\n")
    assert b'"event": "progress"' in event
    assert b'"current": 2' in event


def test_public_form_options_only_include_structured_forms():
    assert FORM_OPTIONS == ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]


def test_web_app_uses_deployment_host_and_port_from_environment(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "7861")
    monkeypatch.setattr("sys.argv", ["web_app.py"])

    args = parse_args()

    assert args.host == "0.0.0.0"
    assert args.port == 7861
