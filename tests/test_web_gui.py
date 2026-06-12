from pathlib import Path

from generation_forms import FORM_OPTIONS
from web_app import parse_args


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


def test_public_form_options_only_include_structured_forms():
    assert FORM_OPTIONS == ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]


def test_web_app_uses_deployment_host_and_port_from_environment(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "7861")
    monkeypatch.setattr("sys.argv", ["web_app.py"])

    args = parse_args()

    assert args.host == "0.0.0.0"
    assert args.port == 7861
