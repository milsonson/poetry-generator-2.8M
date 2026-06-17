from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from generation_forms import FORM_OPTIONS
from generate import generate_text
from poem_scorer import select_best_candidate


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"

_MODEL_CACHE: Optional[Tuple[Any, Dict[str, int], list[str], Any]] = None
_MODEL_ERROR: Optional[str] = None


def resolve_device(torch_module: Any, requested: str) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def load_model_once(device_name: str = "auto") -> Tuple[Any, Dict[str, int], list[str], Any]:
    global _MODEL_CACHE, _MODEL_ERROR
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if _MODEL_ERROR:
        raise RuntimeError(_MODEL_ERROR)

    try:
        import torch

        from model import GPTConfig, PoetryTransformer

        vocab_path = ROOT / "vocab.json"
        checkpoint_path = ROOT / "transformer_poetry.pth"
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = GPTConfig(**checkpoint["config"])
        model = PoetryTransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        device = resolve_device(torch, device_name)
        model.to(device)
        model.eval()
        _MODEL_CACHE = (model, vocab["stoi"], vocab["itos"], device)
        return _MODEL_CACHE
    except Exception as exc:
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_MODEL_ERROR) from exc


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(parsed, high))


def parse_generation_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    start = str(payload.get("start", "春")).strip() or "春"
    form = str(payload.get("form", "七言绝句"))
    if form not in FORM_OPTIONS:
        form = "七言绝句"
    top_k_raw = payload.get("top_k")
    top_k = int(top_k_raw) if top_k_raw not in (None, "", 0) else 40
    candidate_count = clamp_int(payload.get("candidates", 50), 50, 1, 50)
    top_n = clamp_int(payload.get("top_n", 3), 3, 1, candidate_count)
    return {
        "start": start,
        "max_new_tokens": clamp_int(payload.get("max_new_tokens", 80), 80, 1, 240),
        "temperature": float(payload.get("temperature", 0.6)),
        "form": form,
        "top_k": top_k,
        "repetition_penalty": max(1.0, min(float(payload.get("repetition_penalty", 1.5)), 2.5)),
        "repetition_window": clamp_int(payload.get("repetition_window", 64), 64, 1, 240),
        "adaptive_temperature": bool(payload.get("adaptive_temperature", True)),
        "candidates": candidate_count,
        "top_n": top_n,
        "theme": str(payload.get("theme") or start),
        "poet": str(payload.get("poet") or ""),
    }


def score_payload(score: Any) -> Dict[str, Any]:
    return {
        "rank_score": score.rank_score,
        "parts": score.parts,
        "reasons": score.reasons,
        "warnings": score.warnings,
    }


def ranked_candidate_payload(rank: int, item: Any) -> Dict[str, Any]:
    return {
        "rank": rank,
        "text": item.text,
        "rank_score": item.score.rank_score,
        "parts": item.score.parts,
        "reasons": item.score.reasons,
        "warnings": item.score.warnings,
    }


def format_stream_event(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def read_samples() -> list[Dict[str, str]]:
    path = ROOT / "generation_samples.txt"
    if not path.exists():
        return []
    sections: list[Dict[str, str]] = []
    title = "sample"
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("===") and line.endswith("==="):
            if body:
                sections.append({"title": title, "text": "\n".join(body).strip()})
                body = []
            title = line.strip("= ").strip()
        else:
            body.append(line)
    if body:
        sections.append({"title": title, "text": "\n".join(body).strip()})
    return sections


def json_response(handler: SimpleHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class PoetryHandler(SimpleHTTPRequestHandler):
    server_version = "PoetryTransformerWeb/1.0"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.serve_file(STATIC_DIR / "index.html")
            return
        if self.path == "/api/status":
            self.handle_status()
            return
        if self.path == "/api/samples":
            json_response(self, 200, {"samples": read_samples()})
            return
        if self.path == "/loss_curve.png":
            self.serve_file(ROOT / "loss_curve.png")
            return
        if self.path.startswith("/static/"):
            self.serve_file(STATIC_DIR / self.path.removeprefix("/static/"))
            return
        json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            self.handle_generate(stream=False)
            return
        if self.path == "/api/generate_stream":
            self.handle_generate(stream=True)
            return
        json_response(self, 404, {"error": "not found"})

    def handle_generate(self, stream: bool) -> None:
        stream_started = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            settings = parse_generation_settings(raw_payload)
            started = time.perf_counter()

            model, stoi, itos, device = load_model_once()

            from generation_forms import get_form_spec

            spec = get_form_spec(settings["form"])
            prompt_chars = f"{spec.token}{settings['start']}"
            missing = sorted({ch for ch in prompt_chars if ch not in stoi})
            if missing:
                json_response(
                    self,
                    400,
                    {"error": "起始文字不在词表中", "missing": "".join(missing)},
                )
                return

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                stream_started = True
                self.send_stream({"event": "start", "total": settings["candidates"]})

            candidates: list[str] = []
            for index in range(settings["candidates"]):
                if stream:
                    self.send_stream(
                        {
                            "event": "progress",
                            "current": index + 1,
                            "total": settings["candidates"],
                        }
                    )
                candidates.append(
                    generate_text(
                        model,
                        stoi,
                        itos,
                        device,
                        start=settings["start"],
                        max_new_tokens=settings["max_new_tokens"],
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                        form=settings["form"],
                        repetition_penalty=settings["repetition_penalty"],
                        repetition_window=settings["repetition_window"],
                        adaptive_temperature=settings["adaptive_temperature"],
                    )
                )

            selected = select_best_candidate(
                candidates,
                form=settings["form"],
                theme=settings["theme"],
                poet=settings["poet"],
            )
            ranked = [
                ranked_candidate_payload(rank, item)
                for rank, item in enumerate(selected.ranked, start=1)
            ]
            payload = {
                "event": "final",
                "text": selected.text,
                "score": score_payload(selected.score),
                "candidates": ranked[: settings["top_n"]],
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "device": str(device),
                "settings": settings,
            }
            if stream:
                self.send_stream(payload)
            else:
                json_response(self, 200, payload)
        except Exception as exc:
            error_payload = {"event": "error", "error": f"{type(exc).__name__}: {exc}"}
            if stream and stream_started:
                self.send_stream(error_payload)
            else:
                json_response(self, 500, {"error": error_payload["error"]})

    def send_stream(self, payload: Dict[str, Any]) -> None:
        self.wfile.write(format_stream_event(payload))
        self.wfile.flush()

    def handle_status(self) -> None:
        try:
            model, stoi, _itos, device = load_model_once()
            config = model.config.to_dict()
            params = sum(p.numel() for p in model.parameters())
            json_response(
                self,
                200,
                {
                    "ready": True,
                    "device": str(device),
                    "vocab_size": len(stoi),
                    "params": params,
                    "config": config,
                    "samples": read_samples(),
                    "forms": FORM_OPTIONS,
                },
            )
        except Exception as exc:
            json_response(
                self,
                200,
                {
                    "ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples": read_samples(),
                    "forms": FORM_OPTIONS,
                },
            )

    def serve_file(self, path: Path) -> None:
        path = path.resolve()
        allowed_roots = [ROOT.resolve(), STATIC_DIR.resolve()]
        if not any(path == root or root in path.parents for root in allowed_roots):
            json_response(self, 403, {"error": "forbidden"})
            return
        if not path.exists() or not path.is_file():
            json_response(self, 404, {"error": "not found"})
            return
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the poetry generation web UI.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ReusableThreadingHTTPServer((args.host, args.port), PoetryHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Poetry UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
