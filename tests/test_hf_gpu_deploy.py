from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_is_gpu_ready():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-web.txt").read_text(encoding="utf-8")

    assert "pytorch/pytorch:" in dockerfile
    assert "cuda" in dockerfile.lower()
    assert "download.pytorch.org/whl/cpu" not in requirements


def test_hf_deploy_helper_can_request_gpu_hardware():
    script = (ROOT / "scripts" / "deploy_hf_space.py").read_text(encoding="utf-8")

    assert "--hardware" in script
    assert "request_space_hardware" in script
    assert "t4-small" in script
