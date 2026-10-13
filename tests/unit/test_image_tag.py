"""The generated overlay that pins the image of the last build."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "set_image_tag.py"


@pytest.fixture
def set_image_tag(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("set_image_tag", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "kustomization.yaml").write_text("resources: []\n", encoding="utf-8")
    monkeypatch.setattr(module, "OVERLAYS", tmp_path)

    def run(tag: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["set_image_tag.py", "fake", tag])
        assert module.main() == 0
        return yaml.safe_load((tmp_path / "_build" / "kustomization.yaml").read_text(encoding="utf-8"))
    return run


# A short commit SHA is seven hexadecimal characters: some are numbers to a YAML parser
# (474e135 is a float, 1234567 an integer), and Kustomize refuses a tag that is not a string.
@pytest.mark.parametrize("tag", ["474e135", "1234567", "0123456", "3f2a9c1", "20260101-120000", "1e5"])
def test_the_image_tag_is_always_a_string(set_image_tag, tag):
    image = set_image_tag(tag)["images"][0]
    assert image["newTag"] == tag and isinstance(image["newTag"], str)


def test_an_invalid_tag_is_refused(set_image_tag):
    with pytest.raises(SystemExit, match="invalid OCI tag"):
        set_image_tag('dev" #')
