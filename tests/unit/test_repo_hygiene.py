"""Guards for a public repository: no personal or machine-specific content, no marketing filler,
no development history in the docs.

Scans the files git would publish (tracked + untracked, minus .gitignore), so it works
before the first commit too.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()

# Words and word pairs that must never appear, stored as SHA-256 of the lowercase text so this
# file does not publish them either. To add one:
#   python -c "import hashlib; print(hashlib.sha256(b'some words').hexdigest())"
FORBIDDEN_HASHES = {
    "3464f13b59481aa6bc54e9e86694a2e52344ba3dccf24c46475a2f1596271619",
    "02540942346fdfbf21a94f8b823af2ae315f20cb71721c2cc6e205097d560840",
    "320dc1c2c337e7ef3415e7d56d46b035bf6e0f6639c9691f95590b9cf9e70bc0",
    "14143a0020836bd35949bb630f14cec3c81dac96047bb6ee74b0f5609f54db8d",
    "89b7505ad79ad4892d6f2f110320da7b79e4110e0117b8249de318688c3ad83b",
    "8a91bb34721c6489a602f83428cff4adf368331fb8e7435dd8dc993cf482bfd5",
    "b8db18c23a14936f2ec67ccf62ae7a4e5138ea1525294eddc8f9cfad091dfb1f",
    "1e6b294b3ffd8eca07d7d172f4cbc637794c5ad8cb741dbf8a8d991cbb83d52b",
    "92d46204d8e9aeb3b37873c794348b3132ab9c9f33a89f2063c46180914c0104",
    "fb9d78cb1e099ccf5c7b06dca767a1f6b196d719961feb11e0d341049d609164",
    "438287bf455f1f46d6d96d642f02b53006f933f291673e27cad69ba6f98767ec",
    "3cdc6d6e23e0f9475e986208c25b453d7aca1b947d527659c718359b91c81c06",
    "b67edcdb95503aaeb8980b7017c4f7488871ce5908a5c945fb1b6334b808e960",
    "0e6a8e0b849ed9b064c5a25e1ee5592f427e3eb9d250e42069ce46147d00e8d4",
    "ea96f48731c0d789627d827cf2cd69a4e2b7db961b8986f20feb74e99cd16b2e",
    "704c8f27d5c1d0eabb7d0cc667c16049eed0aa3f588a6130ba8108aa04b745ad",
    "1acde07d0ad089044686f500d8e7d9890fa521a0daf7f9f93a417b40fcbbf80f",
    "43d84b38a5e8141f75fd9fff85da904dde6a896bb95abecaea1c98620d8f2cd6",
    "45c849714eed79ed50fe3922aa4fe97b6debc59c784ffa1416b198d5d22275ea",
    "55535fdfc5fbbbba2dff2477459c8ed013e538d7b2f64c6a04aea67159041dd3",
    "08fcf00abeba87664c7f3c3d4d05351a7816e0a31a7f95e6c8b6c266407982aa",
    "184beb3ae27e77ae0c1b4e3e271a8a101925deb4639adbffc20b774be78aa46b",
    "ddb8adb7bc1c7c4a4fe5c3730be06f9afc15e5f6d088e8da033e2b2e5bd57f13",
    "f5331c1ea92ea9f14a8c0c1791bf75dd6b1026b7dac2520aa7ec0a0991ef6c8c",
    "a3c8334a028915d85fa07e51cb15540a59ec0afb52347fd984492db4e5a084d3",
}
WORD = re.compile(r"[a-z0-9]+")

# Machine-specific paths.
MACHINE_PATHS = [r"[A-Za-z]:\\Users\\", r"/home/[a-z]+/", r"/Users/[a-z]+/"]
# The author's identity is expected in these files only.
IDENTITY = [r"\bamine\b", r"\bjmili\b"]
IDENTITY_OK = {"LICENSE", "pyproject.toml", "README.md"}
# A mailbox in a public repository is harvested within days: the only real address allowed is
# GitHub's no-reply one. Reserved domains (RFC 2606, RFC 6762) are fine in examples and tests.
MAIL = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
MAIL_OK = re.compile(r"(^|\.)(users\.noreply\.github\.com|example\.(com|org|net)|internal|local|localhost|test)$")

# The docs describe the system as it is; its development history is not part of them.
HISTORY_PHRASES = [r"lessons? learned", r"the hard way", r"things that broke", r"\bwe learned\b",
                   r"\bi learned\b", r"\bturned out\b", r"\bgotcha", r"\bwas broken\b"]

# Filler words that make docs read like a brochure. Docs only: code may say "robust" in a comment.
BANNED_IN_DOCS = [
    "seamless", "robust", "leverage", "delve", "cutting-edge", "state-of-the-art", "game-changer",
    "unlock", "empower", "effortless", "in today's", "it's worth noting", "comprehensive",
    "supercharge", "elevate",
]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-⛿✀-➿️]")
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".lock"}


def published_files() -> list[Path]:
    if not shutil.which("git") or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = [ROOT / line for line in out.splitlines() if line and line != SELF]
    return [f for f in files if f.is_file() and f.suffix.lower() not in BINARY]


def _text(f: Path) -> str:
    return f.read_text(encoding="utf-8", errors="replace")


def _forbidden(text: str) -> bool:
    words = WORD.findall(text.lower())
    candidates = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}
    return any(hashlib.sha256(c.encode()).hexdigest() in FORBIDDEN_HASHES for c in candidates)


def test_enough_files_are_scanned():
    assert len(published_files()) > 50       # a broken git call must not make the guards pass silently


def test_forbidden_words_detector_works():
    assert _forbidden("some text about Job   Search here")
    assert not _forbidden("a job queue and a search index")


def test_no_forbidden_words():
    hits = [f.relative_to(ROOT).as_posix() for f in published_files()
            if _forbidden(f.relative_to(ROOT).as_posix()) or _forbidden(_text(f))]
    assert not hits, f"forbidden words in: {hits}"


def test_no_machine_paths_and_identity_only_where_expected():
    hits = []
    for f in published_files():
        rel, text = f.relative_to(ROOT).as_posix(), _text(f)
        markers = MACHINE_PATHS + ([] if rel in IDENTITY_OK else IDENTITY)
        for marker in markers:
            for m in re.finditer(marker, text, flags=re.IGNORECASE):
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)!r}")
    assert not hits, "\n".join(hits)


def test_no_personal_mail_address():
    hits = []
    for f in published_files():
        rel, text = f.relative_to(ROOT).as_posix(), _text(f)
        hits += [f"{rel}: {m.group(0)}" for m in MAIL.finditer(text) if not MAIL_OK.search(m.group(1).lower())]
    assert not hits, "\n".join(hits)


def test_no_development_history():
    hits = []
    for f in published_files():
        if "/datalake/" in f.as_posix():
            continue
        text = _text(f).lower()
        hits += [f"{f.relative_to(ROOT).as_posix()}: {p}" for p in HISTORY_PHRASES if re.search(p, text)]
    assert not hits, "\n".join(hits)


def test_docs_avoid_filler_words():
    hits = []
    for f in published_files():
        if f.suffix.lower() != ".md" or "/datalake/" in f.as_posix():
            continue
        text = _text(f).lower()
        # Whole words only: "an elevated prompt" (Windows) is fine, "elevate your stack" is not.
        hits += [f"{f.relative_to(ROOT).as_posix()}: {w}" for w in BANNED_IN_DOCS
                 if re.search(rf"\b{re.escape(w)}\b", text)]
    assert not hits, "\n".join(hits)


def test_no_emoji():
    hits = [f.relative_to(ROOT).as_posix() for f in published_files() if EMOJI.search(_text(f))]
    assert not hits, hits


def test_text_files_use_lf():
    hits = [f.relative_to(ROOT).as_posix() for f in published_files()
            if f.suffix.lower() != ".ps1" and b"\r\n" in f.read_bytes()]
    assert not hits, f"CRLF line endings (see .gitattributes): {hits}"


def test_published_ports_are_bound_to_localhost():
    """Nothing in the lab has authentication worth exposing: every port published on the host must
    name 127.0.0.1, or Docker listens on all interfaces."""
    k3d = (ROOT / "deploy/k8s/k3d.yaml").read_text(encoding="utf-8")
    ports = re.findall(r"^\s*- port:\s*(\S+)", k3d, flags=re.MULTILINE)
    assert ports and all(p.startswith("127.0.0.1:") for p in ports), ports
    assert re.search(r'host:\s*"127\.0\.0\.1"', k3d), "registry host"

    compose = (ROOT / "deploy/compose/docker-compose.yml").read_text(encoding="utf-8")
    mappings = re.findall(r'"((?:[\d.]+:)?\d+:\d+)"', compose)
    assert mappings and all(m.startswith("127.0.0.1:") for m in mappings), mappings


def test_the_three_signals_are_kept_72_hours_in_both_setups():
    """Logs, traces and metrics share one retention, so a trace and its log lines expire together."""
    settings = {
        "deploy/compose/loki.yaml": "retention_period: 72h",
        "deploy/compose/tempo.yaml": "block_retention: 72h",
        "deploy/compose/docker-compose.yml": "--storage.tsdb.retention.time=3d",
        "deploy/k8s/platform/values/loki.yaml": "retention_period: 72h",
        "deploy/k8s/platform/values/tempo.yaml": "retention: 72h",
        "deploy/k8s/platform/values/kube-prometheus-stack.yaml": "retention: 3d",
    }
    missing = [path for path, text in settings.items() if text not in (ROOT / path).read_text(encoding="utf-8")]
    assert not missing, f"retention is not 72 hours in: {missing}"
    for path in ("deploy/compose/loki.yaml", "deploy/k8s/platform/values/loki.yaml"):
        assert "retention_enabled: true" in (ROOT / path).read_text(encoding="utf-8"), path
