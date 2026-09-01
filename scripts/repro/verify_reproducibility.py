#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SHA256_REF = re.compile(r"@sha256:[0-9a-f]{64}$")
REMOTE_ACTION = re.compile(r"^\s*-?\s*uses:\s*([^\s@]+)@([^\s#]+)", re.MULTILINE)
FORBIDDEN_ALIAS = "late" + "st"


def fail(message: str) -> None:
    raise SystemExit(f"reproducibility policy: {message}")


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def readable_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def verify_forbidden_alias() -> None:
    token = re.compile(rf"\b{re.escape(FORBIDDEN_ALIAS)}\b", re.IGNORECASE)
    offenders: list[str] = []
    for path in tracked_files():
        text = readable_text(path)
        if text is not None and token.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        fail("forbidden moving alias found in: " + ", ".join(sorted(offenders)))


def verify_language_lockfiles() -> None:
    requirements: list[tuple[str, tuple[str, ...]]] = [
        ("mise.toml", ("mise.lock",)),
        ("pyproject.toml", ("uv.lock",)),
        ("Cargo.toml", ("Cargo.lock",)),
        ("go.mod", ("go.sum",)),
        ("package.json", ("pnpm-lock.yaml", "yarn.lock", "package-lock.json", "bun.lock")),
    ]
    for manifest, locks in requirements:
        if not (ROOT / manifest).exists():
            continue
        if not any((ROOT / lock).exists() for lock in locks):
            fail(f"{manifest} exists but none of its required lockfiles exist: {', '.join(locks)}")


def verify_mise_tools() -> None:
    config = tomllib.loads((ROOT / "mise.toml").read_text(encoding="utf-8"))
    tools = config.get("tools", {})
    if not isinstance(tools, dict) or not tools:
        fail("mise.toml must define explicit tools")
    exact = re.compile(r"^\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
    for name, value in tools.items():
        if not isinstance(value, str) or not exact.fullmatch(value):
            fail(f"mise tool {name!r} must use one exact version, got {value!r}")


def verify_actions() -> None:
    lock = json.loads((ROOT / "locks/actions.lock.json").read_text(encoding="utf-8"))
    entries = lock.get("actions", {})
    if not isinstance(entries, dict):
        fail("locks/actions.lock.json has invalid actions map")
    for name, entry in entries.items():
        commit = str(entry.get("commit", ""))
        if not HEX40.fullmatch(commit):
            fail(f"action {name} is not locked to a full commit SHA")

    for workflow in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        text = workflow.read_text(encoding="utf-8")
        for match in REMOTE_ACTION.finditer(text):
            name, ref = match.groups()
            if name.startswith("./"):
                continue
            if not HEX40.fullmatch(ref):
                fail(f"{workflow.relative_to(ROOT)} uses non-SHA action reference {name}@{ref}")
            locked = entries.get(name)
            if locked is None:
                fail(f"{workflow.relative_to(ROOT)} uses unregistered action {name}")
            if ref != locked.get("commit"):
                fail(f"{workflow.relative_to(ROOT)} action {name} differs from actions.lock.json")


def verify_hf_revisions() -> None:
    path = ROOT / "locks/hf-revisions.lock.json"
    if not path.exists():
        fail("locks/hf-revisions.lock.json is required")
    lock = json.loads(path.read_text(encoding="utf-8"))
    repos = lock.get("repositories", {})
    if not isinstance(repos, dict) or not repos:
        fail("Hugging Face revision lock is empty")
    for name, entry in repos.items():
        revision = str(entry.get("revision", ""))
        if not HEX40.fullmatch(revision):
            fail(f"Hugging Face entry {name} must use a full commit revision")


def verify_containers() -> None:
    path = ROOT / "locks/containers.lock.json"
    if not path.exists():
        fail("locks/containers.lock.json is required")
    lock = json.loads(path.read_text(encoding="utf-8"))
    images = lock.get("images", {})
    if not isinstance(images, dict) or not images:
        fail("container lock is empty")
    for name, entry in images.items():
        reference = str(entry.get("reference", ""))
        if not SHA256_REF.search(reference):
            fail(f"container {name} must use an OCI digest reference")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if stripped.startswith("ARG BASE_IMAGE="):
            fail("Dockerfile must not provide a mutable BASE_IMAGE default")
        if stripped.startswith("FROM ") and "${BASE_IMAGE}" not in stripped:
            image = stripped.split()[1]
            if not SHA256_REF.search(image):
                fail(f"Dockerfile FROM must use an OCI digest: {image}")


def main() -> None:
    verify_forbidden_alias()
    verify_language_lockfiles()
    verify_mise_tools()
    verify_actions()
    verify_hf_revisions()
    verify_containers()
    print("reproducibility policy: ok")


if __name__ == "__main__":
    main()
