#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def spdx_id(kind: str, name: str, version: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9.-]+", "-", f"{kind}-{name}-{version}").strip("-")
    digest = hashlib.sha256(f"{kind}\0{name}\0{version}".encode()).hexdigest()[:12]
    return f"SPDXRef-{stem}-{digest}"


def checksum(value: str | None) -> list[dict[str, str]] | None:
    if not value or not value.startswith("sha256:"):
        return None
    digest = value.split(":", 1)[1]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        return None
    return [{"algorithm": "SHA256", "checksumValue": digest.lower()}]


def package(
    *,
    kind: str,
    name: str,
    version: str,
    download_location: str = "NOASSERTION",
    purl: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "SPDXID": spdx_id(kind, name, version),
        "name": name,
        "versionInfo": version,
        "downloadLocation": download_location,
        "filesAnalyzed": False,
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
        "supplier": "NOASSERTION",
    }
    if purl:
        item["externalRefs"] = [
            {
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": purl,
            }
        ]
    hashes = checksum(sha256)
    if hashes:
        item["checksums"] = hashes
    return item


def mise_packages() -> list[dict[str, Any]]:
    lock = tomllib.loads((ROOT / "mise.lock").read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = []
    tools = lock.get("tools", {})
    for name, entries in sorted(tools.items()):
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            version = str(entry["version"])
            platforms = entry.get("platforms", {})
            platform = platforms.get("linux-x64") if isinstance(platforms, dict) else None
            if platform is None and isinstance(platforms, dict) and platforms:
                platform = platforms[sorted(platforms)[0]]
            platform = platform or {}
            packages.append(
                package(
                    kind="mise",
                    name=name,
                    version=version,
                    download_location=str(platform.get("url") or "NOASSERTION"),
                    sha256=str(platform.get("checksum") or "") or None,
                )
            )
    return packages


def uv_packages(lock_path: Path, scope: str) -> list[dict[str, Any]]:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = []
    for entry in lock.get("package", []):
        name = str(entry["name"])
        version = str(entry["version"])
        source = entry.get("source", {})
        sdist = entry.get("sdist", {})
        registry = source.get("registry") if isinstance(source, dict) else None
        url = sdist.get("url") if isinstance(sdist, dict) else None
        artifact_hash = sdist.get("hash") if isinstance(sdist, dict) else None
        packages.append(
            package(
                kind="pypi",
                name=f"{scope}:{name}",
                version=version,
                download_location=str(url or registry or "NOASSERTION"),
                purl=f"pkg:pypi/{name}@{version}",
                sha256=str(artifact_hash or "") or None,
            )
        )
    return packages


def action_packages() -> list[dict[str, Any]]:
    lock = json.loads((ROOT / "locks/actions.lock.json").read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = []
    for name, entry in sorted(lock["actions"].items()):
        commit = str(entry["commit"])
        packages.append(
            package(
                kind="github-action",
                name=name,
                version=commit,
                download_location=f"https://github.com/{name}/tree/{commit}",
                purl=f"pkg:github/{name}@{commit}",
            )
        )
    return packages


def hf_packages() -> list[dict[str, Any]]:
    lock = json.loads((ROOT / "locks/hf-revisions.lock.json").read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = []
    for logical_name, entry in sorted(lock["repositories"].items()):
        repo_id = str(entry["repo_id"])
        revision = str(entry["revision"])
        prefix = "datasets/" if entry["repo_type"] == "dataset" else ""
        packages.append(
            package(
                kind="hugging-face",
                name=f"{logical_name}:{repo_id}",
                version=revision,
                download_location=f"https://huggingface.co/{prefix}{repo_id}/tree/{revision}",
            )
        )
    return packages


def container_packages() -> list[dict[str, Any]]:
    lock = json.loads((ROOT / "locks/containers.lock.json").read_text(encoding="utf-8"))
    packages: list[dict[str, Any]] = []
    for logical_name, entry in sorted(lock["images"].items()):
        repository = str(entry["repository"])
        digest = str(entry["digest"])
        packages.append(
            package(
                kind="oci",
                name=f"{logical_name}:{repository}",
                version=digest,
                download_location=str(entry["reference"]),
                purl=f"pkg:oci/{repository}@{digest}",
                sha256=digest,
            )
        )
    return packages


def build_document() -> dict[str, Any]:
    metadata = json.loads((ROOT / "locks/sbom.lock.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    project = package(
        kind="project",
        name=str(pyproject["name"]),
        version=str(pyproject["version"]),
        download_location="https://github.com/yokane/Parakeet-contextual-fusion-research-jp",
    )
    dependencies = (
        mise_packages()
        + uv_packages(ROOT / "uv.lock", "asr")
        + uv_packages(ROOT / "tools/hf-bucket/uv.lock", "hf-bucket")
        + action_packages()
        + hf_packages()
        + container_packages()
    )
    dependencies.sort(key=lambda item: (item["name"], item["versionInfo"], item["SPDXID"]))
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": project["SPDXID"],
        }
    ]
    relationships.extend(
        {
            "spdxElementId": project["SPDXID"],
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": item["SPDXID"],
        }
        for item in dependencies
    )
    return {
        "spdxVersion": metadata["spdx_version"],
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "J-PACF-YOMI-TDT reproducibility inventory",
        "documentNamespace": "https://github.com/yokane/Parakeet-contextual-fusion-research-jp/spdx/reproducibility",
        "creationInfo": {
            "created": metadata["snapshot_created"],
            "creators": ["Tool: scripts/repro/generate_spdx.py"],
        },
        "packages": [project, *dependencies],
        "relationships": relationships,
    }


def render() -> str:
    return json.dumps(build_document(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic SPDX inventory from repository locks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    expected = render()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8")
        print(args.output)
        return
    current = args.check.read_text(encoding="utf-8") if args.check.exists() else ""
    if current != expected:
        raise SystemExit(f"SPDX inventory is stale: {args.check}")
    print(f"SPDX inventory matches locks: {args.check}")


if __name__ == "__main__":
    main()
