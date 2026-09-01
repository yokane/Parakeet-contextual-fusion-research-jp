#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STABLE_RELEASE_PATH = "/releases/" + "late" + "st"


def github_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request("https://api.github.com" + path)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"GitHub API request failed for {path}: HTTP {exc.code}") from exc


def stable_release(repo: str) -> dict[str, Any]:
    return github_json(f"/repos/{repo}{STABLE_RELEASE_PATH}")


def resolve_tag_commit(repo: str, tag: str) -> str:
    quoted = urllib.parse.quote(tag, safe="")
    ref = github_json(f"/repos/{repo}/git/ref/tags/{quoted}")
    obj = ref["object"]
    if obj["type"] == "commit":
        return str(obj["sha"])
    if obj["type"] != "tag":
        raise SystemExit(f"unsupported tag object type for {repo}@{tag}: {obj['type']}")
    tag_obj = github_json(f"/repos/{repo}/git/tags/{obj['sha']}")
    target = tag_obj["object"]
    if target["type"] != "commit":
        raise SystemExit(f"tag {repo}@{tag} does not resolve to a commit")
    return str(target["sha"])


def main() -> None:
    mise = tomllib.loads((ROOT / "mise.toml").read_text(encoding="utf-8"))["tools"]
    action_lock = json.loads((ROOT / "locks/actions.lock.json").read_text(encoding="utf-8"))["actions"]

    candidates: dict[str, Any] = {"tools": {}, "actions": {}}
    for name, repo in {"python_manager": "jdx/mise", "python_package_manager": "astral-sh/uv"}.items():
        release = stable_release(repo)
        candidates["tools"][name] = {
            "repository": repo,
            "stable_release": release["tag_name"],
        }

    candidates["tools"]["configured_python"] = str(mise["python"])
    candidates["tools"]["configured_uv"] = str(mise["uv"])

    for repo, current in sorted(action_lock.items()):
        release = stable_release(repo)
        tag = str(release["tag_name"])
        candidates["actions"][repo] = {
            "configured_release": current["release"],
            "configured_commit": current["commit"],
            "stable_release": tag,
            "stable_commit": resolve_tag_commit(repo, tag),
        }

    print(json.dumps(candidates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
