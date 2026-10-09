probes/hub/persona_hub_probe.py
----
"""Probe Context Hub persona repos from the deployment environment (read-only).

Checks what loading a persona preset (an agent repo holding ``AGENTS.md`` and
``skills/<name>/SKILL.md``, inline or linked) needs from the LangSmith instance:
that the Hub directory endpoints exist, that one key reads several workspaces,
how a persona repo is laid out, whether linked skills resolve, and how long a
pull takes cold, warm and under a burst.

Requires Python 3.12+ and the ``langsmith`` package already in the image (no
repository imports). Reads ``LANGSMITH_API_KEY`` and ``LANGSMITH_ENDPOINT``.
Output holds counts, sizes, timings, entry types and, unless ``--redact``,
repo names truncated to 40 characters. It never prints file content or keys.

    python persona_hub_probe.py --list --workspace-id WS1 --workspace-id WS2
    python persona_hub_probe.py --persona=-/incident-manager --repeat 5 --burst 8
    python persona_hub_probe.py --persona owner/incident-manager --version prod --workspace-id WS1

Exit 0: every requested call succeeded. Exit 1: a call failed (the line says
which). Exit 2: invalid input or missing environment.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any


MAX_NAMES = 20
MAX_LINKS = 10


def _name(value: str, redact: bool) -> str:
    return "<redacted>" if redact else value[:40]


def _err(exc: BaseException) -> str:
    """The exception class and HTTP status only: SDK messages embed repo paths and endpoints."""
    status = re.search(r"\b([45]\d\d)\b", str(exc))
    return f"{type(exc).__name__} status={status.group(1) if status else '-'}"


def _tags(tags: list[str] | None, redact: bool) -> str:
    tags = tags or []
    return f"{len(tags)} tags" if redact else str([tag[:30] for tag in tags[:5]])


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _client(workspace_id: str | None) -> Any:
    from langsmith import Client

    if workspace_id is None:
        return Client()
    try:
        return Client(workspace_id=workspace_id)
    except TypeError:
        print("FAIL client: this langsmith version has no workspace_id parameter")
        sys.exit(2)


def _section_info(client: Any) -> bool:
    print("== 1. instance")
    # The SDK swallows a failed /info and returns an empty record, so an empty version is the failure signal.
    version = str(getattr(client.info, "version", "") or "")
    if not version:
        print("FAIL info: /info unavailable or returned no version (inconclusive)")
        return False
    print(f"version={version[:20]}")
    return True


def _section_list(client: Any, label: str, redact: bool) -> bool:
    print(f"== 2. listing workspace={label}")
    ok = True
    for kind in ("agent", "skill"):
        start = time.perf_counter()
        try:
            listing = client.list_agents(limit=100) if kind == "agent" else client.list_skills(limit=100)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL list_{kind}s: {_err(exc)}")
            ok = False
            continue
        repos = list(getattr(listing, "repos", []) or [])
        foreign = sum(1 for repo in repos if not getattr(repo, "owner", None))
        print(f"{kind}s={len(repos)} total={getattr(listing, 'total', '?')} without_owner={foreign} ms={_ms(start)}")
        for repo in repos[:MAX_NAMES]:
            print(
                f"  {_name(f'{repo.owner}/{repo.repo_handle}', redact)} public={getattr(repo, 'is_public', '?')} {_tags(getattr(repo, 'tags', None), redact)}"
            )
    return ok


def _layout(files: dict[str, Any]) -> dict[str, Any]:
    types = Counter(entry.type for entry in files.values())
    inline_skills = {
        path.split("/")[1] for path, entry in files.items() if entry.type == "file" and path.startswith("skills/") and path.endswith("/SKILL.md")
    }
    linked = {path: entry for path, entry in files.items() if entry.type == "skill"}
    agents_md = [path for path in files if path.lower() in ("agents.md", "agent.md")]
    size = sum(len(entry.content.encode()) for entry in files.values() if entry.type == "file")
    return {"types": dict(types), "inline_skills": len(inline_skills), "linked": linked, "agents_md": agents_md, "bytes": size}


def _frontmatter_ok(content: str) -> bool:
    parts = content.split("---", 2)
    return content.startswith("---") and len(parts) == 3 and "name:" in parts[1] and "description:" in parts[1]


def _resolve_links(client: Any, files: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Pull every linked skill the way a loader would: pinned links at their commit, floating ones at latest."""
    results = []
    for path, entry in list(files.items()):
        if entry.type != "skill":
            continue
        results.append((path, entry, client.pull_skill(f"{entry.owner or '-'}/{entry.repo_handle}", version=entry.commit_hash)))
    return results


def _load_persona(client: Any, ident: str, version: str | None) -> float:
    """Full persona load (parent repo plus every linked skill); returns elapsed ms."""
    start = time.perf_counter()
    context = client.pull_agent(ident, version=version)
    _resolve_links(client, context.files)
    return _ms(start)


def _stats(samples: list[float]) -> str:
    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return f"p50_ms={statistics.median(ordered)} p95_ms={p95}"


def _section_persona(client: Any, ident: str, version: str | None, repeat: int, burst: int, redact: bool) -> bool:
    print(f"== 3. persona {_name(ident, redact)} version={version or 'latest'}")
    start = time.perf_counter()
    try:
        context = client.pull_agent(ident, version=version)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL pull_agent: {_err(exc)}")
        return False
    print(f"pull_agent first_ms={_ms(start)} commit={context.commit_hash[:8]}")
    layout = _layout(context.files)
    print(
        f"entries={layout['types']} bytes_inline={layout['bytes']} agents_md={len(layout['agents_md'])} "
        f"inline_skills={layout['inline_skills']} linked_skills={len(layout['linked'])}"
    )
    bad = [path for path, entry in context.files.items() if entry.type == "file" and path.endswith("SKILL.md") and not _frontmatter_ok(entry.content)]
    print(f"inline SKILL.md without name/description frontmatter={len(bad)}")
    floating = sum(1 for entry in layout["linked"].values() if not entry.commit_hash)
    print(f"linked skills pinned={len(layout['linked']) - floating} floating={floating}")
    for path, entry in list(layout["linked"].items())[:MAX_LINKS]:
        pin = (entry.commit_hash or "floating")[:8]
        start = time.perf_counter()
        try:
            skill = client.pull_skill(f"{entry.owner or '-'}/{entry.repo_handle}", version=entry.commit_hash)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL pull_skill path={_name(path, redact)} pin={pin}: {_err(exc)}")
            return False
        nested = Counter(e.type for e in skill.files.values())
        print(f"  link path={_name(path, redact)} pin={pin} ms={_ms(start)} entries={dict(nested)} SKILL.md_at_root={'SKILL.md' in skill.files}")
    try:
        loads = [_load_persona(client, ident, version) for _ in range(max(repeat, 1))]
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL full load: {_err(exc)}")
        return False
    print(f"full load (parent + links) sequential n={len(loads)} {_stats(loads)}")
    if burst > 1:
        start = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=burst) as pool:
                results = list(pool.map(lambda _: _load_persona(client, ident, version), range(burst)))
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL burst: {_err(exc)}")
            return False
        print(f"full load burst={burst} wall_ms={_ms(start)} {_stats(results)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--workspace-id", action="append", default=[], help="workspace to address (repeatable); omitted means the key's default")
    parser.add_argument("--list", action="store_true", help="list agent and skill repos per workspace")
    parser.add_argument("--persona", action="append", default=[], help="agent repo identifier owner/name or -/name (repeatable)")
    parser.add_argument("--version", help="commit hash or tag to pin the persona pull")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--burst", type=int, default=0)
    parser.add_argument("--redact", action="store_true", help="hide repo names")
    args = parser.parse_args()
    # SDK warnings print endpoints and raw exception text, which the redaction policy must not leak.
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("FAIL env: LANGSMITH_API_KEY is not set")
        return 2
    if not args.list and not args.persona:
        print("FAIL input: pass --list and/or --persona")
        return 2
    ok = True
    for workspace in args.workspace_id or [None]:
        label = "default" if workspace is None else workspace[:8] + "..."
        client = _client(workspace)
        ok &= _section_info(client)
        if args.list:
            ok &= _section_list(client, label, args.redact)
        for ident in args.persona:
            ok &= _section_persona(client, ident, args.version, args.repeat, args.burst, args.redact)
    print("RESULT ok" if ok else "RESULT failures above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

-------

