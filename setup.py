"""One-time setup for the AI Daily Digest Managed Agent.

Idempotent: re-running reuses existing resources unless config drifted.
Prints AGENT_ID and ENVIRONMENT_ID to be saved as GitHub Actions secrets.

Usage:
    python setup.py                    # reuse-if-matching, else recreate
    python setup.py --force            # always create fresh
    python setup.py --prune-duplicates # archive same-name duplicates
"""

import argparse
import hashlib
import json
import logging
import os
import sys

import anthropic

ENV_NAME = "ai-digest-env"
AGENT_NAME = "AI Daily Digest"
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are an AI news curator. Your job is to find the most important AI news from the last 24 hours and produce two outputs:

1. A detailed daily note (Markdown) saved to the Obsidian vault
2. A short bullet-point summary for Slack

## Sources to check
- Anthropic blog and announcements
- OpenAI blog and announcements
- Google AI / DeepMind / Gemini announcements
- AI developer tools news (Claude Code, Cursor, Windsurf, Copilot, Replit, v0, Bolt, etc.)

## What to include
- New product launches and features
- Major updates to existing tools
- New developer-facing capabilities
- Anything relevant to "vibe coding" and AI-assisted development

## What to exclude
- Academic papers and research unless they ship a product
- Rumors or speculation
- Funding/hiring news unless it directly affects a product

## Workflow
1. Use web_search to find recent AI news from each source category.
2. Use web_fetch to get details from relevant articles.
3. (Optional) Read the mounted repo at `/workspace/ai-daily-digest` via `read`/`bash` to see yesterday's note or check whether today's file already exists. READ ONLY - do not attempt to write or git push from bash, you lack credentials for that.
4. Compose the full markdown daily note content in memory, following the Daily note format below.
5. Compute the SHA-256 of the UTF-8-encoded content. A reliable bash recipe:
     printf '%s' "<your exact content>" | sha256sum
   (or use Python via bash: `python3 -c "import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())" <<< "<content>"`)
   Whatever you pass as `content` to the tool MUST be the same bytes you hashed.
6. Call `write_daily_note` with input `{"content": "<full markdown>", "content_sha256": "<64-hex>"}`. The orchestrator will verify the hash, then commit the file at YYYY/MM/YYYY-MM-DD.md on main via the GitHub REST API. It returns one of:
   - `{"committed": true, "commit_sha": "...", "no_op": false}` - real commit happened
   - `{"committed": false, "no_op": true}` - today's file was already byte-identical, commit skipped (still counts as success)
   - `{"committed": false, "error": "..."}` with is_error=true - hash mismatch or API error; you may retry once with a recomputed hash
7. Call `send_slack_message` with a short bullet-point summary. Always fire - even on no-op reruns. Duplicate Slack messages on reruns are allowed by design.

## Daily note format
---
# AI Daily Digest - YYYY-MM-DD

## Anthropic
- **[Title](url)** - 2-3 sentence summary

## OpenAI
- **[Title](url)** - 2-3 sentence summary

## Google AI
- **[Title](url)** - 2-3 sentence summary

## Developer Tools & Vibe Coding
- **[Title](url)** - 2-3 sentence summary

## Slack summary (for reference)
> (the bullet-point version sent to Slack)
---

## Slack summary format
Keep it scannable. Example:
- Anthropic: Claude Code v2.2 ships multi-file editing - link
- OpenAI: GPT-5 API now available in preview - link
- Google: Gemini 2.5 adds tool use support - link

If no news found for a category, write "No updates today." - do not omit the section.
"""

AGENT_TOOLSET = {
    "type": "agent_toolset_20260401",
    "configs": [
        {"name": "write", "enabled": False},
        {"name": "edit", "enabled": False},
    ],
}

WRITE_DAILY_NOTE_TOOL = {
    "type": "custom",
    "name": "write_daily_note",
    "description": (
        "Commit today's daily digest markdown to the Obsidian vault on the main "
        "branch. The orchestrator handles the actual GitHub REST API write. You "
        "supply the full markdown content AND its SHA-256 hash (of the UTF-8 "
        "bytes). The orchestrator verifies the hash matches the content it "
        "received, then creates or updates YYYY/MM/YYYY-MM-DD.md on main. "
        "Returns one of: {committed: true, commit_sha, no_op: false} on a real "
        "commit; {committed: false, no_op: true} if today's file was already "
        "byte-identical; or is_error=true with {error: ...} on hash mismatch "
        "or network failure. Call this exactly once per run, before "
        "send_slack_message. If it returns is_error, you may retry once with a "
        "recomputed hash."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The complete markdown body of today's daily digest. The "
                    "orchestrator writes these exact UTF-8 bytes to main."
                ),
            },
            "content_sha256": {
                "type": "string",
                "description": (
                    "The SHA-256 hex digest (64 lowercase hex chars) of the "
                    "UTF-8 bytes of `content`. Computed by you and verified by "
                    "the orchestrator before any write is attempted."
                ),
            },
        },
        "required": ["content", "content_sha256"],
    },
}

SLACK_CUSTOM_TOOL = {
    "type": "custom",
    "name": "send_slack_message",
    "description": (
        "Send a short bullet-point summary of today's AI news digest to the team "
        "Slack channel. Call this exactly once per run, after write_daily_note "
        "has returned a non-error result. The orchestrator posts the summary "
        "verbatim to a Slack webhook - keep it scannable, one bullet per major "
        "item, with links."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "The Slack-formatted bullet summary to post.",
            },
        },
        "required": ["summary"],
    },
}


# --- Shape-tolerant helpers (see plan) -------------------------------------


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _model_id(m):
    if isinstance(m, str):
        return m
    v = _get(m, "id")
    if v is not None:
        return v
    raise RuntimeError(
        f"cannot extract model id from shape {type(m).__name__!r}: {m!r}"
    )


def _sorted_if_list(v):
    return sorted(v) if isinstance(v, list) else v


def _as_items(obj):
    """Return list of (key, value) pairs from a dict, pydantic model, or None.

    The Anthropic SDK returns pydantic models (e.g. BetaPackages) where dict
    access would fail. Normalize both shapes to plain items.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.items())
    dump = getattr(obj, "model_dump", None)  # pydantic v2
    if callable(dump):
        return list(dump(exclude_none=True).items())
    dump = getattr(obj, "dict", None)  # pydantic v1
    if callable(dump):
        return list(dump(exclude_none=True).items())
    return []


# --- Canonicalization -------------------------------------------------------


def canonical_tool(t):
    """Normalize a tool entry for diffing."""
    out = {"type": _get(t, "type")}
    name = _get(t, "name")
    if name is not None:
        out["name"] = name
    mcp_server_name = _get(t, "mcp_server_name")
    if mcp_server_name is not None:
        out["mcp_server_name"] = mcp_server_name

    default_config = _get(t, "default_config")
    if default_config is not None:
        pp = _get(default_config, "permission_policy")
        out["default_config"] = {
            "permission_policy": {"type": _get(pp, "type")} if pp is not None else None,
        }

    configs = _get(t, "configs")
    if configs:
        norm = [
            {"name": _get(c, "name"), "enabled": _get(c, "enabled", True)}
            for c in configs
        ]
        out["configs"] = sorted(norm, key=lambda c: c["name"] or "")

    input_schema = _get(t, "input_schema")
    if input_schema is not None:
        out["input_schema"] = input_schema if isinstance(input_schema, dict) else dict(
            input_schema
        )

    description = _get(t, "description")
    if description is not None:
        out["description"] = description.strip()

    return out


def canonical_agent(a):
    servers = [
        {
            "name": _get(s, "name"),
            "url": _get(s, "url"),
            "type": _get(s, "type"),
        }
        for s in (_get(a, "mcp_servers") or [])
    ]
    tools = [canonical_tool(t) for t in (_get(a, "tools") or [])]
    return {
        "model": _model_id(_get(a, "model")),
        "system": (_get(a, "system") or "").strip(),
        "mcp_servers": sorted(servers, key=lambda s: (s["name"] or "")),
        "tools": sorted(
            tools,
            key=lambda t: (t.get("type", ""), t.get("name", "") or t.get("mcp_server_name", "")),
        ),
    }


def canonical_env(e):
    cfg = _get(e, "config")
    net = _get(cfg, "networking")
    allowed = _get(net, "allowed_hosts")
    return {
        "config.type": _get(cfg, "type"),
        "networking.type": _get(net, "type"),
        "allowed_hosts": _sorted_if_list(allowed) if isinstance(allowed, list) else [],
        "packages": {k: _sorted_if_list(v) for k, v in _as_items(_get(cfg, "packages"))},
    }


def canonical_hash(canonical_dict):
    return hashlib.sha256(
        json.dumps(canonical_dict, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --- Intended configs --------------------------------------------------------


def intended_env():
    return canonical_env(
        type(
            "E",
            (),
            {
                "config": {
                    "type": "cloud",
                    "networking": {"type": "unrestricted"},
                }
            },
        )
    )


def intended_agent():
    return canonical_agent(
        type(
            "A",
            (),
            {
                "model": MODEL,
                "system": SYSTEM_PROMPT,
                "mcp_servers": [],
                "tools": [AGENT_TOOLSET, WRITE_DAILY_NOTE_TOOL, SLACK_CUSTOM_TOOL],
            },
        )
    )


# --- Pagination / lookup ----------------------------------------------------


def _paginate(list_fn):
    """Yield every item from a paginated SDK list endpoint."""
    cursor = None
    while True:
        kwargs = {"limit": 100}
        if cursor is not None:
            kwargs["after_id"] = cursor
        page = list_fn(**kwargs)
        data = _get(page, "data") or []
        for item in data:
            yield item
        has_more = _get(page, "has_more", False)
        if not has_more:
            return
        last_id = _get(data[-1], "id") if data else None
        if last_id is None:
            return
        cursor = last_id


def _is_archived(obj):
    archived_at = _get(obj, "archived_at")
    if archived_at:
        return True
    status = _get(obj, "status")
    if isinstance(status, str) and status.lower() == "archived":
        return True
    return False


def find_matching(list_fn, name):
    """Return list of non-archived items with matching name, newest first."""
    matches = [
        item
        for item in _paginate(list_fn)
        if _get(item, "name") == name and not _is_archived(item)
    ]
    matches.sort(key=lambda i: _get(i, "created_at") or "", reverse=True)
    return matches


# --- Main -------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Always create fresh; don't reuse."
    )
    parser.add_argument(
        "--prune-duplicates",
        action="store_true",
        help="Archive non-winning same-name duplicates.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("setup")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error(
            "ANTHROPIC_API_KEY is not set. Export it in your shell before running, e.g.:\n"
            '  PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."\n'
            '  bash/WSL:   export ANTHROPIC_API_KEY="sk-ant-..."'
        )
        return 2

    client = anthropic.Anthropic()

    env_id = ensure_environment(client, log, force=args.force, prune=args.prune_duplicates)
    agent_id = ensure_agent(client, log, force=args.force, prune=args.prune_duplicates)

    print()
    print("=" * 60)
    print("Add these to the orchestrator repo's GitHub Actions secrets:")
    print(f"  ENVIRONMENT_ID={env_id}")
    print(f"  AGENT_ID={agent_id}")
    print("=" * 60)
    return 0


def ensure_environment(client, log, *, force, prune):
    intended_hash = canonical_hash(intended_env())
    log.info("intended env canonical hash: %s", intended_hash)

    if not force:
        matches = find_matching(client.beta.environments.list, ENV_NAME)
        if len(matches) > 1:
            log.warning(
                "multiple non-archived environments named %r: %s (keeping newest)",
                ENV_NAME,
                [_get(m, "id") for m in matches],
            )
            if prune:
                for extra in matches[1:]:
                    eid = _get(extra, "id")
                    log.info("archiving duplicate environment %s", eid)
                    client.beta.environments.archive(eid)
        if matches:
            winner = matches[0]
            remote_hash = canonical_hash(canonical_env(winner))
            if remote_hash == intended_hash:
                log.info("reusing environment %s (config matches)", _get(winner, "id"))
                return _get(winner, "id")
            log.warning(
                "environment %s config drift (remote=%s intended=%s) - recreating",
                _get(winner, "id"),
                remote_hash,
                intended_hash,
            )
            try:
                client.beta.environments.archive(_get(winner, "id"))
            except Exception as e:
                log.warning("could not archive stale env: %s", e)

    log.info("creating environment %r", ENV_NAME)
    env = client.beta.environments.create(
        name=ENV_NAME,
        config={"type": "cloud", "networking": {"type": "unrestricted"}},
    )
    eid = _get(env, "id")
    log.info("created environment: %s", eid)
    return eid


def ensure_agent(client, log, *, force, prune):
    intended_hash = canonical_hash(intended_agent())
    log.info("intended agent canonical hash: %s", intended_hash)

    if not force:
        matches = find_matching(client.beta.agents.list, AGENT_NAME)
        if len(matches) > 1:
            log.warning(
                "multiple non-archived agents named %r: %s (keeping newest)",
                AGENT_NAME,
                [_get(m, "id") for m in matches],
            )
            if prune:
                for extra in matches[1:]:
                    aid = _get(extra, "id")
                    log.info("archiving duplicate agent %s", aid)
                    client.beta.agents.archive(aid)
        if matches:
            winner = matches[0]
            remote_hash = canonical_hash(canonical_agent(winner))
            if remote_hash == intended_hash:
                log.info("reusing agent %s (config matches)", _get(winner, "id"))
                return _get(winner, "id")
            log.warning(
                "agent %s config drift (remote=%s intended=%s) - recreating",
                _get(winner, "id"),
                remote_hash,
                intended_hash,
            )
            try:
                client.beta.agents.archive(_get(winner, "id"))
            except Exception as e:
                log.warning("could not archive stale agent: %s", e)

    log.info("creating agent %r", AGENT_NAME)
    agent = client.beta.agents.create(
        name=AGENT_NAME,
        model=MODEL,
        system=SYSTEM_PROMPT,
        tools=[AGENT_TOOLSET, WRITE_DAILY_NOTE_TOOL, SLACK_CUSTOM_TOOL],
    )
    aid = _get(agent, "id")
    log.info("created agent: %s", aid)
    return aid


if __name__ == "__main__":
    sys.exit(main())
