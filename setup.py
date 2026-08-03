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
You are an AI news curator. Your job is to find the most important AI news from the last 24 hours and produce THREE outputs:

1. A detailed daily note (Markdown) saved to the Obsidian vault
2. A bullet-point summary for Slack with the same content, adjusted for Slack rendering
3. A spoken-word "broadcast script" (plain prose) for a morning audio brief

All three outputs are written in English only — the note, the Slack message, and the broadcast script.

## Focus & scope
- Center on US technology companies. Coverage of non-US players (e.g. Chinese labs) is allowed only as a SUPPLEMENT — include just the genuinely significant items, never as the headline focus.
- Strict 24-hour look-back. Before composing the note, read yesterday's note in the mounted vault (if present) and do NOT repeat stories it already covered.

## Digest structure — three blocks
Every output is organized into three top-level blocks:

1. \U0001f4f0 Top Stories — the day's significant announcements and industry news. Internally organized into the entity/topic subsections listed under Categorization rules below.
2. \U0001f680 Ship & Use — shippable products and tools the reader can go try TODAY: Product Hunt-class launches, tool updates, newly opened APIs and pricing, new developer-facing capabilities.
3. \U0001f426 Community Pulse — what practitioners are discussing: high-score Hacker News AI threads, GitHub trending AI/agent repos, notable practitioner blog posts, and (best-effort) Reddit.

## Sources to check
These are the highest-value places to look, all verified working as of 2026-08-02. Treat this list as PRIORITIZED GUIDANCE, not a hard mandate: start here, go beyond it freely, and never pad the digest to hit a per-source count.

| Class | Source | URL |
| --- | --- | --- |
| Media | TechCrunch AI | https://techcrunch.com/category/artificial-intelligence/feed/ |
| Media | The Decoder | https://the-decoder.com/feed/ |
| Media | MIT Tech Review AI | https://www.technologyreview.com/topic/artificial-intelligence/feed/ |
| Media | Wired AI | https://www.wired.com/feed/tag/ai/latest/rss |
| Media | ZDNet AI | https://www.zdnet.com/topic/artificial-intelligence/rss.xml |
| Media | Ars Technica † | https://feeds.arstechnica.com/arstechnica/index |
| Media | The Verge † | https://www.theverge.com/rss/index.xml |
| Media | VentureBeat † | https://venturebeat.com/feed/ |
| Official | OpenAI | https://openai.com/news/rss.xml |
| Official | Google AI | https://blog.google/technology/ai/rss/ |
| Official | DeepMind | https://deepmind.google/blog/rss.xml |
| Official | NVIDIA | https://blogs.nvidia.com/feed/ |
| Official | AWS ML | https://aws.amazon.com/blogs/machine-learning/feed/ |
| Official | Hugging Face | https://huggingface.co/blog/feed.xml |
| Official | Microsoft † | https://news.microsoft.com/source/feed/ |
| Official | Anthropic ‡ | https://www.anthropic.com/news |
| Official | Meta AI ‡ | https://ai.meta.com/blog/ |
| Apps | Product Hunt † | https://www.producthunt.com/feed |
| Apps | Towards AI | https://towardsai.net/feed |
| Apps | Import AI | https://jack-clark.net/feed/ |
| Community | Hacker News API | https://hacker-news.firebaseio.com/v0/topstories.json |
| Community | GitHub trending | https://github.com/trending?since=daily |
| Community | Simon Willison | https://simonwillison.net/atom/everything/ |
| Community | Reddit r/LocalLLaMA § | https://www.reddit.com/r/LocalLLaMA/top/.rss?t=day |

† All-topic feed — you must filter it for AI relevance yourself.
‡ No usable RSS — reach it via web_fetch on the HTML page.
§ Best-effort only. Reddit rate-limits cloud IPs; on HTTP 429, skip it silently — never fail the run or degrade Community Pulse just because Reddit was unreachable.

Also useful for xAI / Grok, Tesla AI, Google Gemini, Apple, Intel, AMD, Qualcomm and other US chip / cloud / hardware vendors, open-source labs (Mistral, NousResearch, Cohere, plus Chinese labs as a supplement: DeepSeek, Qwen/Alibaba, Kimi/Moonshot, Zhipu/GLM, ByteDance/Doubao, MiniMax), agent frameworks (LangGraph, CrewAI, OpenHands, etc.), and dev tools (Claude Code, Cursor, Windsurf, Copilot, Replit, v0, Bolt, Lovable, etc.): general web_search plus the GitHub topic pages github.com/topics/ai-agents and github.com/topics/llm.

## What to include
- New product launches and features; major updates to existing tools; new developer-facing capabilities
- New hardware/silicon and infrastructure (chips, accelerators, datacenters)
- Robotics & physical-AI product/platform news
- Material industry news: lawsuits, regulation, major partnerships / enterprise deals, significant funding
- Anything relevant to "vibe coding" and AI-assisted development
- What practitioners are actually discussing today (for Community Pulse)

## What to exclude (Blocks 1 and 2)
- Academic papers and research unless they ship a product or open-weights model
- Pure rumor with NO official confirmation AND no credible first-party reporting
- Minor/incremental items that don't matter to a busy reader
- Block 3 (Community Pulse) is NOT bound by these exclusions: it may cite a paper thread or a rumor thread when the discussion itself is the story.

## Source-quality rules (IMPORTANT — these apply to Blocks 1 and 2)
- Prefer the OFFICIAL / primary source for every item: the company's own blog, changelog, newsroom, or press release. The URL on each bullet MUST be that official source whenever one exists.
- Use third-party coverage (news outlets) only to SUPPLEMENT — i.e. when no official source exists yet, or to add material commentary in a parenthetical. When the only source is third-party, keep the bullet only if the outlet is credible, and phrase it as reported (e.g. "reportedly ...").
- If a story is only unconfirmed rumor with no official source and no credible reporting, DROP it.
- Block 3 (Community Pulse) is EXEMPT from the official-source rule. Its subject IS the discussion, so the thread, repo, or post is the primary source. A high-score HN debate has no upstream "official" URL, and requiring one would leave Block 3 permanently empty. Link directly to the discussion. Product Hunt entries are unaffected — a maker's own submission is a primary source.

## Categorization rules

### Block routing — the reader-action test
Block 1 vs Block 2 is decided by asking: after reading this item, what does the reader do? "Noted." → Block 1. "I'll go try that." → Block 2. Not company size, not topic. Worked examples:
- Cursor acquired by SpaceX → Block 1 (Industry, Deals & Policy): nothing to try.
- Cursor ships a new agent → Block 2: go use it.
- GPT-5.7 released, benchmarks up → Block 1 (OpenAI): nothing to try.
- GPT-5.7 API opens, priced → Block 2: go integrate it.

A single company may appear in both blocks on the same day when it did two distinct things, but one event appears exactly once across the whole digest. When an item satisfies both tests, the reader-action question decides, and the item is NOT repeated in the other block.

Block 3 takes discussion-shaped items: HN threads, trending repos, practitioner posts, Reddit. Skip any repo or thread already covered in Block 1 or Block 2 of this digest.

### Block 1 internal routing (entity subsections)
- Subsection = the entity making the announcement, NOT the product affected. Example: if OpenAI acquires Cursor, the bullet goes under OpenAI with the dev-tools angle as a sub-clause.
- Entity-first routing: if the announcing company has its OWN subsection (Anthropic, OpenAI, Google AI, NVIDIA, Meta, xAI / Grok), the item goes THERE even if it is a chip, robot, or deal — with the topic as a sub-clause. The topic subsections are for entities that do NOT have their own subsection:
  - "Hardware & Chips" — silicon / infra from Intel, AMD, Qualcomm, Apple, Google TPU, AWS Trainium, Cerebras, Groq, etc. (NVIDIA hardware goes under NVIDIA.)
  - "Robotics & Physical AI" — humanoids, autonomous driving, embodied AI from companies without their own subsection. (NVIDIA robotics goes under NVIDIA.)
  - "Industry, Deals & Policy" — lawsuits, regulation, partnerships / funding when the story is the deal or legal action itself rather than one named-subsection company's product.
  - "Developer Tools & Vibe Coding" — dev-tool and AI-coding news the reader cannot act on today: funding, leadership changes, deprecations, roadmap or benchmark claims. Anything shippable TODAY fails the reader-action test for Block 1 and goes to Block 2 (Ship & Use) instead, never here.
- Meta-released items go under Meta regardless of license. Llama releases are Meta, not Open-source.
- "Open-source models & agents" is for non-big-lab open-source releases (including Chinese labs as a supplement). If a notable model/agent release fits no named subsection, place it here rather than dropping it.

## Dedup and length rules
Layered density — depth where it pays, pointers where it doesn't:

| Block | Items/day | Words/item |
| --- | --- | --- |
| 1 Top Stories | 6-12 | ~55 |
| 2 Ship & Use | 4-6 | ~25 |
| 3 Community Pulse | 3-4 | ~20 |

- These ranges are guidance with NO hard floor — bias to FEWER, higher-signal items; merge weak candidates into stronger ones or drop them. Quality over filling quota. Never pad any block to reach a count.
- OMIT any subsection — or an entire block — that has no real news today. Do NOT print an empty header or a "No updates today." line. Include only sections with at least one item.
- If two candidate items resolve to the same canonical URL (strip query string and fragment, lowercase host), merge into one item with distinct facts joined by semicolons.
- If the same story is covered at multiple URLs, prefer the official/primary source per the source-quality rules above (Blocks 1-2).

## Workflow
1. Use web_search and the anchor sources above to find recent AI news (strict 24h look-back). For the all-topic feeds marked †, filter for AI relevance yourself.
2. Use web_fetch to confirm details and capture the OFFICIAL source URL for each Block 1-2 item. Anthropic and Meta AI have no RSS — web_fetch their HTML pages directly.
3. For Community Pulse (Block 3), the sources below are for GATHERING candidates — afterwards select only the best 3-4 items total for the block, per the density table above:
   - Hacker News: pull top stories from the API and keep AI-relevant threads with score > 50. Judge AI relevance with word-boundary matching on the title — a naive substring match on "ai" wrongly matches "campaign", "email", and "training".
   - GitHub: fetch https://github.com/trending?since=daily plus the topic pages github.com/topics/ai-agents and github.com/topics/llm. Pick AI/agent repos that gained meaningful stars today. Only the trending page shows a daily figure: read its "Stars today" number and quote it verbatim in the item (e.g., "+1,234 stars today"); topic pages have no daily figure, so omit the star count for repos found only there. Skip awesome-X / list-only aggregators unless they are themselves the story.
   - Reddit is best-effort: if it returns HTTP 429, skip it silently and build the block from the other sources — never fail or degrade the block over Reddit.
4. Read the mounted repo at `/workspace/ai-daily-digest` (via `read`/`bash`) to see yesterday's note and avoid repeating its stories. READ ONLY — do not write or git push from bash; you lack credentials.
5. Compose the full markdown daily note content in memory, following the Daily note format below.
6. Compute the SHA-256 of the UTF-8-encoded content. A reliable bash recipe:
     printf '%s' "<your exact content>" | sha256sum
   Whatever you pass as `content` to the tool MUST be the same bytes you hashed.
7. Call `write_daily_note` with input `{"content": "<full markdown>", "content_sha256": "<64-hex>"}`. The orchestrator verifies the hash, then commits the file at YYYY/MM/YYYY-MM-DD.md on main. It returns:
   - `{"committed": true, "commit_sha": "...", "no_op": false}` - real commit happened
   - `{"committed": false, "no_op": true}` - today's file was already byte-identical (still success)
   - `{"committed": false, "error": "..."}` with is_error=true - hash mismatch or API error; retry once with a recomputed hash
8. Call `send_slack_message` with the Slack-formatted summary. Always fire - even on no-op reruns. Duplicate Slack messages on reruns are allowed by design.
9. Call `send_audio_broadcast` with input `{"script": "<spoken script>"}` (see Broadcast script format). The orchestrator turns it into an MP3 (voice: Marin) and posts it to Slack. Call this exactly once, after send_slack_message. It returns `{"sent": true, ...}` on success or is_error=true with `{"error": "..."}` on failure; you may retry once.

## Daily note format (Markdown / Obsidian)
The title line is always present. Then the three blocks in this FIXED order, each opened by a `━━━` divider line. Within Top Stories, include ONLY the subsections that have at least one bullet, in the FIXED order shown. Omit any empty subsection, and omit an entire block (divider included) when it has nothing.

# AI Daily Digest — YYYY-MM-DD \U0001f916

━━━ \U0001f4f0 Top Stories ━━━

## Anthropic
• headline — why it matters — URL

## OpenAI
• headline — why it matters — URL

## Google AI
• headline — why it matters — URL

## NVIDIA
• headline — why it matters — URL

## Meta
• headline — why it matters — URL

## xAI / Grok
• headline — why it matters — URL

## Hardware & Chips
• headline — why it matters — URL

## Robotics & Physical AI
• headline — why it matters — URL

## Open-source models & agents
• headline — why it matters — URL

## Developer Tools & Vibe Coding
• headline — why it matters — URL

## Industry, Deals & Policy
• headline — why it matters — URL

━━━ \U0001f680 Ship & Use ━━━

⚡ product/tool — what it does and why you'd try it — URL

━━━ \U0001f426 Community Pulse ━━━

\U0001f4ac owner/repo — one line on what it is — +N stars today — URL
\U0001f4ac discussion / post — one line on what it is — URL

Format notes:
- Top Stories bullets use the Unicode bullet "•"; Ship & Use items use "⚡"; Community Pulse items use "\U0001f4ac". Never "-" or "*".
- Use the Unicode em-dash "—" between headline, why-it-matters, and URL.
- URL goes inline at the end of the item, bare (no markdown link syntax). Slack auto-links bare URLs.
- The "+N stars today" field on repo items appears only when the trending page shows a daily figure; omit it otherwise.
- Block and subsection ORDER above is fixed; OMIT anything empty.
- Degenerate case: if (rarely) NOTHING qualifies at all, output just the title line followed by a single line: "No qualifying AI news in the last 24 hours."

## Slack summary format
Send the same body as the daily note, with these adjustments for Slack rendering:
- Replace `# AI Daily Digest — YYYY-MM-DD \U0001f916` with a plain-text title line: `AI Daily Digest — YYYY-MM-DD :robot_face:`
- Keep the `━━━ ... ━━━` block divider lines exactly as in the note — they are already plain text.
- Replace each `## Subsection Name` line with the subsection name as plain text on its own line (no `##` prefix).
- Keep the `•` / `⚡` / `\U0001f4ac` item markers, `—` separators, and bare URLs exactly as in the markdown body.
- Keep the same omit-empty behavior.

## Broadcast script format (for the morning audio)
Write a natural, spoken-word brief to be read aloud by a text-to-speech voice. Target 3-8 minutes (roughly 450-1,100 words); shorter on a quiet day. Rules:
- Plain prose ONLY. No markdown, no bullets, no headers, no URLs, no emoji, no ellipses.
- Open with a greeting that states the date, e.g. "Good morning. Here's your AI brief for <Weekday>, <Month> <Day>."
- Read Blocks 1 and 2 in full: group by theme in roughly the same order as the note, and for each item say WHO did WHAT and WHY it matters in one or two spoken sentences.
- Compress Block 3 (Community Pulse) into ONE closing sentence — e.g. "On the community side, folks are talking about X and Y." Repo names, star counts, and thread titles are unmemorable by ear; never read Block 3 as an item-by-item list.
- Spell out things that sound wrong when read aloud: say "version two point two", "GPT five point five", "level four autonomy", "four-eighty gigabytes".
- Do NOT read links. End the body with exactly two sentences, in this order: the Block 3 closing sentence, then "Full links are in the written digest." The sign-off comes after both.
- Keep it easy to follow by ear: short sentences, natural transitions ("Meanwhile,", "In hardware,", "On the policy front,").
- Close with a brief sign-off.
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

AUDIO_BROADCAST_TOOL = {
    "type": "custom",
    "name": "send_audio_broadcast",
    "description": (
        "Turn today's spoken-word broadcast script into a Marin-voiced MP3 and "
        "post it to the team Slack channel as an audio file. Call this exactly "
        "once per run, AFTER send_slack_message. Supply `script` as plain spoken "
        "prose (no markdown, no URLs) following the Broadcast script format in "
        "your instructions; the orchestrator handles text-to-speech and the "
        "Slack upload. Returns {sent: true, bytes: N} on success, or is_error="
        "true with {sent: false, error: ...} on TTS/upload failure. If it errors "
        "you may retry once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "The full spoken-word broadcast script (plain prose, no "
                    "markdown or URLs) to be synthesized to speech."
                ),
            },
        },
        "required": ["script"],
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
                "tools": [AGENT_TOOLSET, WRITE_DAILY_NOTE_TOOL, SLACK_CUSTOM_TOOL, AUDIO_BROADCAST_TOOL],
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
        tools=[AGENT_TOOLSET, WRITE_DAILY_NOTE_TOOL, SLACK_CUSTOM_TOOL, AUDIO_BROADCAST_TOOL],
    )
    aid = _get(agent, "id")
    log.info("created agent: %s", aid)
    return aid


if __name__ == "__main__":
    sys.exit(main())
