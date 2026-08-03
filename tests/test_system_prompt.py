"""Guard tests for setup.SYSTEM_PROMPT (no pytest dependency, no network).

The prompt is a long string with no other coverage; a truncating edit would
ship silently and only surface as a bad digest the next morning. Two groups:

- Structure: block headers, the 24 anchor URLs, absence of removed/dead
  sources, and a length sanity floor.
- Behavioral contracts: one load-bearing exact phrase per rule, so the test
  fails on rule REMOVAL rather than on harmless rewording elsewhere.

Run: python tests/test_system_prompt.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup import SYSTEM_PROMPT

# ---------------------------------------------------------------- structure

BLOCK_HEADERS = [
    "Top Stories",
    "Ship & Use",
    "Community Pulse",
]

# The literal divider lines the note format emits — anchored to the output
# template, not to prose mentions of the block names.
BLOCK_DIVIDERS = [
    "━━━ \U0001f4f0 Top Stories ━━━",
    "━━━ \U0001f680 Ship & Use ━━━",
    "━━━ \U0001f426 Community Pulse ━━━",
]

# The 24-entry anchor keep-list from docs/superpowers/specs/v0.2.0-digest-content-spec.md.
ANCHOR_URLS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://the-decoder.com/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://www.theverge.com/rss/index.xml",
    "https://venturebeat.com/feed/",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://deepmind.google/blog/rss.xml",
    "https://blogs.nvidia.com/feed/",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://huggingface.co/blog/feed.xml",
    "https://news.microsoft.com/source/feed/",
    "https://www.anthropic.com/news",
    "https://ai.meta.com/blog/",
    "https://www.producthunt.com/feed",
    "https://towardsai.net/feed",
    "https://jack-clark.net/feed/",
    "https://hacker-news.firebaseio.com/v0/topstories.json",
    "https://github.com/trending?since=daily",
    "https://simonwillison.net/atom/everything/",
    "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=day",
]

# Sources verified dead 2026-08-02 and removed from the prompt; none may creep
# back. Also guards the standalone "## Trending GitHub" section deleted in
# v0.2.0 (folded into Community Pulse).
REMOVED_SOURCE_STRINGS = [
    "nitter",
    "rsshub",
    "xcancel",
    "bensbites",
    "therundown",
    "aibusiness",
    "cat=533",
    "theverge.com/ai-artificial-intelligence",
    "producthunt.com/topics",
    "anthropic.com/rss",
    "ai.meta.com/blog/rss",
    "## trending github",
]


def test_three_block_headers_present():
    for header in BLOCK_HEADERS:
        assert header in SYSTEM_PROMPT, f"missing block header: {header}"


def test_note_format_emits_block_dividers_and_markers():
    for divider in BLOCK_DIVIDERS:
        assert divider in SYSTEM_PROMPT, f"missing note-format divider: {divider}"
    # Per-block item markers in the note template.
    assert "⚡ product/tool" in SYSTEM_PROMPT
    assert "\U0001f4ac owner/repo" in SYSTEM_PROMPT
    assert "\U0001f4ac discussion / post" in SYSTEM_PROMPT
    # Slack keeps the dividers verbatim.
    assert "block divider lines exactly as in the note" in SYSTEM_PROMPT


def test_all_anchor_urls_present():
    assert len(ANCHOR_URLS) == 24
    for url in ANCHOR_URLS:
        assert url in SYSTEM_PROMPT, f"missing anchor URL: {url}"


def test_no_removed_source_appears():
    lowered = SYSTEM_PROMPT.lower()
    for dead in REMOVED_SOURCE_STRINGS:
        assert dead not in lowered, f"removed/dead source reappeared: {dead}"


def test_length_sanity():
    # Guards against a truncating edit shipping silently.
    assert len(SYSTEM_PROMPT) > 6000


# ------------------------------------------------------ behavioral contracts


def test_layered_density_ranges():
    # The full density-table rows, so swapping ranges between blocks fails too.
    assert "| 1 Top Stories | 6-12 | ~55 |" in SYSTEM_PROMPT
    assert "| 2 Ship & Use | 4-6 | ~25 |" in SYSTEM_PROMPT
    assert "| 3 Community Pulse | 3-4 | ~20 |" in SYSTEM_PROMPT


def test_no_padding_rule():
    assert "bias to fewer" in SYSTEM_PROMPT.lower()


def test_cross_block_dedup_rule():
    assert "one event appears exactly once" in SYSTEM_PROMPT


def test_reader_action_routing_test():
    assert "what does the reader do" in SYSTEM_PROMPT


def test_block3_official_source_exemption():
    # Case-insensitive and hyphen-free so emphasis rewording stays green.
    assert "exempt from the official" in SYSTEM_PROMPT.lower()


def test_broadcast_compresses_block3_to_one_sentence():
    assert "one closing sentence" in SYSTEM_PROMPT.lower()


def test_hn_word_boundary_and_score_threshold():
    assert "word-boundary matching" in SYSTEM_PROMPT
    assert "score > 50" in SYSTEM_PROMPT


def test_reddit_429_skips_silently():
    assert "429" in SYSTEM_PROMPT
    assert "skip it silently" in SYSTEM_PROMPT


def test_english_only_output():
    assert "english only" in SYSTEM_PROMPT.lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all system-prompt guard tests passed")
