"""Runtime orchestrator for the AI Daily Digest Managed Agent.

Drives a single session:
  1. Create session with the Obsidian vault mounted as a github_repository resource.
  2. Stream events; handle send_slack_message custom tool; handle idle turns.
  3. After the agent finishes, verify today's note is on main and matches the
     agent's claimed CONTENT_SHA256 hash. Any mismatch -> SystemExit(1).
  4. Always archive the session in finally.

Env vars required (set by the GitHub Actions workflow):
    ANTHROPIC_API_KEY, AGENT_ID, ENVIRONMENT_ID, GIT_PAT, SLACK_WEBHOOK_URL
"""

import base64
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import anthropic

REPO_OWNER = "ZH-L1N"
REPO_NAME = "ai-daily-digest"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
MOUNT_PATH = "/workspace/ai-daily-digest"
BRANCH = "main"

GITHUB_API = "https://api.github.com"
USER_AGENT = "ai-digest-orchestrator"
HTTP_TIMEOUT = 10.0

CONTENT_SHA256_RE = re.compile(
    r"^CONTENT_SHA256:\s+([0-9a-fA-F]{64})\s*$", re.MULTILINE
)

log = logging.getLogger("run")


# --- Shape-tolerant helper (shared with setup.py) ---------------------------


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --- HTTP helpers -----------------------------------------------------------


def _gh_headers(pat):
    return {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }


def _http_get_json(url, headers):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            status = r.status
            body = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, (e.read().decode("utf-8") if e.fp else "")
    return status, body


def github_get_contents(pat, path):
    """Return (sha, bytes) for path on main, or (None, None) if 404."""
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}?ref={BRANCH}"
    status, body = _http_get_json(url, _gh_headers(pat))
    if status == 404:
        return None, None
    if not (200 <= status < 300):
        raise RuntimeError(f"GitHub contents GET failed: {status} {body[:300]}")
    data = json.loads(body)
    sha = _get(data, "sha")
    b64 = _get(data, "content") or ""
    # GitHub returns content base64-encoded with embedded newlines
    raw = base64.b64decode(b64) if b64 else b""
    return sha, raw


def github_latest_commit_utc(pat, path):
    """Return tz-aware UTC datetime of the most recent commit touching path."""
    url = (
        f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/commits"
        f"?path={path}&per_page=1&sha={BRANCH}"
    )
    status, body = _http_get_json(url, _gh_headers(pat))
    if not (200 <= status < 300):
        raise RuntimeError(f"GitHub commits GET failed: {status} {body[:300]}")
    commits = json.loads(body)
    if not commits:
        raise RuntimeError(f"no commits found for path {path!r}")
    iso = commits[0]["commit"]["committer"]["date"].replace("Z", "+00:00")
    return datetime.fromisoformat(iso)


def post_slack(webhook_url, summary):
    """POST with 2 retries (2s/4s backoff). Returns (ok, detail)."""
    payload = json.dumps({"text": summary}).encode("utf-8")
    delays = [0, 2.0, 4.0]  # first attempt, then two backoff retries
    last_detail = "unknown"
    for i, delay in enumerate(delays):
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                if 200 <= r.status < 300:
                    return True, f"status={r.status}"
                last_detail = f"http {r.status}"
        except urllib.error.HTTPError as e:
            last_detail = f"http {e.code}"
        except Exception as e:  # URLError, timeout, etc.
            last_detail = f"{type(e).__name__}: {e}"
        log.warning("Slack attempt %d/%d failed: %s", i + 1, len(delays), last_detail)
    return False, last_detail


# --- Event processing -------------------------------------------------------


def _event_type(event):
    return _get(event, "type")


def send_custom_tool_result(client, session_id, tool_use_id, text, is_error):
    try:
        client.beta.sessions.events.send(
            session_id,
            events=[
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}],
                    "is_error": is_error,
                }
            ],
        )
    except Exception as e:
        log.error("failed to POST custom_tool_result id=%s: %s", tool_use_id, e)
        raise SystemExit(1)


def backfill_custom_tool_uses(client, session_id, known):
    """Paginate events.list(session_id), adding any custom_tool_use IDs to known."""
    cursor = None
    while True:
        kwargs = {"limit": 200}
        if cursor is not None:
            kwargs["after_id"] = cursor
        page = client.beta.sessions.events.list(session_id, **kwargs)
        data = _get(page, "data") or []
        for ev in data:
            if _event_type(ev) == "agent.custom_tool_use":
                eid = _get(ev, "id")
                if eid is not None:
                    known.add(eid)
        if not _get(page, "has_more", False) or not data:
            return
        last_id = _get(data[-1], "id")
        if last_id is None:
            return
        cursor = last_id


def extract_content_hash(all_text):
    matches = CONTENT_SHA256_RE.findall(all_text)
    return matches[-1].lower() if matches else None


def run_session(client, *, agent_id, env_id, git_pat, slack_webhook_url):
    # Compute today's path once, in ET, before any session work.
    now_et = datetime.now(ZoneInfo("America/New_York"))
    date_str = now_et.strftime("%Y-%m-%d")
    path = f"{now_et.strftime('%Y')}/{now_et.strftime('%m')}/{date_str}.md"
    run_start_utc = datetime.now(timezone.utc)
    log.info("target path=%s run_start_utc=%s", path, run_start_utc.isoformat())

    before_sha, _ = github_get_contents(git_pat, path)
    log.info("before_sha=%s", before_sha)

    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        resources=[
            {
                "type": "github_repository",
                "url": REPO_URL,
                "mount_path": MOUNT_PATH,
                "authorization_token": git_pat,
                "checkout": {"type": "branch", "name": BRANCH},
            }
        ],
    )
    session_id = _get(session, "id")
    log.info("created session %s", session_id)

    try:
        return _drive_session(
            client,
            session_id=session_id,
            path=path,
            date_str=date_str,
            before_sha=before_sha,
            run_start_utc=run_start_utc,
            git_pat=git_pat,
            slack_webhook_url=slack_webhook_url,
        )
    finally:
        try:
            client.beta.sessions.archive(session_id)
            log.info("archived session %s", session_id)
        except Exception as e:
            log.warning("session archive failed (ignored): %s", e)


def _drive_session(
    client,
    *,
    session_id,
    path,
    date_str,
    before_sha,
    run_start_utc,
    git_pat,
    slack_webhook_url,
):
    slack_sent = False
    slack_error = None
    pending_custom_tool_uses = set()
    answered_custom_tool_uses = set()
    mcp_errors_seen = []
    agent_message_texts = []

    kickoff = (
        f"Today's date (America/New_York): {date_str}\n"
        f"Target file path (must use exactly this path): {path}\n"
        f"Branch: {BRANCH}\n"
        f"Fetch the last 24 hours of AI news, write today's digest to {path} in the "
        "Obsidian vault via the GitHub MCP tools, and send the Slack summary. "
        "End your final message with CONTENT_SHA256: <sha256 hex of the file's UTF-8 bytes>."
    )

    with client.beta.sessions.events.stream(session_id) as stream:
        client.beta.sessions.events.send(
            session_id,
            events=[
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": kickoff}],
                }
            ],
        )

        for event in stream:
            et = _event_type(event)
            if et == "agent.message":
                text = _extract_message_text(event)
                if text:
                    agent_message_texts.append(text)
                    log.info("agent.message: %s", text[:200].replace("\n", " "))
            elif et == "agent.mcp_tool_result":
                if _get(event, "is_error"):
                    mcp_errors_seen.append(
                        {
                            "tool_use_id": _get(event, "tool_use_id"),
                            "content": _get(event, "content"),
                        }
                    )
                    log.warning("mcp tool result is_error=true: %s", event)
            elif et == "session.error":
                log.error("session.error: %s", event)
                raise SystemExit(1)
            elif et == "agent.custom_tool_use":
                tool_name = _get(event, "name")
                tool_use_id = _get(event, "id")
                if tool_name != "send_slack_message":
                    log.warning(
                        "unexpected custom_tool_use name=%r id=%s", tool_name, tool_use_id
                    )
                    send_custom_tool_result(
                        client,
                        session_id,
                        tool_use_id,
                        f"unknown tool {tool_name!r}",
                        True,
                    )
                    pending_custom_tool_uses.discard(tool_use_id)
                    answered_custom_tool_uses.add(tool_use_id)
                    continue

                pending_custom_tool_uses.add(tool_use_id)
                summary = _extract_tool_input(event).get("summary", "")
                if not summary:
                    log.warning("send_slack_message called with empty summary")

                ok, detail = post_slack(slack_webhook_url, summary or "(empty summary)")
                if ok:
                    slack_sent = True
                    send_custom_tool_result(
                        client, session_id, tool_use_id, "sent", False
                    )
                else:
                    slack_error = detail
                    send_custom_tool_result(
                        client,
                        session_id,
                        tool_use_id,
                        f"slack post failed: {detail}",
                        True,
                    )
                pending_custom_tool_uses.discard(tool_use_id)
                answered_custom_tool_uses.add(tool_use_id)

            elif et == "session.status_idle":
                if _handle_idle(
                    client,
                    session_id,
                    event,
                    pending_custom_tool_uses,
                    answered_custom_tool_uses,
                ):
                    break  # end_turn
            elif et == "session.status_terminated":
                log.error("session.status_terminated: %s", event)
                raise SystemExit(1)

    # --- Post-loop verification ---------------------------------------------

    if not slack_sent:
        log.error("slack was never sent successfully; last error=%s", slack_error)
        raise SystemExit(1)

    all_text = "\n".join(agent_message_texts)
    agent_claimed_hash = extract_content_hash(all_text)
    if agent_claimed_hash is None:
        log.error("agent did not emit CONTENT_SHA256 line; contract broken")
        raise SystemExit(1)

    after_sha, observed_bytes = github_get_contents(git_pat, path)
    if after_sha is None:
        log.error(
            "post-run: file %s does not exist on main; mcp_errors_seen=%s",
            path,
            mcp_errors_seen,
        )
        raise SystemExit(1)

    observed_hash = hashlib.sha256(observed_bytes).hexdigest()
    latest_commit_utc = github_latest_commit_utc(git_pat, path)

    rule2 = observed_hash == agent_claimed_hash
    rule3 = (latest_commit_utc >= run_start_utc) or (
        before_sha is not None and before_sha == after_sha
    )

    if not (rule2 and rule3):
        log.error(
            "verification failed: "
            "agent_claimed_hash=%s observed_hash=%s "
            "before_sha=%s after_sha=%s "
            "latest_commit_utc=%s run_start_utc=%s "
            "mcp_errors_seen=%s",
            agent_claimed_hash,
            observed_hash,
            before_sha,
            after_sha,
            latest_commit_utc.isoformat(),
            run_start_utc.isoformat(),
            mcp_errors_seen,
        )
        raise SystemExit(1)

    log.info(
        "success: hash_match=%s committed_this_run=%s unchanged_noop=%s",
        rule2,
        latest_commit_utc >= run_start_utc,
        before_sha is not None and before_sha == after_sha,
    )


def _extract_message_text(event):
    content = _get(event, "content") or []
    chunks = []
    for block in content:
        if _get(block, "type") == "text":
            t = _get(block, "text")
            if t:
                chunks.append(t)
    return "\n".join(chunks)


def _extract_tool_input(event):
    input_data = _get(event, "input")
    if isinstance(input_data, dict):
        return input_data
    if isinstance(input_data, str):
        try:
            return json.loads(input_data)
        except Exception:
            return {}
    return {}


def _handle_idle(
    client, session_id, event, pending_custom_tool_uses, answered_custom_tool_uses
):
    """Return True if the session reached end_turn; else continue the loop.

    Raises SystemExit(1) on non-recoverable stop reasons.
    """
    stop_reason = _get(event, "stop_reason")
    reason_type = (
        stop_reason if isinstance(stop_reason, str) else _get(stop_reason, "type")
    )
    log.info("session.status_idle stop_reason=%s", reason_type)

    if reason_type == "end_turn":
        return True

    if reason_type != "requires_action":
        log.error("unexpected stop_reason=%s event=%s", reason_type, event)
        raise SystemExit(1)

    blocking = (
        _get(_get(stop_reason, "requires_action"), "event_ids")
        or _get(stop_reason, "event_ids")
        or []
    )
    blocking = set(blocking)

    if not blocking:
        log.error("requires_action with empty blocking set; raw stop_reason=%s", stop_reason)
        raise SystemExit(1)

    known = pending_custom_tool_uses | answered_custom_tool_uses
    if blocking.issubset(known):
        return False  # we handled / are handling these

    # Paginate events to backfill any custom_tool_use we haven't seen yet.
    backfill_custom_tool_uses(client, session_id, pending_custom_tool_uses)
    known = pending_custom_tool_uses | answered_custom_tool_uses
    if blocking.issubset(known):
        return False

    unknown = blocking - known
    log.error(
        "requires_action has unresolvable blocking ids %s (not custom tool uses); "
        "always_allow policy did not take effect",
        unknown,
    )
    raise SystemExit(1)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    required = ["AGENT_ID", "ENVIRONMENT_ID", "GIT_PAT", "SLACK_WEBHOOK_URL"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        log.error("missing env vars: %s", missing)
        return 2
    # ANTHROPIC_API_KEY picked up implicitly by the anthropic client.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("missing env var: ANTHROPIC_API_KEY")
        return 2

    client = anthropic.Anthropic()
    run_session(
        client,
        agent_id=os.environ["AGENT_ID"],
        env_id=os.environ["ENVIRONMENT_ID"],
        git_pat=os.environ["GIT_PAT"],
        slack_webhook_url=os.environ["SLACK_WEBHOOK_URL"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
