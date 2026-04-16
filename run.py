"""Runtime orchestrator for the AI Daily Digest Managed Agent.

Drives a single session:
  1. Create session with the Obsidian vault mounted as a github_repository
     resource (read-only; writes go through write_daily_note tool below).
  2. Stream events; handle write_daily_note and send_slack_message custom tools;
     handle idle turns.
  3. On write_daily_note: verify the agent's claimed content SHA-256, then
     commit the file to main via the GitHub REST API (orchestrator-side write).
  4. After the agent finishes, verify the file on main matches the last
     content_sha256 the agent supplied. Any mismatch -> SystemExit(1).
  5. Always archive the session in finally.

Env vars required (set by the GitHub Actions workflow):
    ANTHROPIC_API_KEY, AGENT_ID, ENVIRONMENT_ID, GIT_PAT, SLACK_WEBHOOK_URL
"""

import base64
import hashlib
import json
import logging
import os
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
HTTP_READ_TIMEOUT = 10.0
HTTP_WRITE_TIMEOUT = 30.0

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


def _http_request(url, headers, method="GET", data=None, timeout=HTTP_READ_TIMEOUT):
    req = urllib.request.Request(url, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, (e.read().decode("utf-8") if e.fp else "")


def github_get_contents(pat, path):
    """Return (sha, bytes) for path on main, or (None, None) if 404."""
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}?ref={BRANCH}"
    status, body = _http_request(url, _gh_headers(pat))
    if status == 404:
        return None, None
    if not (200 <= status < 300):
        raise RuntimeError(f"GitHub contents GET failed: {status} {body[:300]}")
    data = json.loads(body)
    sha = _get(data, "sha")
    b64 = _get(data, "content") or ""
    raw = base64.b64decode(b64) if b64 else b""
    return sha, raw


def github_put_contents(pat, path, content_bytes, commit_message, sha=None):
    """Create or update a file on main via contents API.

    Returns parsed response JSON on 2xx, or raises RuntimeError with detail.
    """
    url = f"{GITHUB_API}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path}"
    body = {
        "message": commit_message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": BRANCH,
    }
    if sha is not None:
        body["sha"] = sha
    headers = _gh_headers(pat)
    headers["Content-Type"] = "application/json"
    status, resp_body = _http_request(
        url,
        headers,
        method="PUT",
        data=json.dumps(body).encode("utf-8"),
        timeout=HTTP_WRITE_TIMEOUT,
    )
    if not (200 <= status < 300):
        raise RuntimeError(f"GitHub contents PUT failed: {status} {resp_body[:500]}")
    return json.loads(resp_body)


def _webhook_tag(url):
    """Short non-secret identifier for logging (last 6 chars of path)."""
    return "…" + url[-6:] if len(url) > 6 else url


def _post_slack_single(webhook_url, summary):
    """POST to one webhook with 2 retries (2s/4s backoff). Returns (ok, detail)."""
    payload = json.dumps({"text": summary}).encode("utf-8")
    delays = [0, 2.0, 4.0]
    last_detail = "unknown"
    tag = _webhook_tag(webhook_url)
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
            with urllib.request.urlopen(req, timeout=HTTP_READ_TIMEOUT) as r:
                if 200 <= r.status < 300:
                    return True, f"status={r.status}"
                last_detail = f"http {r.status}"
        except urllib.error.HTTPError as e:
            last_detail = f"http {e.code}"
        except Exception as e:  # URLError, timeout, etc.
            last_detail = f"{type(e).__name__}: {e}"
        log.warning(
            "Slack %s attempt %d/%d failed: %s", tag, i + 1, len(delays), last_detail
        )
    return False, last_detail


def post_slack(webhook_spec, summary):
    """POST to one or more Slack webhook URLs.

    `webhook_spec` may be a single URL or a comma-separated list of URLs.
    Strict: returns (True, ...) only if ALL destinations succeeded.
    """
    urls = [u.strip() for u in webhook_spec.split(",") if u.strip()]
    if not urls:
        return False, "no webhook URLs configured"

    all_ok = True
    details = []
    for url in urls:
        ok, detail = _post_slack_single(url, summary)
        details.append(f"{_webhook_tag(url)}={detail}")
        if not ok:
            all_ok = False

    log.info(
        "Slack posted to %d destination(s): %s [%s]",
        len(urls),
        "all ok" if all_ok else "some failed",
        "; ".join(details),
    )
    return all_ok, "; ".join(details)


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


def handle_write_daily_note(git_pat, path, date_str, tool_input, before_sha):
    """Verify hash, commit via GitHub contents API. Returns (result_dict, is_error, claimed_sha)."""
    content = tool_input.get("content")
    claimed = (tool_input.get("content_sha256") or "").lower()

    if not isinstance(content, str):
        return {"committed": False, "error": "content must be a string"}, True, None
    if not isinstance(claimed, str) or len(claimed) != 64:
        return (
            {"committed": False, "error": "content_sha256 must be 64 hex chars"},
            True,
            None,
        )

    content_bytes = content.encode("utf-8")
    computed = hashlib.sha256(content_bytes).hexdigest()
    if computed != claimed:
        return (
            {
                "committed": False,
                "error": f"hash mismatch: claimed={claimed} computed={computed}",
            },
            True,
            None,
        )

    # Re-fetch current state at call time (before_sha was captured at run start;
    # the file may have changed if this is a same-session retry after a no-op).
    current_sha, current_bytes = github_get_contents(git_pat, path)
    if current_sha is not None and current_bytes == content_bytes:
        log.info("write_daily_note: content byte-identical to main; no-op")
        return {"committed": False, "no_op": True}, False, claimed

    commit_message = f"Daily digest {date_str}"
    try:
        resp = github_put_contents(
            git_pat,
            path,
            content_bytes,
            commit_message,
            sha=current_sha,
        )
    except Exception as e:
        log.error("github_put_contents failed: %s", e)
        return {"committed": False, "error": str(e)}, True, None

    commit_sha = _get(_get(resp, "commit") or {}, "sha")
    log.info("write_daily_note: committed %s commit_sha=%s", path, commit_sha)
    return (
        {"committed": True, "commit_sha": commit_sha, "no_op": False},
        False,
        claimed,
    )


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
    note_committed = False
    last_claimed_sha = None
    pending_custom_tool_uses = set()
    answered_custom_tool_uses = set()

    kickoff = (
        f"Today's date (America/New_York): {date_str}\n"
        f"Target file path (must use exactly this path): {path}\n"
        f"Branch: {BRANCH}\n"
        f"Fetch the last 24 hours of AI news, compose today's digest markdown, "
        f"call write_daily_note with the full content and its SHA-256 hash "
        f"(the orchestrator will commit it to {path} on main for you), then "
        f"call send_slack_message with a short bullet summary. "
        f"Reminder: do NOT try to git push or write to the mount from bash — "
        f"you don't have credentials and it will fail. The only write path is "
        f"the write_daily_note tool."
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
                    log.info("agent.message: %s", text[:200].replace("\n", " "))
            elif et == "session.error":
                log.error("session.error: %s", event)
                raise SystemExit(1)
            elif et == "agent.custom_tool_use":
                tool_name = _get(event, "name")
                tool_use_id = _get(event, "id")
                pending_custom_tool_uses.add(tool_use_id)
                tool_input = _extract_tool_input(event)

                if tool_name == "write_daily_note":
                    result, is_error, claimed = handle_write_daily_note(
                        git_pat, path, date_str, tool_input, before_sha
                    )
                    if not is_error:
                        note_committed = True
                        last_claimed_sha = claimed
                    send_custom_tool_result(
                        client,
                        session_id,
                        tool_use_id,
                        json.dumps(result),
                        is_error,
                    )

                elif tool_name == "send_slack_message":
                    summary = tool_input.get("summary", "")
                    if not summary:
                        log.warning("send_slack_message called with empty summary")
                    ok, detail = post_slack(
                        slack_webhook_url, summary or "(empty summary)"
                    )
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

                else:
                    log.warning(
                        "unexpected custom_tool_use name=%r id=%s",
                        tool_name,
                        tool_use_id,
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

    if not note_committed:
        log.error("write_daily_note never succeeded; note was not committed")
        raise SystemExit(1)
    if not slack_sent:
        log.error("slack was never sent successfully; last error=%s", slack_error)
        raise SystemExit(1)
    if last_claimed_sha is None:
        log.error("internal: note_committed=True but last_claimed_sha is None")
        raise SystemExit(1)

    # Belt-and-suspenders: confirm what's on main matches what we thought we wrote.
    after_sha, observed_bytes = github_get_contents(git_pat, path)
    if after_sha is None:
        log.error("post-run: file %s unexpectedly missing on main", path)
        raise SystemExit(1)
    observed_hash = hashlib.sha256(observed_bytes).hexdigest()
    if observed_hash != last_claimed_sha:
        log.error(
            "post-run hash mismatch: observed=%s expected=%s path=%s after_sha=%s",
            observed_hash,
            last_claimed_sha,
            path,
            after_sha,
        )
        raise SystemExit(1)

    log.info(
        "success: path=%s after_sha=%s content_sha256=%s",
        path,
        after_sha,
        observed_hash,
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
