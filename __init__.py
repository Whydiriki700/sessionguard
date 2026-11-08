"""SessionGuard - work survives the session.

Hermes can resume a conversation with /resume, but a transcript is not
work state: it does not say which steps already changed the outside
world, and after a gateway death or a compaction the next session
starts blank or lossy and the agent has to guess where it was -
redoing side effects or dropping the task entirely.

SessionGuard keeps a local work-state checkpoint per session, built
from tool evidence, not from prose:

  - `pre_llm_call` records the current task text each turn; on the
    first turn of a session it briefs the agent twice over: how to
    checkpoint its own progress with the `session_checkpoint` tool,
    and - when a previous session left an unclosed checkpoint - a
    recovery briefing (task, files touched, last tool calls and
    outcomes, failures, the previous agent's own notes).
  - `post_tool_call` records every tool call (short redacted summary),
    the files it touched, and failures.
  - `on_session_end` (which Hermes fires per turn) records each
    turn's outcome on the checkpoint. A checkpoint closes only when
    the agent marks its task done via `session_checkpoint`; every
    other session's checkpoint stays open, and the most recent open
    one from the same origin (same platform and sender) is what the
    next session gets briefed about. Subagent and cron sessions
    neither record checkpoints nor receive briefings.

Everything is local under <HERMES_HOME>/sessionguard/ (state.json plus
an append-only events.jsonl audit trail). No network, no LLM. Storage
errors never break the agent loop: every hook degrades to a no-op.

Setting (config: plugins.entries.sessionguard.settings.enabled):
  - true (default): checkpointing and recovery briefings active
  - false: plugin observes nothing and injects nothing
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

try:
    from . import store
except ImportError:  # loaded as a top-level module in tests
    import store  # type: ignore

logger = logging.getLogger(__name__)

_CTX: Any = None

_STANDING_BRIEF = (
    "SessionGuard is active in this profile. As you work, keep a work-state "
    "checkpoint with the session_checkpoint tool: call it after meaningful "
    "steps with a short note (what is done, what is next), and call it with "
    "status='done' when the task is fully complete. If this process dies or "
    "the session is compacted, the next session will be briefed from that "
    "checkpoint instead of starting from zero."
)


def _enabled() -> bool:
    if _CTX is not None:
        try:
            value = _CTX.get_config("enabled", True)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() not in {"false", "0", "no", "off"}
        except Exception:
            pass
    return True


def _ago(ts: float) -> str:
    delta = max(0, int(time.time() - ts))
    if delta < 90:
        return f"{delta}s ago"
    if delta < 5400:
        return f"{delta // 60}m ago"
    if delta < 172800:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _recovery_brief(checkpoint: Dict[str, Any]) -> str:
    lines = [
        "SessionGuard recovery briefing: a previous session ended without "
        "closing its work checkpoint, so its task may be unfinished.",
        f"Previous session: {checkpoint.get('session_id')} "
        f"(platform {checkpoint.get('platform') or '-'}, last active {_ago(checkpoint.get('last_seen') or 0)}, "
        f"turns {checkpoint.get('turns') or 0}, failed tool calls {checkpoint.get('failures') or 0}).",
    ]
    if checkpoint.get("task"):
        lines.append(f"Its latest task was: {checkpoint['task']}")
    notes = checkpoint.get("notes") or []
    if notes:
        lines.append("Its own progress notes (latest last):")
        for note in notes[-3:]:
            lines.append(f"- {note.get('note')}")
    files = checkpoint.get("files_touched") or []
    if files:
        lines.append("Files it touched: " + ", ".join(files[-10:]))
    events = checkpoint.get("tool_events") or []
    if events:
        lines.append("Its last tool calls (latest last):")
        for event in events[-5:]:
            lines.append(f"- {event.get('tool')}: {event.get('summary')} [{event.get('status') or '?'}]")
    lines.append(
        "Before redoing any step, verify its real-world state first (files, "
        "messages, records): some steps may already be done. Continue the "
        "work if the user wants it continued, and update the checkpoint as "
        "you go."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def on_pre_llm_call(
    is_first_turn: bool = False,
    user_message: str = "",
    session_id: str = "",
    platform: str = "",
    **kwargs: Any,
) -> Optional[Dict[str, str]]:
    if not _enabled() or not session_id:
        return None
    sender_id = str(kwargs.get("sender_id") or "")
    if kwargs.get("parent_session_id") or (platform or "") == "cron":
        # Subagent and cron sessions neither record checkpoints nor
        # receive briefings: their work is not a user's open thread, and
        # on multi-user gateway profiles it must never leak across users.
        return None
    try:
        store.record_turn(session_id, user_message or "", platform or "", sender_id)
    except Exception:
        logger.debug("SessionGuard: record_turn failed", exc_info=True)
        return None
    if not is_first_turn:
        return None
    parts = [_STANDING_BRIEF]
    try:
        stale = store.latest_open_checkpoint(
            exclude_session=session_id, platform=platform or "", sender_id=sender_id
        )
        if stale is not None and not stale.get("briefed"):
            parts.append(_recovery_brief(stale))
            store.mark_briefed(str(stale.get("session_id")))
            store.append_event(
                {
                    "ts": time.time(),
                    "event": "recovery_briefing_injected",
                    "from_session": stale.get("session_id"),
                    "into_session": session_id,
                }
            )
    except Exception:
        logger.debug("SessionGuard: recovery brief failed", exc_info=True)
    return {"context": "\n\n".join(parts)}


def on_post_tool_call(tool_name: str, args: Any = None, result: Any = None, **kwargs: Any) -> None:
    if not _enabled():
        return
    session_id = str(kwargs.get("session_id") or "")
    if not session_id:
        return
    try:
        store.record_tool(session_id, tool_name, args, kwargs.get("status"))
    except Exception:
        logger.debug("SessionGuard: record_tool failed", exc_info=True)


def on_session_end(session_id: str = "", **kwargs: Any) -> None:
    # In Hermes this hook fires at the end of every turn, not when a
    # session's work is done, so it only records the turn outcome. A
    # checkpoint closes when the agent marks the task done through the
    # session_checkpoint tool; anything else stays open and brief-worthy.
    if not _enabled() or not session_id:
        return
    try:
        if kwargs.get("interrupted"):
            outcome = "interrupted"
        elif kwargs.get("failed"):
            outcome = "failed"
        elif kwargs.get("completed"):
            outcome = "completed"
        else:
            outcome = "ended"
        store.record_turn_outcome(session_id, outcome)
        store.append_event(
            {"ts": time.time(), "event": f"turn_{outcome}", "session_id": session_id}
        )
    except Exception:
        logger.debug("SessionGuard: session_end handling failed", exc_info=True)


# ---------------------------------------------------------------------------
# User surfaces: session_checkpoint tool, /guard command, CLI
# ---------------------------------------------------------------------------


def session_checkpoint_tool(params: Dict[str, Any] | None = None, **kwargs: Any) -> str:
    params = params or {}
    note = str(params.get("note") or "")
    status = str(params.get("status") or "working").lower()
    # Only the real session_id kwarg Hermes passes: a session_id inside
    # the model-supplied params must never let the model write or close
    # another session's checkpoint.
    session_id = str(kwargs.get("session_id") or "")
    if not session_id:
        return json.dumps({"success": False, "error": "no session_id available"})
    checkpoint = store.add_note(session_id, note, close=(status == "done"))
    return json.dumps(
        {
            "success": True,
            "session_id": session_id,
            "closed": checkpoint.get("closed", False),
            "notes": len(checkpoint.get("notes") or []),
            "tool_events": len(checkpoint.get("tool_events") or []),
        }
    )


def _format_checkpoints() -> str:
    rows = store.all_checkpoints()
    if not rows:
        return "SessionGuard has no checkpoints yet."
    lines = ["SessionGuard checkpoints (oldest first):"]
    for row in rows[-5:]:
        state = "closed" if row.get("closed") else "OPEN"
        lines.append(
            f"- {row.get('session_id')} [{state}] task={row.get('task') or '-'} "
            f"| tools={len(row.get('tool_events') or [])} files={len(row.get('files_touched') or [])} "
            f"failures={row.get('failures') or 0} last={_ago(row.get('last_seen') or 0)}"
        )
    return "\n".join(lines)


def guard_command(raw_args: str = "", **kwargs: Any) -> str:
    return _format_checkpoints()


def sessionguard_status_tool(params: Dict[str, Any] | None = None, **kwargs: Any) -> str:
    rows = store.all_checkpoints()
    open_rows = [r for r in rows if not r.get("closed")]
    return json.dumps(
        {
            "success": True,
            "checkpoints": len(rows),
            "open_checkpoints": len(open_rows),
            "latest_open": open_rows[-1] if open_rows else None,
            "note": "State lives in <HERMES_HOME>/sessionguard/state.json",
        }
    )


def _cli_setup(subparser: Any) -> None:
    sub = subparser.add_subparsers(dest="sessionguard_action")
    sub.add_parser("status", help="Show SessionGuard checkpoints")


def _cli_handler(args: Any) -> int:
    print(_format_checkpoints())
    return 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    global _CTX
    _CTX = ctx

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)

    ctx.register_command(
        "guard",
        guard_command,
        description="Show SessionGuard work-state checkpoints",
    )
    ctx.register_tool(
        name="session_checkpoint",
        toolset="sessionguard",
        schema={
            "name": "session_checkpoint",
            "description": (
                "Save a work-state checkpoint for this session: a short note on "
                "what is done and what is next, or status='done' to close the "
                "checkpoint when the task is complete. If the session dies, "
                "the next session is briefed from these checkpoints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Progress note: done steps and next steps"},
                    "status": {
                        "type": "string",
                        "enum": ["working", "done"],
                        "description": "working keeps the checkpoint open, done closes it",
                    },
                },
            },
        },
        handler=session_checkpoint_tool,
    )
    ctx.register_tool(
        name="sessionguard_status",
        toolset="sessionguard",
        schema={
            "name": "sessionguard_status",
            "description": "Return SessionGuard checkpoint counts and the latest open checkpoint.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=sessionguard_status_tool,
    )
    ctx.register_cli_command(
        "sessionguard",
        "Inspect SessionGuard work-state checkpoints",
        _cli_setup,
        _cli_handler,
        description="Show SessionGuard checkpoints recorded on this machine",
    )
