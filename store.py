"""SessionGuard storage: local work-state checkpoints.

  <HERMES_HOME>/sessionguard/state.json   - latest checkpoint per session
  <HERMES_HOME>/sessionguard/events.jsonl - append-only audit trail

A checkpoint is work state, not a transcript: the current task text, the
files touched, the most recent tool calls with their outcomes, failures,
and the agent's own progress notes. Nothing here talks to the network,
and argument values are stored only as short summaries with credential
shaped values redacted.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_MAX_EVENTS = 30
_MAX_FILES = 40
_MAX_NOTES = 10
_SECRETISH = re.compile(r"(?i)(sk-[a-z0-9_\-]{8,}|ghp_[a-z0-9]{8,}|bearer\s+\S+|api[_-]?key\s*[:=]\s*\S+)")


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def data_dir() -> Path:
    path = hermes_home() / "sessionguard"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return data_dir() / "state.json"


def load_state() -> Dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            return data
    except Exception:
        pass
    return {"sessions": {}}


def save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_event(record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        with open(data_dir() / "events.jsonl", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def clean_text(text: Any, limit: int = 300) -> str:
    value = _SECRETISH.sub("[redacted]", str(text or ""))
    value = " ".join(value.split())
    return value[:limit]


def summarize_args(tool_name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return clean_text(args, 120)
    for key in ("command", "path", "file_path", "url", "query", "title", "name"):
        if args.get(key):
            return clean_text(args[key], 120)
    keys = ", ".join(list(args.keys())[:5])
    return f"{tool_name}({keys})"


_PATH_TOKEN = re.compile(r"(?<![\w/])(/(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z0-9]{1,6})\b")


def touched_files(tool_name: str, args: Any) -> List[str]:
    """Files a tool call touched: explicit path arguments first, and for
    terminal calls the absolute file paths named inside the command."""
    found: List[str] = []
    if not isinstance(args, dict):
        return found
    for key in ("path", "file_path", "file", "target", "filename", "output_path"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip()[:200])
    if tool_name in ("terminal", "shell", "bash", "exec"):
        command = args.get("command")
        if isinstance(command, str):
            for match in _PATH_TOKEN.finditer(command):
                found.append(match.group(1)[:200])
    return found


def get_or_create(state: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    sessions = state.setdefault("sessions", {})
    checkpoint = sessions.get(session_id)
    if checkpoint is None:
        checkpoint = {
            "session_id": session_id,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "platform": "",
            "task": "",
            "turns": 0,
            "tool_events": [],
            "files_touched": [],
            "failures": 0,
            "notes": [],
            "closed": False,
            "close_reason": "",
            "briefed": False,
        }
        sessions[session_id] = checkpoint
        if len(sessions) > 50:
            oldest = sorted(sessions.values(), key=lambda c: c["last_seen"])[: len(sessions) - 50]
            for stale in oldest:
                sessions.pop(stale["session_id"], None)
    return checkpoint


def record_turn(session_id: str, task_text: str, platform: str = "") -> Dict[str, Any]:
    with _LOCK:
        state = load_state()
        checkpoint = get_or_create(state, session_id)
        checkpoint["last_seen"] = time.time()
        checkpoint["turns"] = int(checkpoint.get("turns") or 0) + 1
        if task_text:
            checkpoint["task"] = clean_text(task_text, 300)
        if platform:
            checkpoint["platform"] = platform
        save_state(state)
        return checkpoint


def record_tool(session_id: str, tool_name: str, args: Any, status: Any) -> None:
    with _LOCK:
        state = load_state()
        checkpoint = get_or_create(state, session_id)
        checkpoint["last_seen"] = time.time()
        event = {
            "ts": time.time(),
            "tool": tool_name,
            "summary": summarize_args(tool_name, args),
            "status": str(status or ""),
        }
        events = checkpoint.setdefault("tool_events", [])
        events.append(event)
        checkpoint["tool_events"] = events[-_MAX_EVENTS:]
        if str(status or "").lower() not in ("", "success", "ok", "completed"):
            checkpoint["failures"] = int(checkpoint.get("failures") or 0) + 1
        files = checkpoint.setdefault("files_touched", [])
        for path in touched_files(tool_name, args):
            if path not in files:
                files.append(path)
        checkpoint["files_touched"] = files[-_MAX_FILES:]
        save_state(state)


def add_note(session_id: str, note: str, close: bool = False) -> Dict[str, Any]:
    with _LOCK:
        state = load_state()
        checkpoint = get_or_create(state, session_id)
        checkpoint["last_seen"] = time.time()
        if note:
            notes = checkpoint.setdefault("notes", [])
            notes.append({"ts": time.time(), "note": clean_text(note, 400)})
            checkpoint["notes"] = notes[-_MAX_NOTES:]
        if close:
            checkpoint["closed"] = True
            checkpoint["close_reason"] = "agent_marked_done"
        save_state(state)
        return checkpoint


def close_session(session_id: str, reason: str) -> None:
    with _LOCK:
        state = load_state()
        checkpoint = get_or_create(state, session_id)
        checkpoint["closed"] = True
        checkpoint["close_reason"] = reason
        checkpoint["last_seen"] = time.time()
        save_state(state)


def record_turn_outcome(session_id: str, outcome: str) -> None:
    """on_session_end fires per turn in Hermes, so it records the last
    turn's outcome; it never closes a checkpoint by itself."""
    with _LOCK:
        state = load_state()
        checkpoint = get_or_create(state, session_id)
        checkpoint["last_turn_outcome"] = outcome
        checkpoint["last_seen"] = time.time()
        save_state(state)


def latest_open_checkpoint(exclude_session: str) -> Optional[Dict[str, Any]]:
    state = load_state()
    candidates = [
        c
        for c in state.get("sessions", {}).values()
        if not c.get("closed")
        and c.get("session_id") != exclude_session
        and (c.get("tool_events") or c.get("notes"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.get("last_seen") or 0)
    return candidates[-1]


def mark_briefed(session_id: str) -> None:
    with _LOCK:
        state = load_state()
        checkpoint = state.get("sessions", {}).get(session_id)
        if checkpoint is not None:
            checkpoint["briefed"] = True
            save_state(state)


def all_checkpoints() -> List[Dict[str, Any]]:
    state = load_state()
    rows = list(state.get("sessions", {}).values())
    rows.sort(key=lambda c: c.get("last_seen") or 0)
    return rows
