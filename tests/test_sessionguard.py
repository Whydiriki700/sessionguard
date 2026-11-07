import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("sg_init", ROOT / "__init__.py")
init = importlib.util.module_from_spec(spec)
sys.modules["sg_init"] = init
spec.loader.exec_module(init)
store = init.store


class FakeCtx:
    def __init__(self, config=None):
        self.config = config or {}
        self.hooks = {}
        self.tools = {}
        self.commands = {}

    def get_config(self, key, default=None):
        return self.config.get(key, default)

    def register_hook(self, name, cb, **kw):
        self.hooks[name] = cb

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = handler

    def register_command(self, name, cb, description=""):
        self.commands[name] = cb

    def register_cli_command(self, *a, **kw):
        pass


def setup_module():
    os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="sessionguard-test-")
    init.register(FakeCtx())


def test_tool_events_and_files_recorded():
    init.on_pre_llm_call(is_first_turn=True, user_message="Build the report", session_id="sess-a", platform="cli")
    init.on_post_tool_call("write_file", {"path": "/tmp/report.txt", "content": "x"}, session_id="sess-a", status="success")
    init.on_post_tool_call("terminal", {"command": "make build"}, session_id="sess-a", status="error")
    rows = store.all_checkpoints()
    cp = [r for r in rows if r["session_id"] == "sess-a"][0]
    assert cp["task"] == "Build the report"
    assert cp["files_touched"] == ["/tmp/report.txt"]
    assert len(cp["tool_events"]) == 2
    assert cp["failures"] == 1
    assert cp["closed"] is False


def test_recovery_briefing_injected_into_next_session_once():
    init.on_pre_llm_call(is_first_turn=True, user_message="Build the report", session_id="sess-src", platform="cli")
    init.on_post_tool_call("write_file", {"path": "/tmp/report.txt", "content": "x"}, session_id="sess-src", status="success")
    first = init.on_pre_llm_call(is_first_turn=True, user_message="unrelated", session_id="sess-b", platform="cli")
    assert first is not None
    ctx_text = first["context"]
    assert "recovery briefing" in ctx_text.lower()
    assert "Build the report" in ctx_text
    assert "/tmp/report.txt" in ctx_text
    second = init.on_pre_llm_call(is_first_turn=True, user_message="again", session_id="sess-d", platform="cli")
    assert second is not None and "recovery briefing" not in second["context"].lower()
    third = init.on_pre_llm_call(is_first_turn=True, user_message="Continue please", session_id="sess-c", platform="cli")
    assert third is not None and "recovery briefing" not in third["context"].lower()


def test_completed_turn_records_outcome_but_does_not_close():
    init.on_pre_llm_call(is_first_turn=True, user_message="Quick task", session_id="sess-e", platform="cli")
    init.on_post_tool_call("terminal", {"command": "ls"}, session_id="sess-e", status="success")
    init.on_session_end(session_id="sess-e", completed=True, interrupted=False)
    rows = store.all_checkpoints()
    cp = [r for r in rows if r["session_id"] == "sess-e"][0]
    assert cp["closed"] is False
    assert cp["last_turn_outcome"] == "completed"


def test_agent_checkpoint_tool_notes_and_done():
    init.on_pre_llm_call(is_first_turn=True, user_message="Long job", session_id="sess-g", platform="cli")
    result = json.loads(init.session_checkpoint_tool({"note": "step 1 done, next step 2", "status": "working"}, session_id="sess-g"))
    assert result["success"] is True and result["closed"] is False
    result = json.loads(init.session_checkpoint_tool({"note": "all finished", "status": "done"}, session_id="sess-g"))
    assert result["closed"] is True
    rows = store.all_checkpoints()
    cp = [r for r in rows if r["session_id"] == "sess-g"][0]
    assert len(cp["notes"]) == 2


def test_secrets_redacted_in_task_text():
    init.on_pre_llm_call(is_first_turn=True, user_message="use api_key = abcdefgh12345678 please", session_id="sess-h", platform="cli")
    rows = store.all_checkpoints()
    cp = [r for r in rows if r["session_id"] == "sess-h"][0]
    assert "abcdefgh12345678" not in cp["task"]


def test_interrupted_session_stays_open():
    init.on_pre_llm_call(is_first_turn=True, user_message="Fragile job", session_id="sess-i", platform="cli")
    init.on_post_tool_call("terminal", {"command": "run"}, session_id="sess-i", status="success")
    init.on_session_end(session_id="sess-i", completed=False, interrupted=True)
    rows = store.all_checkpoints()
    cp = [r for r in rows if r["session_id"] == "sess-i"][0]
    assert cp["closed"] is False
