# SessionGuard

**Work survives the session.**

Hermes can resume a conversation with `/resume`, but a transcript is not
work state: it does not say which steps already changed the outside world,
and after a gateway death or a context compaction the next session starts
blank or lossy, so the agent has to guess where it was, redoing side
effects or dropping the task entirely.

SessionGuard keeps a local work-state checkpoint for every session, built
from tool evidence rather than prose.

## What it does

- **`pre_llm_call`** records the current task text each turn. On the first
  turn of a session it briefs the agent twice over: how to checkpoint its
  own progress with the `session_checkpoint` tool, and, when a previous
  session left an unclosed checkpoint, a **recovery briefing** (previous
  task, files touched, last tool calls with outcomes, failure count, and
  the previous agent's own progress notes), with an explicit instruction
  to verify real-world state before redoing any step.
- **`post_tool_call`** records every tool call as a short redacted
  summary, tracks the files it touched (path arguments, plus absolute
  paths named inside terminal commands), and counts failures.
- **`session_checkpoint` tool** lets the agent save progress notes as it
  works and close the checkpoint with `status="done"` when the task is
  complete. A checkpoint closes only this way; everything else stays
  open and brief-worthy.
- **`on_session_end`** records each turn's outcome (completed, failed,
  interrupted) on the checkpoint.

State lives under `<HERMES_HOME>/sessionguard/`: `state.json` (latest
checkpoint per session, capped at 50 sessions) and `events.jsonl`
(append-only audit trail). Inspect with `/guard`, the
`sessionguard_status` tool, or `hermes sessionguard status`.

## Configuration

```yaml
plugins:
  entries:
    sessionguard:
      settings:
        enabled: true   # false disables checkpointing and briefings
```

## Disclosure

SessionGuard observes user messages (it keeps only the latest task text,
truncated and with credential-shaped values redacted), tool call
arguments (stored only as short summaries), and turn outcomes, in
sessions of profiles where it is enabled. Everything is stored locally
under `<HERMES_HOME>/sessionguard/`. It makes no network calls and
starts no processes. Storage errors degrade every hook to a no-op; the
agent loop is never broken by a checkpoint failure. Declared
capabilities in `plugin.yaml` match the registered hooks and tools
exactly.

## Verification

- `hermes plugins validate`: passed, security scan safe.
- `hermes plugins doctor`: OK, 2 tools and 3 hooks registered under the
  real runtime contracts.
- Unit tests cover checkpoint recording, file tracking (including paths
  inside terminal commands), one-shot recovery briefings, agent notes
  and done-closing, secret redaction in task text, and interrupted
  sessions staying open.
- Live two-session test: session one created files and saved a
  `session_checkpoint` note, then stopped mid-project; a fresh session
  accurately reported the unfinished work (files done, missing part)
  from the injected recovery briefing alone.
