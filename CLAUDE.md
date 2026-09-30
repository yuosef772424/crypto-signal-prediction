# Working Instructions for This Project

## Language
- **Reply to the user in Arabic.** Instructions, agent prompts and protocol documents may be in English (models follow English instructions more faithfully); translate on demand.

## Subagents
- **Default:** delegate large, self-contained tasks to a background agent — e.g. building a tool or package, long experiments, or reading and analysing many files or large run logs.
- **Goal: protect the main conversation's context, not reduce total usage.** An agent starts with an empty context and returns only a short report, so the main conversation stays light and focused on the core line (model, training, decisions) and is compacted later, losing fewer details. Total tokens may go up, not down, because the agent re-reads what it needs.
- **An agent's prompt carries only what it needs:**
  - the goal and the definition of done;
  - paths and branches;
  - constraints;
  - relevant prior results, so it doesn't redo settled experiments;
  - the report format expected from it.
- **Do not delegate:**
  - small tasks: a one-line edit, a quick question, reading a single file;
  - work that depends on a step-by-step discussion with the user;
  - work that needs most of the conversation's context — explaining it would cost more than doing it.
- **Working with running agents:**
  - never duplicate the work of an agent that is still running;
  - send it new requirements by message instead of spawning another agent;
  - relay a summary of its report to the user, not the full text.
- **Model choice (cost):** spawn subagents with `model: "sonnet"` by default — tasks are delegated with a clear spec, so the cheaper model suffices. Use a stronger model only for open-ended research or design work, and say why when you do.
- **Auditor/builder rounds** follow `docs/research/audit/PROTOCOL.md`; the auditor's independence from the builder's reasoning is its most important rule.

## Scope
- Do what the task asks, no more. Extra research, broad audits or extra verification passes only when the user asks for them.
