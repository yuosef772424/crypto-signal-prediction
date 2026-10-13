# Working method: one planner session, builder agents, low token cost

A method for running a multi-phase project with Claude Code. One session plans and reviews, and cheaper
subagents do the building. It is written to be reused in other projects: copy this file and adapt §6.

## 1. Roles

| Role | Where it runs | Model | Does |
|---|---|---|---|
| Owner | phone / app | — | decides, runs Colab cells, pastes results, runs `/compact` when asked |
| Planner/reviewer | the one main session | strong model | plans, writes agent prompts, reviews by evidence, merges, keeps STATUS.md |
| Builder | a subagent launched by the planner (`Agent` tool) | `sonnet` | implements one phase = one branch = one PR, runs tests, reports ≤ 40 lines |
| Fix / split agent | a subagent launched by the planner | `sonnet` | fixes one PR or moves work between branches; never merges |

The owner does not open separate build sessions: every builder is a subagent of the planner session.

## 2. Why the cost is high, and the main rule

Every time the planner is *invoked*, it re-reads its whole context: system prompt, tools, conversation. Cached
tokens cost about 10 % of normal input, but with a large context and many wake-ups they add up. For example,
30 wake-ups on a 350k-token context re-read about 10M tokens. **The number of planner invocations is the main cost driver.**

What invokes the planner:
1. a message from the owner;
2. a background subagent finishing (one wake per agent, at the end only; its progress does not wake the planner);
3. a background shell command finishing;
4. GitHub events on a PR the session is subscribed to. **Creating a PR subscribes the session automatically**,
   so every CI result, ready-for-review flip, comment and merge is a separate wake.

**Main rule: one invocation does a whole batch.** In a single turn the planner reviews, decides, merges, writes
the docs update and launches the next agents. It never wakes up just to acknowledge something.

## 3. Planner rules

- **Batch.** Do as much as possible in each turn: for example, verify branch A → open its PR → mark it ready →
  merge it → update STATUS → launch the agent for B.
- **PRs: open, mark ready and merge in the same turn**, or call `unsubscribe_pr_activity` right after creating a PR
  that will stay open. Never keep a subscription "to watch".
- **No background shell commands in the planner.** Long tests (pytest ≈ 13 min) run inside the agent's task, and
  the agent reports the counts.
- **Agents.** Launch an agent in the background only when the planner has nothing else to do; it costs one wake
  when it ends. If the result is needed in the same turn, run it in the foreground.
- **Parallel agents get `isolation: "worktree"`** (or their own clone). Two agents, or an agent and the planner,
  must never share one working tree: they overwrite each other's checkout. When the planner edits docs while an
  agent runs, it uses its own `git worktree`.
- **Ignore a stop hook about "uncommitted changes" that a running agent owns**: never commit another agent's
  work in progress.
- **Keep the planner context small.**
  - Agents return short reports; the planner never reads large files or logs itself.
  - Verify by cheap commands: `git log --oneline`, `git diff --stat`, the registry diff, merge-tree.
  - Keep ≤ 50k tokens of messages. Before that limit, give the owner a ready `/compact <what to keep>` text.
- **Do not reply to routine notifications** (merged / ready / subscription created). Read them and stay silent,
  or answer in one line when a reply is unavoidable.

## 4. Agent prompts (cache-friendly)

- **Shared prefix.** Prompt caching is prefix-based. Agents of the same type share only an identical prefix:
  system prompt, tools, and identical leading prompt text. Start every agent prompt with the same
  **COMMON RULES** block, byte-for-byte, and put the task-specific part last.
- **File reads are per agent.** Every agent re-reads the files it needs. Give it exact paths and line ranges,
  and list the prior results it must not redo.
- **Each prompt carries:**
  - the goal and the definition of done;
  - the branch, and whether it is new from `origin/master`;
  - the constraints (never force-push, never merge, append-only registries, one change type per commit);
  - the checks to run before pushing;
  - the report format (≤ 40 lines: full head SHA, commits, test counts, deviations from the brief).
- **Deviations.** The agent lists every deviation; the planner accepts or rejects each one explicitly.
- **Default model `sonnet`.** Use a stronger model only for open-ended research or design, and say why.

## 5. Review and merge (planner)

1. Read the agent's report. Re-check what it claims with cheap commands, not by trusting the narrative:
   commit order, protected files, append-only registry diff, `merge-tree` for conflicts.
2. Apply the accept rule mechanically. A failed phase is a valid result and is recorded as a failure.
3. Create the PR → mark it ready → merge with the full 40-character `expectedHeadSha`, all in one turn.
4. The STATUS.md update rides in the same or the next batch, never in a turn of its own.

## 6. Owner checklist

- **Talk to the planner only to:** decide, paste a Colab bundle or raw data, or run the `/compact` text it gives.
- **Measure:** use `/context` for context size and `/usage` for cost. The planner's messages should stay under
  about 50k tokens.
- **Keep sessions apart:** start a new project in a new planner session. Copy this file and the project's
  STATUS/ROADMAP into it; never reuse a huge old context.

## 7. Lessons that produced these rules (this project, 2026-10)

- One planner wake ≈ the planner's whole context re-read. Notification chains (subscription created → ready →
  merged) each cost a wake.
- Measured agent reviews cost 95k–160k tokens each. That is still cheaper than doing the same reading in the
  planner context, which would then be re-read on every later wake.
- An agent and the planner shared the same clone: the planner's `git checkout` failed against the agent's
  uncommitted file. Hence the worktree isolation rule.
