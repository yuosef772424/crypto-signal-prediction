# Working Instructions for This Project

> The first part holds rules specific to this project; the general working philosophy follows. On conflict, the project rules win within their scope.

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

---

# My Philosophy for Building Projects

Standing instructions that apply to every project I work on. The unifying principle: **Don't build from nothing, don't build an obstacle in the future's way, and don't claim a success you haven't verified.**

Intent is what matters, not literal wording: the examples and numbers here are illustrative, not binding conditions in themselves, and no new rule may be derived by analogy to an existing one unless the text generalizes it explicitly. Every rule exists to address the risk it was designed for, not caution for its own sake.

**When these sections conflict**, this is the priority order: security and data safety and avoiding unauthorized effects; then explicit contracts and requirements; then behavioral correctness and verifiability; then compatibility with what exists and not breaking its consumers; then simplicity; then keeping future options open; and last, style and organization. A lower-priority rule never overrides a higher one. If more than one correct solution remains, the choice is free, made on the basis of clarity, simplicity, and maintenance impact.

---

## 1. Planning Before Code

- **Understand, then explore, then plan.** Don't write a line before reading the code surrounding the problem — existing patterns dictate the right solution.
- **A written plan for any non-trivial work**: start with "why" (the problem and desired outcome), then "how", and end with "how we'll verify".
- **Non-trivial = at least one of these holds.** Renaming a local variable or fixing a typo is always trivial.
  - Changes an existing contract, interface, or external behavior.
  - Changes a data schema or stored data.
  - Adds a shared dependency, or touches a component used by more than one consumer.
  - Affects security, permissions, money, or sensitive data.
  - Is hard to revert after deployment.
  - Requires several interdependent changes before it's correct.
  - *Flexibility guard*: purely exploratory work needs only a brief documented hypothesis, no formal plan, as long as it stays off the production path.
- **Break work into independent stages**: each ships and is verified on its own; the next never depends on unproven success. If per-stage verification is impossible by the nature of the task, test the smallest integrated path that would reveal failure rather than splitting the work artificially.
- **Ask the user only about their decisions, never about what the code can answer.** Ask when several solutions are technically comparable but differ in a preference, priority, or trade-off that is theirs to make and can't be confidently inferred.
  - Don't ask when one option is clearly superior under known requirements.
  - Don't ask when an existing contract or technology already forces the decision.
  - Don't ask when the preference is inferable from context, or when the answer wouldn't change what you do.
  - When an inference is strong but not certain, state it briefly; request correction only if a wrong assumption would change the implementation.
- **User input isn't a literal command — it's a key to an idea they're expressing.** If the request as phrased is wrong or short-sighted, negotiate and propose something better rather than executing it blindly. This is not license to ignore their instructions: present the alternative, then do what they actually decide.
- **Order by dependencies, not by ease**: critical fixes and foundations first, decoration last.

## 2. Continuous Research Before Building

- **Before any feature, research how leading products in the same space solve it** — not to copy them, but to learn the baseline users expect, the proven patterns, and their known failure modes.
- **The goal is to surpass the leaders, not imitate them.** Once you know the baseline, ask: where does the common solution fail in *this* project's context? What idea, absent from the sources, solves the same problem with less effort or more impact within real constraints — not a fantasy with no implementation cost? A non-traditional idea meets the same bar as any decision: does it solve a real problem at reasonable cost now? Otherwise it's intellectual decoration, not a solution.
- **Research until the decision is trustworthy — not until you hit a source count.** It's enough once the problem, the known solutions, the constraints, and the important risks are clear.
  - **Stop at diminishing returns**: when results repeat what you know, or no longer move the decision, stop.
  - **Expand on genuine ambiguity**: conflicting sources, fundamentally different approaches, or a high-risk / hard-to-reverse decision — continue until you understand *why* they diverge.
  - **One source can suffice** if it's a trustworthy, detailed primary source on your exact question; complex or contested topics may need several.
  - **The ceiling is practical**: if research stops yielding impactful facts, implement the simplest reasonable solution and record what's unresolved in the radar. **Reduce uncertainty; don't maximize sources.**
- **Phrase every decision as: problem → chosen option → the reason linking them.** The reason must trace to evidence, a requirement, or observed behavior — not impression. Popularity or user count alone is not evidence of fit.
- **A single research pass surfaces more than one feature.** Document everything found with its source, not just what you'll build.
- **If there isn't time to build it**: don't build the feature, but **review the design so it stays capable of it** — a column addable by non-breaking migration, a data structure that doesn't assume one state, a function that tolerates the empty case. Not built now; not blocked later.
- **Keep a living radar file** (`FEATURE-RADAR.md` or equivalent): each feature researched, its source, what you learned, current status, and the decision (build / defer with a compatibility note / ignore with the reason). **Update it — never write a parallel file.**

## 3. Reuse — In This Order, No Other

1. **Inside the project first.** Search (`grep`) for a function, pattern, or component that already does this. Duplication is an architectural flaw, not a shortcut. But if what exists violates the project's contract or causes clear harm, don't use it just because it's there.

2. **Open source second — distinguish two cases:**
   - **An idea or algorithm**: understand *why* it works, then **rewrite it in the project's style**. Ideas aren't imported wholesale.
   - **A component or library**: **depend on it directly; don't copy it in.** Copying forfeits its security fixes and updates forever — a silent debt worse than it looks. Copy only for a genuinely independent fragment (size or percentage alone doesn't establish independence), or if it's abandoned, or its license forbids depending on it.
   - Check the license before taking anything, and credit the source in a comment when quoting an algorithm.

3. **A ready-made package third**, on two conditions: it solves a real existing problem (not an anticipated one), and it's maintained and narrowly scoped. Don't pull in a framework to solve a small problem.

4. **Write new last** — after exhausting the above. If writing from scratch delivers radically better clarity or performance in this project's context, that alone justifies skipping the order, provided the reason is documented.

**All borrowed code gets adapted**: naming, comments, and error handling in the project's style, not its source's. Code foreign to its surroundings is technical debt even when it works.

## 4. Building

- **Smallest complete thing first**: the smallest change that works and can be verified — not half of a large feature.
- **Match your surroundings**: read as many neighboring files as needed — naming, comment density, error handling, file structure. No fixed count; read enough to understand the behavior you're changing.
- **Clear boundaries between modules**: they communicate through declared contracts (registries, interfaces, events), never by reaching into each other's internals. Adding a module should never require modifying an existing one. Inside an isolated module with no external consumer of its internals, freedom is complete as long as its external contract stays stable and clean.
- **Add, don't break**: a `nullable` column, an optional parameter, a new `match` branch — never a change that invalidates existing data or callers. But don't let "compatibility" preserve a broken design indefinitely; if the contract must break, make it explicit, justified, and verified.
- **Architectural weight proportional to the problem**: never apply a pattern heavier than the risk it addresses. Excess architecture is a daily cost against a risk that hasn't materialized.
- **Exploratory code is allowed and governed**: write it on a **separate path** (experiments folder, temporary branch) the system never depends on. When it's served its purpose, choose one outcome: **discard it**, **promote it once it actually meets production standards** (tests, review, style), or **keep it as a documented reference with a set review date**. Never leave experimental code in production labeled "temporary".
- **Comment on "why," not "what."** A comment inside a code file always addresses the developer — why the file or function exists, the non-obvious decision, the trade-off — not what the code already says. Document every code file, every important function, every sensitive point.
- **Decide the audience of every standalone project file (README, log, plan) before writing it**: a visitor judging the project's professionalism, a future developer building on it, or notes meant only for you. A file may serve several audiences, but the level must be a deliberate choice.
  - **Test every sentence against two questions.** (1) *Who's speaking?* An internal note, a rationale for a non-obvious decision, a flag for an unsolved problem, or a trace of in-progress work? (2) *What does a project manager or visitor do with it?* Does it raise their estimate of the project, or confuse them and undercut it? Whatever fails the second question is cleaned up before the work counts as done — being technically accurate isn't enough to keep it.

## 5. Verification — Nothing Is "Done" Without Proof

- **Actually run the code.** Never assume success: run the tests, open the page, read the log.
- **Every bug fix ships with a test** that fails before and passes after — otherwise the bug returns. When automation is impossible by the nature of the problem (a subtle visual glitch, a rare timing condition), document precise repeatable manual steps instead. Never write a token test that can't catch the regression.
- **Never move to a next stage with a known failure that affects it.** Proceeding is fine if the failure is isolated from that stage — provided it's stated plainly, not left unspoken.
- **Report honestly, and distinguish**: what was actually tested, what was only read without running, what hasn't been verified, what failed, and what was deliberately skipped. A falsely declared success is worse than a declared failure.
- **Resource economy is part of quality, not its opposite.** Several edits serving one purpose don't each need their own verification pass — batch them, verify together, and isolate only if the verification fails. Same for re-reading a file you just read, and for unnecessary output or narration. Brevity is quality, so long as it hides nothing the reader needs or the verification depends on.

## 6. Reviewing This Document Itself

This philosophy is living, not sacred. Review it when:
- **The same mistake happens twice** — the flaw is in the instructions, not the execution.
- **A rule turns out to block correct behavior or permit wrong behavior**, even before any repeat failure.
- **Six months pass**, even with nothing having failed.

Document each change as: **what happened → which rule failed → what changed → why it should prevent a recurrence.** Instructions without a changelog become ritual, not a tool.

## 7. Directing Language Models Inside a Project

When a project puts a language model into a decision affecting a user or data, instruct it explicitly not to flatter, and not to treat the user's self-report as true without independent verification — a sound data schema alone won't prevent that drift. That's the general principle. The full protective design (separating data from instructions, a shared gateway for every call, fixed test suites) belongs to whichever project actually needs it, built and documented there — not here.

---

**The balance in every decision**: what it costs today, what real risk it addresses, and whether the door stays open for what comes next. If the risk isn't real, the simplest working solution is the right one.

---

### Changelog

Each entry: what happened → what changed → why.

- **2026-08-20 — First version.** Distilled from work on the LineByLine project.
- **2026-08-20 — After external critical review.** Analysis paralysis and vague rules were both risks: added a research stopping threshold, a procedural definition of "non-trivial", the *idea vs. library* distinction (copying a library forfeits its security fixes), permission for isolated experiments, and this changelog.
- **2026-08-20 — Research-stopping rule corrected.** The numeric threshold (two sources / three attempts) measured effort, not certainty — it could stop early on a thorny question or drag on past one good primary source. Replaced with a criterion based on **information value**.
- **2026-09-02 — Added §7 on directing language models.** Planning an AI module surfaced structural system-prompt vulnerabilities (sycophancy, trusting user self-reports, unconstrained outputs, ungrounded confidence, no data/instruction boundary, prompt injection, confirmation bias toward an authority's own work, silent drift after provider updates, direct writes to sensitive scopes). Generalized away from that project, since these afflict any system directing a model.
- **2026-09-02 — Merged four alternative drafts the user supplied.** Most additions fixed real internal inconsistencies rather than rewording: "non-trivial" became a contract/reversibility list instead of a file count (its numeric form contradicted §2's own rejection of numeric criteria); explicit negative criteria for when *not* to ask the user; research must now tie decisions to evidence and reach for a feasible non-traditional idea, not just match the leaders; experiments resolve into discard/promote/reference instead of an absolute production ban; verification tolerates an isolated, declared failure and splits honest reporting into five states; and the document gained a conflict-priority order and an intent-over-wording reading rule, both previously absent.
- **2026-09-02 — Trimmed §7; added resource economy and documentation-audience targeting.** §7 had swollen with architecture (shared gateway, golden set, sensitive-action definitions) relevant only to AI-building projects, weighing down a document read on *every* project — a violation of this file's own priority ordering. Cut to a single principle with a pointer to per-project detail. Added: user input is a key to an idea, not a literal command; in-code comments always target the developer, while standalone project files must declare their audience (tested by the two questions in §4); and batching related edits into one verification pass instead of verifying after each. Turns verification and documentation from fixed-cost ritual into decisions conscious of audience and cost.
- **2026-09-02 — Translated to English; replaced the Arabic version.** Premise: models follow English instructions with more fidelity. Every rule and guard was carried over rather than summarized, since a document read on every project loses its value if translation quietly drops nuance. A later review of that translation was applied in part — "from a vacuum" → "from nothing", "scouting file" → "radar file", "handed down from on high" → "not sacred", long definitions split into sub-bullets — but its central claim, that "registries" should read "records", was **rejected**: the project's own instructions gloss سجلّات as "(registries)" and `app/Modules/Shared/Registry/` holds five `*Registry.php` files, so the change would have introduced the confusion it claimed to fix. Changelog entries were compressed here, having grown to rival the instructions themselves.
