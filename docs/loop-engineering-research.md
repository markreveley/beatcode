# Loop engineering: a research spike

An examination of Addy Osmani's ["Loop Engineering"](https://addyosmani.com/blog/loop-engineering/)
(June 7, 2026): a fact-check against primary sources, a comparison
with what loop engineering looks like on Elixir/Jido 2 and in Rust,
and an assessment of its applicability to this repository.

Research date: 2026-08-23. Method: the article was read in full; its
product claims were verified against official documentation, the
Claude Code changelog, npm registry publish timestamps, June-2026
Wayback snapshots of the OpenAI Codex docs, and X-post metadata
(snowflake-ID timestamps). Jido and Rust sections draw on hexdocs,
hex.pm, crates.io, GitHub, and vendor docs, checked this session.

---

## 1 · The article in brief

The thesis: the leverage point has moved from *prompting* coding
agents to *designing the systems that prompt them*. A loop is "a
recursive goal where you define a purpose and the AI iterates until
complete." The post's structural claim is that the loop's five
pieces — plus one — no longer require a hand-maintained pile of bash;
they ship inside both major products (the Codex app and Claude Code):

1. **Automations** — scheduled discovery/triage (the heartbeat)
2. **Worktrees** — filesystem isolation for parallel agents
3. **Skills** — project knowledge written down once (SKILL.md)
4. **Plugins/connectors** — MCP-based reach into real tools
5. **Sub-agents** — the maker/checker split
6. **State** — durable memory outside the context window (a markdown
   file, a board): "The agent forgets, the repo doesn't."

It closes with the three problems that *sharpen* as loops improve:
verification is still yours, comprehension debt grows faster, and
"cognitive surrender" — the same loop serving understanding for one
person and avoidance of it for another.

## 2 · Fact-check

### 2.1 Overall verdict

**Substantially accurate — unusually so for a blog post.** Every
major feature claim checked out against primary sources *as of the
June 7, 2026 dateline*, both quotes are genuine, and the post's
chronology is internally consistent (all cited tweets decode to
June 6–7, 2026; every Claude Code feature named had shipped before
the post). The errors found are subtle: one wrong filename, one
mechanism over-generalized from Claude Code to Codex, one
experimental feature presented as standard, and some terminology
drift — plus normal staleness where OpenAI renamed things after
publication.

### 2.2 The quotes

| Quote | Verdict |
|---|---|
| Steinberger: "You shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents." | **Genuine** — X post of June 7, 2026 (~5–6M views). The tweet's opening clause "Here's your monthly reminder that" is silently trimmed; substance verbatim. |
| Cherny: "I don't prompt Claude anymore. I have loops running that prompt Claude and figuring out what to do. My job is to write loops." | **Genuine in substance** — a transcription of spoken remarks from "Boris Cherny: Claude Code & the Future of Engineering" (Acquired Unplugged, presented by WorkOS, June 2, 2026; youtube.com/watch?v=RkQQ7WEor7w). Several slightly different transcriptions circulate; the post's is faithful. His title "head of Claude Code at Anthropic" is accurate (LinkedIn: "Creator & Head of Claude Code"; used by Fortune, June 2026). |

### 2.3 Claude Code claims

Verified against code.claude.com/docs, the public changelog, and npm
publish dates for the version that introduced each feature:

| Claim | Verdict | Detail |
|---|---|---|
| `/loop` re-runs a prompt/command on an interval | **True** | v2.1.71, Mar 6, 2026 ("run a prompt or slash command on a recurring interval"), alongside in-session cron tools |
| `/goal` runs until a condition holds, with a **separate small model** grading completion each turn | **True** | v2.1.139, May 11, 2026. Docs: the condition and conversation are sent to "your configured small fast model, which defaults to Haiku" — the maker/checker split applied to the stop condition, exactly as the article says |
| Scheduled tasks / cron | **True** | In-session cron tools (v2.1.71) + desktop scheduled tasks + cloud Routines |
| Lifecycle hooks (shell commands at lifecycle points) | **True** | Long-standing, documented |
| "Push the whole thing to GitHub Actions" | **Imprecise** | Real integration, but you run Claude Code *inside* GHA via `claude-code-action` with GitHub's own `schedule:` trigger — you don't export `/loop` |
| `--worktree` flag | **True** | v2.1.49, Feb 19, 2026 |
| `isolation: worktree` on a subagent, self-cleaning | **True** | v2.1.50, Feb 20, 2026 |
| Subagents in `.claude/agents/` | **True** | Documented |
| "Agent teams" | **True but overstated** | Shipped v2.1.32 (Feb 5, 2026) as a **research preview, off by default** behind `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, labeled token-intensive. The post presents it as a standard capability |
| Skills = SKILL.md folders, explicit or implicit invocation | **True** | Same format as Codex — the post's central "same shape" observation holds |
| Skills author, plugins distribute | **True** | Docs make exactly this distinction |
| MCP servers + plugins | **True** | — |
| State in "Markdown (**AGENTS.md**, progress files)" | **Wrong filename** | Claude Code's memory file is **CLAUDE.md**. The docs say explicitly: "Claude Code reads `CLAUDE.md`, not `AGENTS.md`. If your repository already uses `AGENTS.md` for other coding agents, create a `CLAUDE.md` that imports it." AGENTS.md is the Codex-side/cross-tool convention. (As a scratch progress file, any markdown works — but the table names the wrong canonical file.) |

### 2.4 Codex claims

Verified against June-2026 Wayback snapshots of
developers.openai.com/codex (which, note, 308-redirect to
learn.chatgpt.com since ~August 2026):

| Claim | Verdict | Detail |
|---|---|---|
| Automations tab: project, prompt, cadence, environment; Triage inbox; empty runs self-archive | **True (June)** | Docs verbatim: findings go to the "'Triage' section… your inbox"; Codex "automatically archives the task if there's nothing to report"; local checkout vs. worktree choice; custom cron syntax. **Renamed since**: Automations → "scheduled tasks," Triage → "Scheduled" view, docs folded into ChatGPT Learn |
| OpenAI internal uses (issue triage, CI-failure summaries, commit briefings, last-week bug hunts) | **True in substance** | Launch post: "daily issue triage, finding and summarizing CI failures, generating daily release briefs, checking for bugs." "Commit briefings"/"bugs somebody added last week" blend in the docs' *example* automations (a 24-hour commit exec briefing; a `$recent-code-bugfix` skill) |
| Automations can call `$skill-name` | **True** | Docs verbatim |
| Codex `/goal`: run to a verifiable stop condition, pause/resume/clear | **True** | June docs confirm (then behind a feature flag). |
| …"same thing" as Claude Code's `/goal` | **Subtly wrong** | The command matches; the *checker architecture* doesn't. Codex docs describe self-assessment ("stop running when it's confident," evidence-based) — **no separate grader model**. The separate-small-model mechanism is Claude Code's. Ironic, since maker≠checker is the article's own core principle |
| Built-in worktree per thread; concurrent threads on one repo | **True** | Launch post + docs ("Handoff" moves threads between Local and Worktree) |
| Skills: SKILL.md dir, `$name` / `/skills`, implicit matching | **True** | Docs verbatim (locations: `.agents/skills` etc.) |
| "Connectors (MCP) plus plugins for distribution" | **Terminology off** | Codex docs bundle **"apps"** and **"MCP servers"** into plugins; "connectors" is ChatGPT/Claude-surface vocabulary. Substance (MCP + plugins-as-distribution) correct |
| Subagents as TOML in `.codex/agents/`; name/description/instructions; optional model + reasoning effort; spawn only on request; parallel + consolidated | **True with nits** | Fields are `developer_instructions` and `model_reasoning_effort` (post paraphrases). "Only when you explicitly ask" was true in June; loosened by August (project/skill instructions can now trigger delegation) |
| "Codex app" as product name | **True (June–Aug)** | macOS app Feb 2, 2026; Windows Mar 4. Current docs re-frame around the "ChatGPT desktop app" |

### 2.5 What the fact-check adds up to

- The post's **load-bearing claim survives**: by mid-2026 both
  vendors ship the same six loop primitives natively, in
  near-isomorphic shapes (SKILL.md is literally the same format), so
  a loop design ports across harnesses. That was verifiable and is
  verified.
- Its notable errors cluster where the two products *differ*:
  AGENTS.md vs CLAUDE.md, self-graded vs separately-graded `/goal`,
  experimental vs standard multi-agent. The post's symmetry framing
  ("the shape is the same") occasionally paves over real asymmetries
  — and the asymmetries are precisely the maker/checker details the
  post itself says matter most.
- Post-publication drift is real but not the author's error: OpenAI's
  August-2026 rename (Automations → scheduled tasks; docs →
  learn.chatgpt.com) has already dated the post's links and
  terminology within ~10 weeks — itself a small lesson about building
  loop knowledge on product nouns rather than the underlying shape.

## 3 · Loop engineering on Elixir — Jido 2

### 3.1 What Jido 2 is

[Jido](https://github.com/agentjido/jido) (hex `jido`; Jido 2.0
stable Feb 22, 2026, v2.3.3 as of Aug 10, 2026) is a BEAM-native
agent framework. Its core is deliberately pure-functional: an agent
is an immutable struct with schema-validated state, and everything
flows through one function —

> **Signal → routing → `Agent.cmd/2` → `{updated_agent, directives}` → runtime executes directives**

Signals are CloudEvents envelopes on a bus with persistent
subscriptions, replay, and DLQs; **directives** are inert
descriptions of effects (`Emit`, `SpawnAgent`, `Schedule`, `Cron`,
`StartSensor`, …) executed only by the runtime; **strategies**
(`Direct`, `FSM` in core; ReAct/CoT/ToT/TRM/Adaptive in `jido_ai`
via ReqLLM's ~11 providers) are the pluggable engine behind `cmd/2`.
Agents run under OTP supervision (`Jido.AgentServer`), hibernate and
thaw through checkpoint storage (ETS/File/Redis), and journal history
in an append-only Thread. `jido_mcp` speaks MCP in both directions,
including generating proxy `Jido.Action` modules from a remote MCP
server's tools.

### 3.2 The article's six pieces, mapped

| Article piece | Claude Code / Codex | Jido 2 |
|---|---|---|
| Automations | Automations tab; `/loop`; cron; Routines | Declarative `schedules:` (cron on the agent), `Directive.Cron`, `Jido.Sensors.Heartbeat`, strategy `tick/2`. In-house scheduler since 2.1.0; **at-most-once, no missed-run catch-up** (docs punt exactly-once to Oban) |
| Run-until-done | `/goal` (CC: separate small-model judge) | No built-in primitive — compose it: an FSM strategy with a terminal state, or a Checker agent gating the transition (which *can* be a different model via ReqLLM aliases) |
| Worktrees | Built-in / `--worktree` / `isolation: worktree` | **No analog.** Process isolation ≠ file isolation; you shell out to `git worktree` yourself |
| Skills | SKILL.md prose, loaded into context | `Jido.Plugin` (1.x literally called these "Skills"): compiled behavior modules bundling actions + signal routes + a namespaced state slice + config schema. Typed and unit-testable — but code, not prose |
| Plugins/connectors | MCP + marketplaces | `jido_mcp` client+server (stdio, streamable HTTP, same-VM `:beam`); ecosystem also has Anubis (ex-Hermes) and ExMCP. MCP is where the two worlds interoperate |
| Sub-agents | `.claude/agents/`, teams; `.codex/agents/` TOML | `Directive.SpawnAgent` + `emit_to_parent`, orphan policies, `Jido.await`; docs name **Maker/Checker** as a first-class pattern; durable multi-agent **Pods** topologies; worker pools |
| State | Markdown files, Linear | Schema state + append-only Thread journal + checkpoints; `InstanceManager` idle-hibernation/thaw; Redis/File/ETS (Postgres DIY) |

The article's morning-triage loop in Jido shape: a `TriageAgent`
(`use Jido.AI.Agent`, ReAct) with `schedules: [{"@daily",
"triage.run"}]`, tools that are Actions wrapping CI/issue APIs (or
MCP proxies); per finding it emits `spawn_agent(FixerAgent, …)`; a
`CheckerAgent` subscribes to `fix.drafted` signals and gates
`pr.open`; state lives in the Thread journal + a Redis checkpoint.
The realistic twist: the Fixer's *hands* would shell out to
`claude -p` / `codex exec` — Jido as the loop layer, product CLIs as
the coding layer. The Jido org is itself building exactly that
bridge: `jido_harness` ("normalized Elixir protocol for CLI AI
coding agents") and `jido_console` ("local control plane for
reliable coding agents").

### 3.3 Advantages

1. **The loop itself is fault-tolerant.** Supervision trees,
   restart-on-crash, orphan policies, `let it crash`. A product-side
   loop dies with the session or laptop unless exported to CI; a
   Jido loop is a supervised process. For a thing whose defining
   property is "runs while you are not watching," a runtime built
   for unattended recovery is the natural home.
2. **The loop is testable.** `cmd/2` is pure — loop logic unit-tests
   without processes, LLMs, or wall clocks. Product loops are config
   plus prose; you learn they're wrong by watching them misbehave.
3. **Maker/checker is structural.** Separate processes, separately
   configured models, signal-routed; the checker *cannot* be the
   maker. The article's most important discipline becomes an
   architecture rather than a convention.
4. **Real observability and state.** CloudEvents signals with replay
   and DLQs, OpenTelemetry, append-only journals — the "markdown
   file as memory" upgraded to an event-sourced record.
5. **Concurrency economics.** Thousands of lightweight keyed agents
   per node with idle hibernation — loop-per-repo / loop-per-customer
   shapes the products can't express.
6. **No vendor coupling.** The same loop can drive Anthropic, OpenAI,
   or local models — or all of them, A/B'd.

### 3.4 Disadvantages

1. **You're back to owning the pile.** The article's core empirical
   observation is that the pieces now ship in the products. Jido
   returns you to build-and-maintain — a framework-shaped pile
   rather than a bash pile, but yours forever.
2. **No coding-agent batteries.** No worktrees, no repo tools, no
   sandboxing/permission model, no SKILL.md ecosystem or
   marketplace. The hard 80% of a *coding* loop arrives only by
   shelling out to the very products the article describes.
3. **Maturity and bus factor.** Effectively a single-author project
   (~1.8k stars, ~127k downloads; 1.0→2.0 was a full rewrite 14
   months in). Docs are genuinely excellent; production evidence is
   thin and mostly unverifiable.
4. **Scheduling guarantees are honest but weak** — at-most-once, no
   catch-up; exactly-once means adding Oban.
5. **Skills-as-code cuts both ways**: a compiled Plugin can't be
   rewritten by the loop mid-run the way an agent can edit its own
   SKILL.md.

**Net:** Jido is loop engineering *as software engineering* — the
wrong tool to replace Claude Code, a compelling tool to *conduct* it:
the supervised control plane above N headless agent CLIs.

## 4 · Loop engineering in Rust

Applicable — in three distinct senses, each with different maturity.

### 4.1 Rust is the language the loop runner is written in

OpenAI rewrote Codex CLI from Node to Rust (now ~95% of the repo);
the stated reasons — no runtime dependency, no GC pauses in
long-lived agent processes, native sandboxing, millisecond startup
*explicitly for parallel `codex exec` fan-out in CI* — are loop
engineering requirements. Block's Goose is Rust (and its scheduler
wraps `tokio-cron-scheduler`, with a platform tool that lets the
agent manage its own schedules). Zed's agent and the Agent Client
Protocol are Rust-native. Warp is Rust. (Contrast: Claude Code, Amp,
OpenCode are TypeScript; Crush is Go.) The protocol layer is where
Rust is vendor-blessed: `rmcp`, the official MCP Rust SDK, has
~21.7M downloads; ACP's official SDK is Rust.

### 4.2 Writing your own loop in Rust: proven, but assembled

- **The idiomatic stack:** `tokio::process` spawning
  `claude -p --output-format stream-json` / `codex exec --json`
  (both officially documented headless modes: NDJSON event streams,
  structured-output schemas, session resume, sandbox flags, cost
  fields) + `tokio-cron-scheduler` or Apalis for the heartbeat +
  sqlx/SQLite for durable state + `rmcp` for protocol surfaces.
- **Existence proof:** `ralph-orchestrator` (~3.1k stars, Rust) —
  the Ralph-loop pattern (Geoffrey Huntley: agent in a `while` loop,
  fresh context per pass, state on disk) industrialized: drives
  Claude Code/Codex/Gemini/Amp/others via their CLIs, iterates until
  `LOOP_COMPLETE` or a cap, rejects incomplete work through
  "backpressure gates" (tests/lint/typecheck), rotates personas,
  persists per-workspace state. The ralph-loop GitHub topic lists
  nine Rust implementations. This is the article's five-pieces-plus-
  state as a single static binary.
- **Inner-loop frameworks** if you don't shell out: `rig`
  (rig-core, ~2.3M downloads, multi-turn agent loop + tools, named
  production users) and `swiftide` (agents + RAG, lifecycle hooks,
  stop conditions; powers kwaak's parallel Docker-isolated agent
  teams). Younger: adk-rust (community port, has a literal
  `LoopAgent`), AutoAgents, graph-flow.
- **Durability:** Temporal's Rust SDK reached public preview May
  2026; Restate ships an official (pre-1.0) Rust SDK — but both
  vendors' *agent* integrations remain TS/Python-first.

### 4.3 The honest gaps

Neither Anthropic nor OpenAI ships an official Rust SDK (the Claude
Agent SDK is Python/TS; the official Codex SDK is TS — which itself
just spawns the Rust CLI and speaks JSONL over stdio, a protocol any
Rust crate could speak; nobody official ships that crate). Community
wrappers are fragmented and tiny; the pragmatic move is ~200 lines
of `tokio::process` + serde against the documented JSON contracts.
Rust is the language loops are *built* in; Python/TS remain the
languages vendors assume loops are *scripted* in.

### 4.4 Rust codebases as loop substrate

A loop is only as good as its verifier, and cargo's are sharp, fast,
and binary: `fmt --check`, `clippy -D warnings`, `cargo test`, plus
a type system that turns whole classes of "the checker must catch
this" into "does not compile." Deterministic builds make stop
conditions crisp. Which brings us to this repository.

## 5 · This repository

### 5.1 beatcode is already a loop-engineering artifact

Every commit in this repo's history is Claude-authored, in two acts:
PR #1 (the *seed*: SPEC.md, frozen goldens, examples, PLAN.md) and
PR #2 (the *build*: Phases 0–3 in four commits, then
adversarial-audit hardening). Mapped onto the article's six pieces:

| Article piece | beatcode equivalent |
|---|---|
| Automations | PLAN.md's phase sequence with named gates; CI re-verifying the flagship claim on every push |
| Worktrees | Branch-per-run (`seed`, `build/v0.1`, this branch); single-agent, so no parallel collision to manage |
| Skills (codified intent) | SPEC.md itself — "Everything needed is in this repository… No external references are required or expected." §12.4's "seven traps (each has bitten a port like this before)" is exactly the article's "we don't do it like this because of that one incident" knowledge, written where every run reads it |
| Plugins/connectors | GitHub PRs as the delivery mechanism |
| Sub-agents / maker–checker | The goldens came from a separate reference implementation (an oracle whose recorded behaviors — `Float.parse` overflow errors, `badarith` crashes, negative-index-from-the-end lane reads — carry BEAM/Elixir fingerprints, a neat coincidence given §3 above); post-build, an adversarial audit produced the hardening commits |
| State/memory | SPEC-GAPS.md — nine under-determination decisions recorded with cites and rationale so no later run re-derives them; PLAN.md's gates as the progress ledger. "The agent forgets, the repo doesn't," implemented literally |

### 5.2 The CI is the interesting part: a checker the maker can't game

`.github/workflows/ci.yml` reads as a checklist of unattended-agent
failure modes, each with a structural counter:

- **Frozen goldens**: a sha256 manifest of `goldens/` plus a
  file-count check — the loop cannot "fix" a failing test by editing
  the oracle.
- **Test-roster manifest**: `cargo test -- --list` diffed against a
  committed roster — test deletions always fail; additions must
  appear in a reviewable diff. `#[ignore]` is grepped and *also*
  checked semantically via `--list --ignored`.
- **Banned-token grep**: transcendental/FMA tokens in `src/` fail CI
  (SPEC §8.4's determinism rule), so the loop can't quietly reach
  for `sin()` where bit-stability forbids it.
- **Cross-OS render-hash matrix**: ubuntu and macOS re-render the
  four example scores and diff sha256s against committed hashes —
  the product's flagship claim, re-proven on every push.
- **Working-tree-unchanged**: nothing may mutate tracked files
  during CI.

This is the article's "verification is still on you" answered
structurally: the verifier is cheap, total, and adversarially
hardened, so an unattended "done" approaches a proof rather than a
claim. (Re-verified in this session: the full test suite and the
render-hash check pass in this container, byte-identical on a third
platform beyond the CI matrix.)

Note also the pleasant miniature: `bc loop` — the jam loop (200 ms
mtime poll; re-render + play on save; errors print
`!! <msg> (fix and save again)` and *keep watching*) — is a human
feedback loop with the same anatomy: watch → act → verify → survive
errors → keep going. SPEC-GAPS #5 even records its liveness posture.

### 5.3 Why this codebase is an unusually good loop substrate

Determinism is the product: same score + same seed ⇒ byte-identical
WAV, sha256 printed as a receipt. That hands any future loop a
perfect, binary stop condition — the thing `/goal`-style loops are
weakest at is fuzzy completion criteria, and this repo has none.

### 5.4 Loops worth building here

1. **Fuzz/property loop** (`/loop` or a scheduled task): generate
   random scores → `bc events` + double `bc render` → assert
   invariants (sorted events, finite `performed_s`, byte-equal
   double render, line-cited errors). Findings append to a
   `FINDINGS.md` state file; human triage in the morning.
   Determinism makes every failure a one-file repro.
2. **Toolchain-bump automation**: on each new stable Rust, a
   scheduled run bumps `rust-toolchain.toml` in a worktree, runs the
   full suite and `scripts/check_renders.sh`; green ⇒ PR, red ⇒
   triage report. The committed hashes are a tripwire for silent
   codegen drift — the exact risk §8.4 exists to kill.
3. **Mutation-testing loop**: nightly `cargo-mutants`; surviving
   mutants filed as coverage gaps. The test-protection CI means the
   loop can't respond by weakening tests.
4. **v0.2 as a spec-first loop**: repeat the v0.1 pattern — write
   the SPEC delta and new frozen goldens first, then let a fresh
   agent build to the gates. PLAN.md's "make the call, record it in
   SPEC-GAPS.md, and continue — do not stall" is a loop-liveness
   rule, already written.
5. **The repo as an agent eval**: seed + gates + goldens form a
   repeatable benchmark. A loop can re-run "build from seed" per
   model or configuration and score the result against §11.3 —
   turning the repository into exactly the kind of harness the
   article's "factory model" post gestures at.

### 5.5 Honest limits

A tiny, finished, zero-dependency repo with no issue traffic gives
the article's *discovery* loops (daily issue triage, CI-failure
summaries) nothing to chew on. Applicability here is verification
amplification and regression watch, not throughput. The deeper
point runs the other way: beatcode is less a place to *apply* loop
engineering than a demonstration of its precondition — an
environment where "done" is machine-checkable and the checker is
hardened against the maker. The article argues you should build the
loop; this repo shows what you have to build *first* so the loop
can be trusted.

---

## Sources

**Article + quotes**: addyosmani.com/blog/loop-engineering/ ·
x.com/steipete/status/2063697162748260627 ·
youtube.com/watch?v=RkQQ7WEor7w (Acquired Unplugged, June 2, 2026) ·
fortune.com (June 9 & 11, 2026 Cherny pieces).
**Claude Code**: code.claude.com/docs (goal, scheduled-tasks,
worktrees, agent-teams, skills, plugins, memory, hooks,
github-actions, headless) · github.com/anthropics/claude-code
CHANGELOG.md · registry.npmjs.org publish timestamps.
**Codex**: web.archive.org June-2026 snapshots of
developers.openai.com/codex/{app/automations, skills, subagents,
app/worktrees, use-cases/follow-goals, plugins} · current
learn.chatgpt.com equivalents · openai.com/index/introducing-the-codex-app/.
**Jido**: jido.hexdocs.pm (readme, core-loop, strategies, signals,
directives, sensors, plugins, scheduling, storage, orchestration,
pods) · hex.pm/packages/jido (versions) · github.com/agentjido/{jido,
jido_ai, jido_mcp, jido_harness, jido_console} · jido.run blog ·
news.ycombinator.com/item?id=47263036.
**Rust**: github.com/openai/codex · github.com/block/goose ·
zed.dev/acp · crates.io ({rig-core, swiftide, rmcp,
agent-client-protocol, tokio-cron-scheduler, apalis, async-openai,
temporalio-sdk, restate-sdk}) · github.com/mikeyobrien/ralph-orchestrator ·
github.com/topics/ralph-loop · github.com/snwfdhmp/awesome-ralph ·
code.claude.com/docs/en/headless ·
learn.chatgpt.com/docs/non-interactive-mode.md.
**This repo**: SPEC.md · PLAN.md · SPEC-GAPS.md · goldens/README.md ·
.github/workflows/ci.yml · git history (PRs #1–#2).
