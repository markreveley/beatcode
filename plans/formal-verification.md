# Formal verification: beatcode as a spec-determined codebase

Status: proposal / analysis. Nothing here is implemented yet.

## 1 · The question

Agents now write much of the code. The open problem is how to know
that what they wrote is correct, and how to stop invariants from
drifting as agents rewrite things. This note asks whether beatcode
can be a worked example of an answer: a defined set of invariants and
specs in a domain that suits verification, such that

- the program reproduces prescribed outputs bit for bit;
- a verification run establishes its correctness; and
- the codebase could in principle be regenerated from the spec and
  come out the same.

**Verdict: yes, with one distinction kept clear.** beatcode already
has most of the setup. What it has today is *conformance testing*,
not *formal verification*. The plan below says how to close that gap,
and names the thing that is actually new here.

## 2 · What already exists

The repo was built spec-first. `SPEC.md`, `goldens/` and `PLAN.md`
landed as a seed, and an agent built the implementation from them.
CI then keeps any later agent from weakening the checks:

| Mechanism | Where | What it prevents |
|---|---|---|
| Frozen goldens, hash manifest | `.github/goldens.sha256`, CI "goldens are frozen" | editing the oracle so it agrees with the code |
| Test roster must match exactly | `.github/test-manifest.txt`, CI "no test deleted or ignored" | deleting or `#[ignore]`-ing a failing test |
| Banned-token check | CI, SPEC §8.4 | libm transcendentals and FMA entering the render path |
| Render-hash diff on {ubuntu, macos} | `scripts/check_renders.sh`, `goldens/renders-v0.1.txt` | cross-machine drift in the WAV bytes |
| Byte-exact event goldens | `goldens/events/*.events.jsonl` | any change to the compiled event lists |
| `SPEC-GAPS.md` | repo root | decisions made without a record |

That is a real correctness setup aimed at agents: the oracle, the
test list and the determinism rules are all outside the code the
agent is allowed to change freely.

## 3 · Why this domain fits

1. **The core is a pure function.** Score text goes in and bytes come
   out. There is no I/O, clock, network or concurrency in the compile
   and render path. Correctness is a property of a function, which is
   the easiest case to verify.
2. **Class A avoids hard math on purpose.** SPEC §1.3: compiling a
   score to events uses exact `i64` fractions, basic IEEE-754
   operations (`+ − × ÷`) and one fully specified decimal rounding
   routine (§12.5). No transcendentals. Everything there can be
   stated precisely, which is rare for float code.
3. **The spec is arbitrary, and that is useful.** "Does it sound
   good?" can't be verified. "Does it match the spec?" can. Music
   separates *validation* (a human listens) from *verification* (a
   machine checks) more cleanly than most domains, and a bug is
   something you can hear, which makes for a good demo.
4. **It's small.** About 1,900 lines in `src/`, the largest module
   (`score.rs`) about 530. That is within reach of real proof tools,
   not just testing.
5. **Determinism is already the product.** "Same score and seed give
   byte-identical WAV on any machine" is the headline feature, so
   verification work lines up with what the project is for.

## 4 · Where the current approach falls short

### 4.1 Goldens are samples, not proofs

Four example scores (212 events in all), 87 PRNG vectors, 42 parser
cases and a set of float probes pin behavior at those points. Any
input they don't cover is unconstrained. A rewrite can pass every
golden and still differ on a score nobody tested.

### 4.2 The spec is not yet complete

`SPEC-GAPS.md` records nine decisions the building agent had to make
because the spec didn't. Examples: what `voice` with no name does
(#3), whether `loop` renders at startup (#5), how a malformed swing
sub-division is handled (#2), how renders beyond the WAV size limit
fail (#9). Each gap is a place where two independent builds could
both be correct and still differ. "Regenerable" requires this list to
be folded back into `SPEC.md` until nothing is left to decide.

### 4.3 Rebuilding would not reproduce the WAV hashes

SPEC §8 says this implementation's synthesizer is "its own design",
and §11.2 says `renders-v0.1.txt` is "this build's own output ... not
a frozen reference vector". A fresh build from the spec would produce
byte-identical event lists (Class A) but different audio. For
regeneration to reproduce the audio, the spec has to pin:

- the sine polynomial: its coefficients, range-reduction steps and
  evaluation order;
- every envelope constant `k = e^(−1/(sr·τ))` as an exact f64 bit
  pattern;
- the order of operations in each voice recipe and in the mix loop.

Then Class C becomes a spec-level contract ("Class A for audio")
instead of a regression pin.

### 4.4 Spec prose isn't machine-checkable

`SPEC.md` is about 1,000 lines of careful English. The goldens check
some of it. The rest (the ordering claims in §1.1, lane indexing in
§6.3, swing in §6.5) is enforced only by whichever tests someone
thought to write.

## 5 · The plan, cheapest first

Each layer is useful alone. Together they move beatcode from "passes
its goldens" to "behavior determined by a machine-checked spec".

### Layer 1 · N-version regeneration and cross-fuzzing

The core experiment for "can the codebase be regenerated".

1. Give two or more agents, in isolated worktrees with no access to
   `src/`, only the seed: `SPEC.md`, `goldens/`, `examples/`,
   `PLAN.md`. Each builds its own implementation.
2. Write a harness that generates random scores (Layer 2's
   generator), runs every implementation on each, and compares
   `bc events` output byte for byte, and accept/reject decisions plus
   the cited line numbers on errors.
3. Sort every disagreement into one of three kinds: a bug in one
   build, a spec gap (becomes a `SPEC-GAPS.md` entry and then a spec
   edit), or an ambiguity in the spec (spec edit).
4. Repeat until independent builds agree on N generated scores with
   no disagreements.

**The measurement:** agreement rate across independent builds is a
number for how complete the spec is. Tracking it over spec revisions
shows whether the spec actually determines the program. This is the
most novel part of the proposal; most spec-driven agent work has no
way to measure spec completeness at all.

Deliverables: `tools/xfuzz/` (harness), `plans/` report of
disagreements found and how each was resolved, updated `SPEC.md`.

### Layer 2 · Generated-input property tests

No new dependencies needed: the crate's own PRNG (`src/prng.rs`) can
drive a score generator in `tests/`. Invariants to check on every
generated score:

| Invariant | Spec |
|---|---|
| Totality: every input either compiles or returns a clean, line-cited error; never a panic | §5.9 |
| Every `performed_s` is finite and ≥ 0 | §2.3, §6.9 |
| Output is sorted by the §6.11 key | §6.11 |
| `grid` values are exact and match the step index × step length | §3, §6.1 |
| Swing at 50% changes nothing; at any amount it moves only odd multiples of the sub | §6.5 |
| Lane values index by time (`floor_i(grid ÷ div)`), not by event count | §6.3 |
| Humanize output depends only on seed, voice and step, not processing order | §4.5, §6.8 |
| Compile and render are deterministic (two runs, same bytes) | §11.1 |
| Every s16 sample is within ±32767; post-normalization peak ≤ 0.98 | §9.5, §9.6 |
| WAV header sizes equal the actual data length | §9.7 |

These extend the six §11.1 properties from fixed inputs to generated
ones. New tests go into `.github/test-manifest.txt` in the same
commit, as CI requires.

### Layer 3 · Bounded proofs with Kani

[Kani](https://github.com/model-checking/kani) is a model checker for
Rust. It proves a property for *every* input within bounds, not just
the sampled ones, and it models IEEE floats bit-precisely. Kani
harnesses live behind `#[cfg(kani)]` and add no runtime dependency.
First targets:

1. **`rational.rs`**: `add`, `mul`, `divr` either return the exact
   reduced result or `Err`; they never overflow silently; results are
   always normalized (den > 0, gcd 1). `floor_i` and `is_int` agree
   with their definitions.
2. **`decfmt.rs`**: `round_dec` matches the §12.5 algorithm for all
   finite `x` in its domain, including sign of zero and ties rounding
   away from zero. `format_dec` output parsed back as a decimal equals
   `round_dec`'s value. This is the routine every event field goes
   through (~1,500 times across the goldens), so a proof here covers
   a lot.
3. **`prng.rs`**: the u64 → f64 conversion stays in [0, 1] (§4.4
   says 1.0 is included).
4. **No panics** in `score::parse` for inputs up to a bounded length.
   Kani checks every indexing, arithmetic overflow and `unwrap`
   reachable from there.
5. **`wav.rs`**: the header's size fields equal `36 + data_size` and
   `data_size` for every frame count within the u32 limit.

CI adds a `cargo kani` job. A proof that stops holding fails the
build, the same way a golden does.

### Layer 4 · An executable reference model

Write the Class A compiler a second time as a *specification*, not an
implementation: simple and obviously correct, speed irrelevant. Two
options:

- **Lean 4.** Write the event compiler as a Lean function, prove the
  §1.1 claims about it (offsets read the pristine grid; the clamp is
  applied once at the end; humanize is order-invariant), and derive
  the goldens from it. The prose spec then becomes a checked one.
  Floats are the hard part: Lean would need an IEEE-754 model (or
  treat Class A float ops as axioms matching IEEE), which is real but
  scoped work.
- **A slow, plain Rust model** in `tests/model/`, differentially
  tested against `src/` through Layer 1's harness. Much cheaper, no
  proofs, but still an independent statement of behavior.

Recommendation: start with the Rust model, since Layer 1 needs a
third opinion anyway. Take on Lean for the ordering and rational-time
claims once Layers 1–3 have settled the spec.

### Layer 5 · Pin the audio path (Class C → spec-level)

Resolve §4.3: move the synth's polynomial coefficients, envelope
constants (as bit patterns) and operation order into `SPEC.md`, and
promote `renders-v0.1.txt` to a frozen golden. Then the render path
is basic operations plus `sqrt`, all exactly specified by IEEE-754,
and Layers 1–3 apply to audio as well as events. What stays a human
judgment is only whether the kit sounds right.

## 6 · Success criteria

The approach has worked when all of these hold:

1. `SPEC-GAPS.md` is empty; every past entry is folded into `SPEC.md`.
2. Two or more independently regenerated implementations agree byte
   for byte on events *and* WAV output over a large generated corpus
   (target: 10⁶ scores with zero disagreements).
3. Kani proofs for `rational`, `decfmt`, the PRNG conversion, parser
   totality and WAV header sizes run green in CI.
4. The Layer 2 invariants hold on every generated score.
5. Optionally: the §1.1 ordering claims are proved in Lean against a
   reference model that produces the Class A goldens.

At that point the claim is precise and defensible:

> beatcode's behavior is fully determined by a machine-checked spec.
> Independent agent-built implementations agree byte for byte, and
> the core arithmetic is proved, not just tested. The spec and its
> checks are the artifact; the code is regenerable.

## 7 · Framing and limits

- **Don't call it a "formally verified music compiler"** unless Layer
  4 is done in Lean. Before that, "spec-determined" or "verified
  against a machine-checked spec" is accurate.
- **What is and isn't proven.** Kani proofs are bounded (input size,
  loop unrolling). The fuzzing layers are strong evidence, not proof.
  The Lean layer proves properties of the model; linking the model to
  `src/` stays differential, not a proof of equivalence.
- **The toolchain is trusted.** Bit-exact claims assume the pinned
  `rustc` and the target CPU follow IEEE-754 for basic operations
  without FMA contraction. That is what the §1.4 rules rely on and
  what the CI matrix checks, but it is an assumption.
- **Aesthetics are out of scope by design.** The spec can be proved
  met; whether the spec makes good music is a separate question, and
  keeping them apart is part of what makes this a clean example.

## 8 · Suggested order of work

| Step | Work | Size |
|---|---|---|
| 1 | Score generator + Layer 2 invariants in `tests/` | small |
| 2 | Kani harnesses for `decfmt` and `rational`; CI job | small–medium |
| 3 | Fold `SPEC-GAPS.md` #1–#9 into `SPEC.md` | small |
| 4 | Two independent regenerations + cross-fuzz harness (Layer 1) | medium |
| 5 | Resolve disagreements; publish the agreement-rate report | medium |
| 6 | Pin the audio path in the spec (Layer 5); freeze render hashes | medium |
| 7 | Rust reference model; later Lean for the §1.1 claims | large |

Steps 1–2 give quick, concrete results. Steps 4–5 are the experiment
that makes this a pioneering example rather than a well-tested
project.
