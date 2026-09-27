# Spec-first in Lean: a research spike on the `beatcode-lean` scenario

The scenario under examination: extract the simplest possible feature of
this repository into a new repository, restart development from formal
specifications in [Lean 4](https://lean-lang.org/), and treat the code as
an ephemeral implementation detail. Four questions were asked of it:

1. How realistic is this?
2. What defines the seam between the side where humans must understand
   and audit the code and the side where they need not?
3. Developers increasingly rubber-stamp agent-written pull requests. Is
   formal verification necessary to pursue?
4. Or will every high-velocity AI-driven system without formal
   verification eventually escape the comprehension of its operators?

Research date: 2026-09-27. Method: five readers mapped the crate module by
module for Lean formalizability; five research sweeps checked the
September-2026 state of Lean's floating point, Rust-to-Lean tooling, Lean
as an implementation language, the evidence on AI code review, and the
track record of spec-first projects (seL4, CompCert, Fiat-Crypto, HACL\*,
CakeML, AWS Cedar, Microsoft SymCrypt), all against fetched primary
sources; three ranking lenses (a verification purist, a product engineer,
a two-week pragmatist) independently ranked the candidate extractions;
five modules were rewritten in Lean 4.34.1, checked against the frozen
goldens and the Rust binary, then re-run and audited by five independent
agents; and finally twenty refuters and a completeness critic attacked a
draft of this document, whose corrections are incorporated. The Lean
sources are in [`lean-spike/`](lean-spike/) and recompile with
`lean-spike/run.sh`. Numbers from this session's container are marked as
such; the machine was four cores, Lean 4.34.1 (released 2026-09-24), core
+ Std only, no Mathlib, no network.

**Terms used below.** The *kernel* is Lean's small trusted proof checker;
`leanchecker` is an independent re-implementation of it that re-verifies
compiled proof files (`.olean`). `decide` asks the kernel to evaluate a
closed proposition; `decide +kernel` does the same bypassing the
elaborator. `#print axioms` lists everything a theorem ultimately trusts;
the three *standard axioms* are `propext`, `Classical.choice` and
`Quot.sound`, and a theorem that lists only those trusts nothing but the
kernel. `native_decide` and `bv_decide` (a SAT-based bit-vector decision
procedure) instead run Lean's *compiled* code and record that trust as an
extra per-theorem axiom named `<theorem>._native.…`. `sorry` is an
admitted gap. `@[extern]`/`@[implemented_by]` mark definitions replaced by
C at run time, which the kernel ignores and the compiled binary trusts.
The Lean FRO is the Lean Focused Research Organization, the maintainers.
The *oracle* is the prior Elixir implementation (running on the BEAM, the
Erlang virtual machine) whose recorded behaviour became this repository's
goldens. SPEC §1.3's classes: **A** = surfaces that must be byte-exact on
every platform (integers, rationals, strings, decimal-rounded floats);
**B** = the oracle's transcendental audio, characterized but not matched;
**C** = this build's own transcendental-free audio, bit-stable across
machines and pinned by committed hashes.

---

## 1 · The short answers

**Realistic for the integer-only modules, which are about half the crate
by lines; five of them are already ported, the parser is not.** In four
builder-hours (plus about the same again of independent audit, roughly
1.2 million subagent tokens in all), `prng`, `decfmt`, `rational`,
`sha256` and the kick/hat/sine slice of `synth` — 423 of the crate's
1,888 Rust lines; the 526-line parser was not attempted — became about
1,400 lines of hand-written Lean (2,700 with generated golden tables and
test drivers). Every scalar value in the 87 PRNG golden vector lines (19
hashes, 64 keys, 64 draw bit patterns, 32 noise bit patterns), all 33
value-rounding and 29 formatter lines of the decimal-rounding goldens
(every such line except the one the spec itself marks expected-to-diverge,
which is pinned as diverging), and four FIPS SHA-256 vectors are now
theorems checked by the kernel with only the three standard axioms, so
none of them trusts Lean's compiler (the million-byte SHA-256 vector and
its bit-identity lemmas do, and are listed separately). The central
rounding algorithm of SPEC §12.5 is proved correct on its stated domain
for every branch; the rational type carries SPEC §3's invariants as fields
and is proved to agree with Lean's core `Rat` on every success path; and
Lean's own float runtime reproduced two of the kit's buffers (kick and
hat) and the sine and note-frequency tables bit-for-bit. Four of the five
modules surfaced a place where the SPEC text says something the code does
not do (Appendix A); none is a code bug that reaches a golden. The audits
were sharper than the builders on the point that matters for the
scenario: none of the five artifacts justifies *deleting* the Rust today,
because the proved region is largely the low-risk region, and because the
Lean definitions a human would now have to read are 1.2–4× longer than
the Rust they replace. A total estimate for the whole restart was not
produced; summing the lenses' per-piece estimates gives roughly five to
eight months of Lean-fluent effort (§6), and the cheaper alternative that
delivers the same verification value without a restart — keep the Rust,
add the Lean model, test them against each other — is weighed in §6.

**The seam is not "code versus spec". It is three conditions, and in this
crate it falls on the IEEE rounding sites and on whether a golden was
derived or recorded.** A component needs no human reading of its code
when (1) its observable behaviour is fixed by a statement short enough to
read and endorse independently of any implementation, (2) the machine
checks the executable definition against that statement with no escape
hatch (no `sorry`, no compiler-trusting axiom, no user-introduced
`@[extern]`/`@[implemented_by]` in the checked path), and (3) the thing
that runs is compiled from the checked definition, or is differentially
tested against it with a spec-determined verdict for every disagreement —
and the testing half of (3) is weaker: a sampled implementation is
verified on the inputs drawn, not made ephemeral. Everything else a human
still reads: the statements and every definition they mention, the
assumption list, the toolchain, the effect interpreter, and the CI
checker. SPEC §1.3's classes approximate this line but the seam cuts
through Class A: its integer surface (`rational`, `prng`, `decfmt`,
`score`) is left of it and the f64 event arithmetic of `events.rs` is
right of it. Lean makes that finer partition a type. A surprise of the
research is that Lean 4.33 (August 2026) moved the line: basic IEEE-754
double arithmetic now reduces in the kernel, so per-operation float facts
are provable, while audio-scale loops, `floor`, `round`, string printing
and everything platform-specific remain on the "test, don't prove" side.

**Formal verification is not necessary in general; an independent,
human-owned statement of intended behaviour that the machine enforces
is. For this repository: yes, pursue it, for the integer and rational
surface first.** Proof is the strongest form of such a statement: the
only form whose checker is a small fixed kernel with an independent
re-implementation rather than a per-project harness (the kernel, the
`#print axioms` gate, `leanchecker`, and the compiler that turns the
definition into a binary still have to be trusted; §8), and the only form
whose *proof* effort now scales with AI-written code rather than with
human attention — statement-review effort does not, and the one interview
study of practitioners finds it goes up. This repository's frozen goldens
plus hardened CI are the rung just below proof, and the evidence says that
rung has a known hole: agents saturate visible tests while hidden-test
gaps widen. The evidence also says proof relocates the audit rather than
removing it: in the largest 2026 industrial case (Microsoft SymCrypt,
16.7 thousand lines of Rust under 237 thousand lines of Lean) humans still
review every theorem statement, and in the best current benchmark
(SWE-Proof, 500 real issues) only 56% of agent-written specifications
survive audit. The case for proof here is about future agent-written
changes, not present defects: the spike found none.

**No, not inevitably. Velocity outruns intent when the only statement of
what the system should do is derived from the code, and that is
avoidable.** Such a statement can no longer distinguish a defect from a
regression without a human re-reading it against purpose, and that
re-reading is exactly what rubber-stamping omits. Mechanism-level
comprehension of this repository is not being exercised (every code
commit is agent-authored; the owner's commits are merges and one spec
explanation; this question was put to an agent) — the 1,888-line crate is
still readable, but nobody is reading it, and at scale that does not come
back, any more than anyone reads a compiler's machine code. What can be
kept is behaviour-level comprehension, and it survives only while the
statements humans read stay small, stable, and written before the code.
The spike shows the mechanism: SPEC.md is partly a transcript of the
oracle, and formalizing a transcript yields theorems that pin one of two
readings the spec leaves open (an accent velocity of 57 rather than 58)
instead of settling which was meant.

---

## 2 · The crate, seen from Lean

Columns: *class* is SPEC §1.3's; *A by integers* means byte-exactness
achieved with integer arithmetic only, *A by IEEE* means achieved by
correctly-rounded f64 operations; *roundings* counts the IEEE-754 rounding
operations between the integer/rational core and the bytes the module
emits.

| module | LOC | class | roundings on its byte surface | goldens | Lean verdict |
|---|---|---|---|---|---|
| `rational.rs` | 106 | A by integers | none (`to_f` excluded) | 212 `grid` strings, 2 overflow tests | maps onto core `Rat` + an i64 window; fully provable |
| `prng.rs` | 73 | A | one (`u64 as f64`), one more in `noise` | 87 vectors | integer chain fully provable; the cast is a 40-line model |
| `decfmt.rs` | 109 | A | one on output (`q'/10^n`), a rounded threshold on input | 99 probe lines | integer core fully provable; float step per-op checkable |
| `score.rs` | 526 | A (no arithmetic) | one (decimal → f64 parse, only at range checks) | 42 accept/reject cases | provable; needs the 25-code-point Unicode whitespace set spelled out |
| `events.rs` + `jsonl.rs` | 303 | A by IEEE | ~14 per event when swing/time/hum/accent are present; none for a straight grid | 4 event files, 6 properties | structure provable over ℚ; bytes need the f64 model or an exact-rational respecification |
| `wav.rs` + `sha256.rs` | 175 | A | none | header hex, FIPS vectors | fully provable; SHA-256 "is FIPS" stays transcription + vectors |
| `synth.rs` + `render.rs` | 342 | C | hundreds per buffer | this build's own hashes | executable in Lean bit-identically; values testable, structure provable |
| `cli.rs` + CI | 213 + YAML | effects | one (`seconds`) | none | the effect interpreter and the checker are what cannot become ephemeral |

Three structural facts fall out of the map.

**The spec is two documents interleaved.** Sections 1.1–1.2, 3, 4 (the
keyed-PRNG thesis), 6.3, 6.5, 6.11 and 7 are intent: a musician or an
engineer can say why they are right. Sections 5.5–5.10, 6.2, 6.4, 6.10,
8 and much of 12 are a transcript of the oracle raised to normative
status: left-to-right float association because that is how the BEAM
evaluated it, an accent rule computed on an f64 product, regexes quoted
verbatim, a sixteen-row catalogue of "match the quirk". A Lean spec of the
first kind yields theorems about music; a Lean spec of the second kind
yields theorems about another program.

**The spec's exactness promise stops at `grid`, and the kernel can now
show where.** SPEC §1.1 promises exact rational beats but defines
`performed_s` as an f64 sum, and §6.4's accent rule is stated in exact
decimals (`115 → 132.25`) without saying whether `1.15` is the f64
literal. Two reachable dyadic ties (values exactly halfway between two
decimals, which only binary fractions can be) show the gap. An accented
hit with lane velocity 50 is `50 × 1.15`: the exact rule gives 57.5 → 58;
the f64 product is 57.4999… → 57 (likewise 90 → 103 not 104, 110 → 126
not 127). At `tempo 128` a grid position of 49/12 beats is exactly
1.9140625 s, which rounds half-away to `1.914063`; the f64 pipeline lands
one ulp (one unit in the last place of the double) below and prints
`1.914062`. Both facts were proved in this container by kernel `decide`
on Lean's float model (`lean-spike/kernel-checks/KernelChecks.lean`) and
reproduced independently by the Rust binary and by Python. None of the
four example scores hits either case, so the goldens do not pin the
divergence; a spec-first restart has to decide which reading is the spec
(§6).

**The goldens smuggle a float parser into the trusted base.** Every
float in the golden files is a shortest-round-trip decimal string, so
even the integer-only modules need a correctly-rounded decimal→binary
conversion to be checked against them. The spike used Python's for the
tables and Lean's `Float.ofScientific` for literals (verified correctly
rounded on all 28 constants `synth.rs` uses, but that is a per-version
empirical fact). A spec-first repository should freeze vectors as
integers or (mantissa, exponent) pairs.

---

## 3 · What Lean 4.34 can actually do, as of this month

The timing of the question is unusually good. Two blockers that would
have settled the answer a year ago were removed this summer.

- **Float has a logical model.** Lean 4.33.0 (2026-08-10) added
  `Float.Model`, an IEEE-754 binary64 model over `UInt64` validated
  against the Berkeley TestFloat and UCBTest suites, and redefined `Float`
  to wrap it. Addition, subtraction, multiplication, division, `sqrt`,
  `abs`, negation, comparisons, integer conversions, `toBits`/`ofBits`
  and decimal literals (`ofScientific`, rewritten to round correctly) now
  reduce in the kernel; compiled code is unchanged. Lean has no `fma`.
  Still opaque: `floor`, `ceil`, `round`, `scaleB`, `frExp`, `toString`,
  `pow`, `cbrt` and every transcendental. Verified here: SPEC §4.4's
  `u64::MAX / 2^64 = 1.0` and its tie band, the two divergences above, a
  Horner sine polynomial's bit pattern, and decfmt's final division for
  two golden rows, all by `decide` with the standard axioms, about 15 s
  in total.
- **Scale is the limit, not expressiveness.** A modelled float operation
  costs the kernel roughly 5–10 ms; a research agent measured 2,000
  dependent multiplies at 18–48 s and 20,000 at out-of-memory on a 15 GB
  machine. The largest event golden (`poly.bc`, 86 events × about 14
  operations ≈ 1,200 float operations plus the PRNG chain) is within
  reach; a 44.1 kHz render (10^5–10^6 operations) is not. The Lean FRO
  says this is by design and that the model-to-native correspondence is
  tested, not proved.
- **The compiler is contraction-free.** `leanc` passes
  `-ffp-contract=off` (added 2026-06-03, present in the 4.34.1 release
  flags), float operations compile to plain C `double` operators, and the
  spike's native binary contained zero fused multiply-add (FMA)
  instructions. This is the same discipline SPEC §1.4 imposes on the Rust
  build, and it is why the Lean audio matched.
- **Compiler trust is visible per theorem.** Since 4.29.0 (2026-03-27)
  each native computation that `native_decide` or `bv_decide` actually
  performs appears in `#print axioms` as its own `<theorem>._native.…`
  axiom (a `bv_decide` call closed by its preprocessing, like the spike's
  `splitmix64_leftInverse`, adds none), so "no theorem depends on compiled
  code" is a mechanical CI gate; `leanchecker`, shipped in the toolchain,
  re-verifies `.olean` files externally. The spike used both.
- **Everything else the crate needs is in core or a pure-Lean package.**
  Exact normalized `Rat`; `UInt64` over `BitVec` with wrapping arithmetic
  and kernel-reducible `decide`; `ByteArray` and binary file I/O; a JSON
  type with exact decimal numbers; a NIST-validated pure-Lean SHA-256
  package exists (`kim-em/lean-crypto-hash`, not end-to-end proved),
  though the spike wrote its own 216-line one and fetched nothing. Native
  performance is adequate: a research agent synthesized 60 s of 44.1 kHz
  audio in 81 ms and hashed a 5 MB WAV in 160 ms; the spike's SHA-256 ran
  at 12 ms per 500 KB natively versus 2 ms in Rust and 2 s interpreted,
  so a 60-second render's receipt costs about a quarter of a second, well
  inside `loop` mode's save-to-sound budget.
- **The Rust-to-Lean extraction route is closed for this crate.**
  Aeneas/Charon, the tool behind Microsoft's SymCrypt verification,
  rejects `f64` outright (open issue since March 2026), crashes on
  `&str` methods, and gives every theorem that unfolds a string literal a
  compiler-trust axiom. So the two viable architectures are the Cedar
  pattern (Lean model, Rust implementation, differential random testing
  with a spec-determined verdict) or Lean as the implementation language.
  Both are practised in production; only the second makes the Rust
  ephemeral, and §6 weighs them.

---

## 4 · The experiment: five modules, five audits

Each builder had about an hour and no Mathlib. Each auditor recompiled
from unmodified copies, ran `#print axioms` on every theorem, tried to
refute every golden check, and answered "would you delete the Rust?".
The load-bearing results:

**`prng`** (129-line `Prng.lean`, 85 min). All 87 vector lines — every
integer and every float bit pattern in `goldens/prng-vectors.jsonl` — are
374 theorems (each vector also checks its seed mask and intermediate
finalizer output) proved by `decide +kernel` in about 12 s, standard
axioms, re-verified by `leanchecker`. `splitmix64` is injective via an
explicit inverse; the SAT-based `bv_decide` could not do the 64-bit
multiplier cancellation in 573 s, and a one-line algebraic proof did. The
noise-prefix keyed property of SPEC §4.6 is a theorem. The `u64 → f64`
cast is modelled explicitly and the "flt reaches exactly 1.0" band is a
theorem about the model; the auditor closed the one gap (model band →
bit pattern of 1.0) in three lines and fuzzed the model against Lean's
`Float` and Python on 3,293 inputs including every tie pattern at every
exponent: zero mismatches. The auditor also wrote an independent Python
implementation from the SPEC text alone that reproduces every golden
value. Surprise: `goldens/README.md` promises "intermediate keys"; the
`flt` vectors carry only final keys.

**`decfmt`** (453-line `Decfmt.lean`, 45 min). The central theorem the
task asked for is proved: on the stated domain (`N < 2^128`, |x| ≤
2^53/10^n) and for every branch (k ≤ 127, k = 128, k ≥ 129) the
top-dropped-bit trick of SPEC §12.5 equals round-half-away-from-zero of
`m·10^n / 2^k`, stated from raw IEEE bits with only the spec's own
hypotheses. All 33 value-rounding and 29 formatter golden lines pass by
kernel `decide`; the one expected-to-diverge input is pinned. The
formatter's shape and integer round-trip are proved. The auditor
re-derived every golden row from an oracle written from the spec text and
confirmed 70 bit-exact comparisons of the float step against Rust, then
found two spec/implementation seams: `decfmt.rs` compares against a
float-rounded domain threshold where §12.5 states an exact rational
(observably harmless), and §6.10's "format-after-round is idempotent"
holds only for `q' < 2^52`: of 200,000 random inputs per precision in
the upper half of the stated domain it fails for 15.8% (two decimals),
2.3% (three) and 4.4% (six) — unreachable by the pipeline's magnitudes,
but stated unconditionally.

**`rational`** (301-line `Rational.lean`, 40 min). `den > 0`, reduced,
and the i64 window are fields of the structure, so a value cannot exist
without them. Every successful `add`/`mul`/`divr` is proved equal to core
`Rat`'s field operation; for a nonzero denominator the only non-success
outcome is `Overflow`; `floor_i` is floor toward −∞ without mentioning
division; commutativity holds on all paths including which error is
returned. The auditor noted the theorems were soundness-only (an
always-`Overflow` implementation satisfies them), proved the converse in
ten lines (`Converse.lean`), and extended the 108-line Rust/Lean table to
a 22,000-case three-way differential (Rust, Lean, an independent Python
big-integer model): byte-identical including 5,895 f64 bit patterns.
Surprises: SPEC §3 says "`checked_*` arithmetic" but `rational.rs` uses
`i128` intermediates and `i64::try_from`; SPEC §3's "`to_f` is correctly
rounded" is false once either operand exceeds 2^53 (`(2^53+1)/7`
double-rounds in Rust, Lean and Python alike — harmless at musical
magnitudes, wrong as stated); and `1/i64::MIN` is an `Overflow` although
both inputs fit, because sign normalization needs +2^63.

**`sha256`** (216-line `Sha256.lean`, 45 min). The FIPS vectors for `""`,
`"abc"`, and the 56- and 112-byte NIST messages are computed by the
kernel at roughly 1.5–2.5 s per 64-byte block; the million-`a` vector
needs `native_decide`. Hex length 64, the padding length facts, and
totality are theorems; the `rotr`/`Ch`/`Maj` identities are proved by
`bv_decide` and therefore carry per-theorem compiler-trust axioms. The
auditor added 219 independent goldens (every length 0–200, random buffers
to 70 KB) and cross-checked the 64 round constants against the prime
cube-root derivation. The auditor's verdict is the honest one for this
module: "the proved properties are exactly the ones nobody doubted; the
property that matters, *is this SHA-256 for all inputs*, is in both
languages a transcription plus test vectors."

**`synthb`** (298-line `SynthB.lean`, 35 min). Lean's `Float` reproduced
the Rust crate's Class C output bit-for-bit on everything tried: 200/200
`sin_p` inputs (plus 20,009 more from the auditor), 128/128 `note_freq`
values (plus 604), and the complete 13,230-sample kick and 3,307-sample
hat buffers with identical SHA-256s, in both the interpreter and a native
build. Snare, clap, pluck, the mixdown and the WAV path were not ported.
Buffer lengths and the noise-prefix property are theorems. One theorem
was rightly marked overstated by the auditor: a cache-transparency model
assumed the synthesized buffer is a function of the cache key, but the
pluck key is `round(freq × 100)` while synthesis uses the unquantized
frequency, and `note_freq(45)` (the note `a2`) = 109.99999999999987 Hz
shares key 11000 with the pitchless 110.0 Hz default — so SPEC §8's "a
cache is order-free" is false for that pair (deterministically so).

Three cross-cutting observations. First, the machine caught the agents'
own mistakes cheaply: `bv_decide` refuted a hand-written xorshift inverse
missing its `>>> 62` term and a wrong modular-inverse literal in seconds;
the kernel refuted this author's off-by-one tie boundary and a mis-stated
threshold theorem. That is the argument for formal checks in one
sentence. Second, the same auditors answered "delete the Rust?" with "not
yet" five times, for reasons that are about the seam and not about Lean:
the proved region is the low-risk region; the Lean artifacts are not
drop-in (no streaming API, no out-of-domain fallback, no NaN path); and
running Lean as the binary moves trust from `rustc` to Lean's compiler,
C runtime and `libm`. Third — and the document's own thesis makes this
the most important caveat — the auditors were agents. No human has read
any theorem statement produced by this spike, so every "is this the right
theorem?" judgement was made by the kind of process §7's evidence says
cannot yet be trusted with it unaided.

---

## 5 · The seam, precisely

Every deployed verified system in the record draws the line in the same
place. seL4's remaining assumptions are hardware, the prover, and "the
spec matches expectations". CompCert's "most delicate issue" is whether
its C semantics match the standard, and six CPU-years of random testing
found bugs only in its unverified front end. Fiat-Crypto's generated
field arithmetic ships in Chrome and Go marked `DO NOT EDIT`, yet
BoringSSL hand-edits the carry chains and marks them "edited after
generation". Project Everest found that generating *readable* C, so
adopters could maintain it without the proofs, was a significant enabler.
Cedar's Lean model is about a tenth the size of its production Rust
(1,673 lines of model and 5,714 of proof against 15,693 of Rust) and is
the document people read; the Rust is kept honest by nightly differential
testing, and no release ships unless model, proofs and tests are current.
In the September-2026 SymCrypt report — 16.7 KLOC of Rust verified via
237 KLOC of Lean, with agents writing 125 KLOC of the proofs — the authors
still say "careful human review of theorem statements remains
essential".

So the audited surface is (i) the theorem statements and every
definition they mention, (ii) the assumption list, including which
theorems rest on `native_decide`, `bv_decide`, user `@[extern]`/
`@[implemented_by]` or foreign-function calls, (iii) the toolchain that
turns the checked definition into the running object, and (iv) the
boundary to anything merely tested. The unaudited region is the proof
terms and any code that is proved to refine the statements. "Code is an
ephemeral implementation detail" is true exactly of that region.

That has a consequence the spike measured. When the checked definition
is also the executable — as in the spike, and in every week-two theorem
of §6, all of which mention `compile` — the definition humans read *is*
the implementation, and the unread region collapses to the proof terms.
The spike's Lean definitions are longer than the Rust they replace
(`Prng.lean` 129 lines against 73, `Decfmt.lean` about 160 lines of
definitions against 109, `Rational.lean` 301 in all against 106). Only
theorems with a statement separate from the definition — `qprime_eq`,
the `Rat` refinement theorems — shrink what a human reads; golden
theorems like `fnv "kick" = 17268634781200901759` do not. Cedar's tenth
comes precisely from keeping model and implementation as separate
objects. Making the executable ephemeral therefore requires a smaller,
spec-level definition separated from it by a refinement proof, which the
spike did not build.

Mapped onto beatcode:

- **Left of the seam today**: the whole ℤ/ℚ/`String`/bytes surface —
  grid strings, step counts, lane indexing, sort keys, JSONL key order
  and escaping, parser accept/reject and cited lines, SHA-256 and the WAV
  header, the PRNG chain. Purity and determinism (SPEC §11.1 P1/P2, the
  `HashMap` and thread bans) become definitional: a Lean `def` is a
  function. The two unreachable `expect`s in `score.rs` become dependent
  pattern matches.
- **Movable with `Float.Model`**: the per-event f64 arithmetic of §6.2,
  the accent product, the `u64 → f64` cast, decfmt's final division. Each
  is now a kernel-checkable operation, so a byte-exact theorem for each
  event golden is feasible. The cost is that the theorem then certifies
  the BEAM's evaluation order. `round` and `floor` are opaque and must be
  re-expressed on integers (the spec's own ties-away rule already is).
- **Right of the seam, and staying there**: render loops (kernel scale),
  the model-to-native correspondence, `libm`'s `floor`, Rust's shortest
  round-trip `Display` fallback, NaN payloads (Lean canonicalizes them),
  the effect interpreter (`play`, `loop`, mtime polling, spawning a
  player), and the CI checker itself (a YAML+shell process assumption:
  the golden manifest and test roster are editable in the same pull
  request, and the banned-token grep is evadable by `f64::sin(x)`).
- **The mechanical marker**: `#print axioms`. A statement that lists a
  `_native` axiom is "verified by running the compiler", the same
  epistemic status as `cargo test` though better labelled; one that lists
  only the standard axioms is verified by a kernel small enough to have
  an independent checker. The spike's `prng` and `decfmt` files are
  entirely in the second class, and so are `rational` and `kernel-checks`
  apart from two deliberately labelled one-line probes of the mechanism
  (`Rat64.floor_i_neg_quarter_native` by `native_decide` beside its
  kernel-`decide` twin, and `sm_step` by `bv_decide`), each of which duly
  shows its own `_native` axiom.

The seam also has a data-dependent edge. For `examples/four.bc` the
exact-rational reading and the f64 pipeline coincide (two refuters
recomputed all 24 events from exact fractions and matched the golden
byte-for-byte), so its golden is left of the seam; for `tempo 128` with a
1/48-note clock they differ in the last printed digit, so the same code is
right of it. Which side a golden falls on depends on whether it is
*derived* from a statement or *recorded* from a run. The render hashes
are self-fingerprints by SPEC's own admission (§11.2: "this build's own
output"); they pin the code, not the spec, and no proof can disagree with
them.

---

## 6 · The simplest possible feature, and the decisions it forces

The three lenses agreed on the shape and disagreed only on the first
commit. Estimates below are in person-days or person-weeks of someone
fluent in Lean 4; the SymCrypt experience suggests agents compress the
proof share of that sharply, and not the statement-review share.

**Week one — what the spike already delivered.** `prng` integer core,
`rational`, and `decfmt`'s integer core: three files, about 880 lines of
Lean, every golden a kernel theorem, differential tables against the
Rust. The pragmatist's reason to start here is that it is a two-day
existence proof with the smallest trusted base in the repository (kernel
`decide` only). The purist's objection is fair: none of these theorems
says anything a musician recognizes. And by the audits' criterion week
one is a lateral move until a statement layer separate from the
executable definitions exists (§5).

**Week two — the feature that tests the thesis.** `bc events` restricted
to what `examples/four.bc` exercises: tempo, bars, per-voice clock, gate,
velocity and pitch lanes, no swing, no time lane, no humanize, no
accents. This is the largest user-facing slice that lies entirely left of
the seam, its 3,673-byte golden is reproducible from ℚ with the spec's
decimal rule, and its first four theorems are the ones that state the
product's design:

- `edit_locality`: the events of voice *j* are a function of voice *j*
  and the four globals, so editing any other voice cannot move them.
  In this subset no `prob` or `hum` draw occurs, so this is a definitional
  consequence of per-voice compilation; the keyed-PRNG thesis of SPEC §4
  becomes a theorem only in the second milestone, when draws exist;
- `lane_time_indexed`: `laneVal lane grid = vals[⌊grid / div⌋ mod len]`,
  the sequencing model of §6.3;
- `straight_order`: for positive tempo the output is in (grid, voice,
  step) order;
- `four_golden`: `eventsJsonl (compile fourScore) = fourGoldenBytes` by
  `decide`, turning SPEC §11.3 item 3 into a checked proposition.

Estimated at 7–10 person-days after week one by the pragmatist; 3–5
person-weeks for a shipped binary (call it `bcl`, the `beatcode-lean`
command) including the parser subset, by the product lens. The full
`bc events` with swing, time lanes, humanize and accents is the second
milestone, not the first: three of the four goldens require the f64
model, signed zero and the non-finite guards, and the pragmatist's
warning deserves quoting: a Lean port mirroring the Rust op order over
`Float` would be byte-identical on all four goldens inside a week, with
differential CI green and *zero theorems that mean anything*, because
the spec would then be the code's evaluation order transcribed into
another language.

**The reading of `performed_s`.** Week two's differential CI is expected
to go red on a small fraction of random scores (the two dyadic-tie
families of §2). That red build is the point: `beatcode-lean` must choose
a reading. The two functional readings are

1. **exact-rational Class A** — `performed_s` is the exact rational
   `grid × 60/tempo` plus exact offsets, rounded half-away to six decimals
   by the integer rule; accents round the exact 23/20 product. The
   theorems are about music; the goldens are re-frozen as integers. All
   four current event goldens still pass (none hits a tie); only new
   tie-hitting scores diverge from the v0.1 crate, whose `1.914062` and
   `57` become documented divergences and whose golden files become
   compatibility fingerprints; or
2. **f64-order-as-spec** — the SPEC §6.2 association is normative and
   every event golden becomes a `Float.Model` theorem. This is what SPEC
   §1.1 and §6.10 already state for `performed_s` (an f64 sum, rounded
   from its exact binary value); it keeps the transcript's evaluation
   order normative and gives up the exact-rational reading that §6.4's
   worked example and the musician's intent suggest.

A third, relational reading — the exact rational plus a proved one-ulp
bound on the f64 chain, with either neighbour admitted at the ties — is
the usual shape of a refinement specification; it is closed here only
because SPEC §2.3 makes the contract byte-exact. The recommendation is
(1), because the scenario's premise is a spec derived from intent, and
because (2) cannot be extended to Class C anyway. Class C in
`beatcode-lean` is then a Lean-compiled renderer (bit-compatible with the
Rust, as the spike showed), with the structural theorems proved and the
bytes covered by the same double-render and cross-OS hash matrix this
repository already runs. Playback is a spawned player process, mtime
polling is an `IO` loop: a small effect shell that is read, not proved.
Users of the binary need no Lean toolchain; developers need about 3 GB
of one, and a clean build of the spike takes 90 s.

**Restart, or keep the Rust?** The Cedar pattern — keep the Rust as the
implementation, add the Lean model and theorems, run differential tests
nightly with the spec text as the verdict — is cheaper (no parser or
renderer port, no new effect shell), keeps a readable implementation in
the language the owner can read, and yields every theorem the restart
would, because the model is the same artifact either way. What it does
not give is the scenario's premise: the Rust stays audited-by-sampling,
not ephemeral. The honest recommendation for this repository is to build
the model first under the Cedar pattern (the spike's week one and the
week-two feature are exactly that model) and to decide about retiring the
Rust only when the Lean definitions plus statements a human reads are
smaller than the Rust they would replace — which, per §5, they are not
yet.

**A total.** No lens estimated the whole restart. Summing their pieces
(parser 2–3 weeks; the full event compiler 1–2 months under reading (1),
an extrapolation, versus 3–5 months under reading (2) with a float model;
renderer 2–3 months to a working binary with its own hashes; effect shell
and CI a week) gives roughly five to eight person-months of Lean-fluent
effort, before agent compression of the proof share.

**Who reads the statements.** The seam collapses if the human who owns
the statements cannot read them. If the owner of `beatcode-lean` does not
read Lean, every theorem statement needs a prose gloss the owner checks
and signs, the statement files need human review on every diff, and the
Cedar-pattern argument for keeping a readable Rust gets stronger. Everest
found readable output was the adoption enabler; the same applies to the
statements.

**CI for the new repository, so the seam is enforced rather than
described:** no `sorry` and no non-standard axiom on any theorem
(`#print axioms` diffed against a committed list); `leanchecker` on every
`.olean`; the theorem-statement files owned by humans with review
required on any diff, in the style of the Lean FRO's `comparator`;
goldens as integer literals; and a differential test against the Rust
binary whose disagreements are triaged by the spec text, never by editing
the spec to match.

---

## 7 · On rubber-stamping, necessity, and comprehension

**What the evidence says about review.** The best randomized controlled
trial still comes from early-2025 tools: METR (a non-profit evaluation
lab) found experienced open-source developers 19% slower with AI
assistance while believing they were 20% faster (16 developers, 246
issues); its late-2025 follow-up, reported 2026-02-24, estimated −18% and
−4% with confidence intervals spanning zero, and METR itself called the
signal unreliable and is redesigning the study. DORA 2025, Google's
annual delivery survey of nearly 5,000 practitioners, finds 90% adoption,
throughput up, delivery stability down, and 30% reporting little or no
trust in generated code. Mining studies of GitHub in 2026 find that most
AI-generated pull requests receive no human review at all and are
reviewed, when reviewed, by other agents (arXiv 2605.02273); in a
manually inspected sample (353 rejected, 364 merged) drawn from 11,048
closed agentic pull requests, a third of the rejections had no observable
rationale and only 15% of merges show a reviewer feedback loop or
reviewer-applied commits (arXiv 2605.22534); AI-to-AI review loops grew
by more than two orders of magnitude in 2025 with a median cross-product
latency of 1.2 minutes (arXiv 2608.21311). A randomized trial at
Anthropic found AI-assisted learners scoring 50% versus 67% on a
comprehension quiz minutes later (52 participants). The phenomenon the
question names is measured, not anecdotal.

**What the evidence says about tests as the spec.** On SpecBench (arXiv
2605.21384, 30 systems tasks up to an OS kernel) every frontier agent
saturates the visible tests while the gap to the hidden ("hold-out")
tests grows about 28 percentage points of pass rate per tenfold increase
in code size, and one agent produced a 2,900-line "compiler" that
memorized the test inputs. This repository's CI is unusually well
defended against that (frozen golden manifest, test-roster manifest,
banned tokens, cross-OS hashes), and the loop-engineering spike in this
repository already argued that a hardened checker is what makes an
unattended "done" trustworthy. But goldens are examples; four event files
and 87 vectors are a finite set; and the checker's own soundness is a
process assumption.

**What the evidence says about proof.** In one year, proof generation on
the VERINA Lean benchmark (arXiv 2505.23135) went from under 5% to
vendor-claimed 97–99%; agentic Claude Code certified 87.5% of the CLEVER
benchmark's implementations (arXiv 2605.23772); SymCrypt's authors report
that verifying the ML-KEM number-theoretic transform initially took about
six months, whereas an additional algorithm now takes about a week of
agent time plus a few days of human review (arXiv 2609.15648);
CryptoProver reproduced an eight-month, five-person verification in 11.4
hours for $467 (arXiv 2608.00965). So the bottleneck moved. On SWE-Proof
(arXiv 2609.21190, 500 real issues), Claude Opus 4.8 resolves 85% of
issues unaided from the issue text and about 95% when handed a correct
formal specification, while a structured natural-language specification
confers none of that improvement — but only 56% of the *agent-written*
formal specifications pass audit, a quarter of test-passing patches had
counterexamples, and the typical failure is a spec that constrains part
of the behaviour and leaves the rest free. Specification hacking by
RL-trained provers is documented (arXiv 2605.30914); an audit of five Lean
benchmarks found 398 mechanically certified defects in the *statements*
(arXiv 2606.29493); a case study of an LLM-driven formalization found
"sorries are not the hard part" — the problematic definitions were (arXiv
2606.13925). Kevin Buzzard, receiving a 13.4-million-line agent-generated
Lean proof of Fermat's Last Theorem this month, put the seam in one reply
(2026-09-08, under his 2026-09-04 post): "No! The machine checks it for
us! That's the beauty of Lean! Someone has to read the *statement* to
check that it corresponds to the right theorem — but I did that." Terence
Tao added the other half (arXiv 2608.16753): "A proof that no human can
properly explain should be viewed as incomplete, even if it has been
formally verified."

**So: is formal verification necessary?** Not as such. What is necessary
is a statement of intended behaviour that (a) a human owns and can read,
(b) is smaller than the code and stable while the code churns, and (c)
the machine enforces. Tests, goldens, contracts, model checking and
proofs are rungs on one ladder ordered by how much of the human's
attention moves from code to statement and by how much of the checker
must itself be trusted. Proof is the top rung on both axes and the only
rung whose proof cost is now paid mostly in tokens. Three findings cut
against treating it as sufficient. Proof size grows quadratically with
statement size (Matichuk et al., ICSE 2015), so the discipline that
matters is keeping statements small, not writing more proofs. The OOPSLA
2025 interview study of experienced Dafny users found that verification
*increases* review burden — specs, implementation, and transpiled output
— and that formal-first reviewers read the specs while engineering-first
reviewers still read the code. And every audit in this spike found the
proved region was the low-risk region: the spike found no code bug, and
every Appendix A item is a spec-text error or an unreachable case. Formal
verification is worth pursuing here because this crate is unusually well
shaped for it (decidable data, byte-exact contracts, a spec that already
partitions by arithmetic class) and because the risk it addresses is the
next agent-written change, not the current code.

**Will velocity without it escape comprehension?** Two kinds of
comprehension have to be separated. Mechanism-level comprehension — how
the code does it — is not being exercised for this repository and for most
agent-built systems, and at scale it does not come back; nobody reads the
machine code a compiler emits either. Behaviour-level comprehension — what
the system is supposed to do, stated independently of what it does — is
the thing that can be kept, and it is weakened in a specific, avoidable
way: when the statement is produced after the fact from the code, by the
same process that wrote the code. Then "what it should do" collapses into
"what it does", the goldens become recordings, and checking against them
can catch regressions but not defects; intent is recoverable only by a
human re-reading the statement against purpose, which is how this spike
found the accent quirk (SPEC §1.1's "exact" survived beside the
transcript) and which unattended review does not do. The way out is not
slower development but an artifact hierarchy with a human at the top: a
short statement file that humans write and diff, a machine-checked
refinement below it, and code below that which nobody needs to read.
Velocity is then bounded by how fast humans can read statements, which
is a bound worth having.

---

## 8 · Honest limits

- The kernel proves facts about `Float.Model`; its agreement with the
  native implementation is tested (about a million TestFloat cases and
  five million parse cases, per the Lean FRO), not proved, and the model
  is six weeks old. The two tie divergences of §2 do not depend on it:
  they were also reproduced by the Rust binary and by Python. Rust's `f64`
  agreement with the model is established here only by the spike's
  differential runs. Every `@[extern]` is an unverified promise.
- Lean's compiler, bundled clang, runtime and `libm` are unverified; a
  theorem about a `def` says nothing checked about its machine code. A
  `native_decide` soundness bug was found in 2023; per-theorem axioms make
  such dependence visible but not absent.
- Kernel float evaluation does not scale to renders; whether the memory
  blow-up at ~2×10^4 dependent operations is inherent was measured on one
  machine only.
- The toolchain moves monthly and the spike felt it: `String` became
  byte-backed in 4.34 (plain `decide` on string literals stopped
  working; `decide +kernel` still does), several core names were renamed
  or deprecated mid-session, and `bv_decide`'s axiom surfaced under a new
  name. Pinning and upgrade policy are real costs; Everest called this
  "changing tires on a moving car".
- No human has read any theorem statement produced by this spike. The
  transcription of §3, §4 and §12.5 into Lean is the new place a mistake
  can hide; the only link to the oracle is the goldens, and the strongest
  evidence of a faithful transcription — an auditor's independent Python
  written from the SPEC text reproducing every PRNG golden — is evidence,
  not proof.
- Nothing here measured review depth directly; the rubber-stamping
  evidence is proxies (no-review rates, latency, suggestion adoption,
  self-report). No randomized study compares defect rates of
  AI-verified versus AI-tested code in production.
- The `beatcode-lean` repository itself was not created or inspected;
  this spike's Lean lives under `docs/lean-spike/` in this repository as
  seed material.

---

## Appendix A · SPEC text versus code, as exposed by the spike

Nine of the eleven are places where SPEC text or `goldens/README.md`
says something the code does not do; items 5 and 11 are restatements a
spec-first restart should make explicit. None affects a committed golden
and none is a code defect.

1. SPEC §3 "`to_f` … is correctly rounded": false when either operand
   exceeds 2^53 (two roundings). Counterexample `(2^53+1)/7`.
2. SPEC §3 "`i64` with `checked_*` arithmetic": `rational.rs` uses `i128`
   intermediates and `i64::try_from`; the module comment repeats the
   wording.
3. SPEC §3 (unstated): `1/i64::MIN`, `3/i64::MIN`, `i64::MIN/−1` are
   `Overflow` errors although both inputs fit in i64.
4. SPEC §6.4 states the accent rule in exact decimal terms
   (`115 → 132.25`, round-half-away) without saying whether `1.15` is the
   f64 literal; the implementation uses the f64 product, giving 57/103/126
   for velocities 50/90/110 where the exact rule gives 58/104/127. The
   rule is under-specified.
5. SPEC §1.1 / §6.10 (not a discrepancy, but worth stating): `performed_s`
   is the f64 pipeline value, not the rounded exact rational; at
   `tempo 128`, grid 49/12 the f64 pipeline prints `1.914062` where the
   exact tie 245/128 would round to `1.914063`.
6. SPEC §6.10 / §7 "format-after-round is idempotent": true only for
   `q' < 2^52`; of 200,000 random inputs per precision in `[2^52, 2^53)`
   it fails for 15.8% (two decimals), 2.3% (three), 4.4% (six).
7. SPEC §12.5: the `k = 128` / `k ≥ 129` branch tests presuppose
   `N < 2^128`, which the text guarantees only implicitly through its
   `u128` note (`2^53 · 10^6 < 2^73`); a spec-first restart should state
   the bound as a hypothesis. Separately, `decfmt.rs`'s domain guard
   compares against a float-rounded threshold (`2^53 as f64 / 10^n as
   f64`) rather than the stated exact rational (observably harmless).
8. SPEC §6.8 "the short-circuit at `prob ≥ 1.0` matters only for avoiding
   the draw": since `flt` equals exactly 1.0 for 1,024 of the 2^64
   possible finalizer outputs, the short-circuit is load-bearing for
   `prob 1.0`; without it a step would be dropped with probability 2^-54.
9. SPEC §8 "A cache is order-free (never iterated), so any map type is
   fine": true for iteration order, but key aliasing (`a2`, midi 45, at
   109.99999999999987 Hz and the pitchless 110.0 Hz default share key
   11000) makes the shared pluck buffer depend on which event is rendered
   first.
10. `goldens/README.md` "key chains (with intermediate keys)": the `flt`
    vectors carry only the final key (one intermediate hop, `fnv("kick|1")`,
    appears incidentally among the `fnv` vectors).
11. SPEC §9.4 defines `frames = last + 22050` where `last` is an index, so
    the silent tail is 22,049 frames; the text already notes the
    "half-second" intent is off by one. A spec-first restart should decide
    which it wants (the intent or the count) and state that one as the
    definition.

## Appendix B · What is in `docs/lean-spike/`

See [`lean-spike/README.md`](lean-spike/README.md) for the per-module
table, the trust marker, and `run.sh`. Everything recompiles on Lean
4.34.1 in about 90 s with no network access; the SHA-256 is self-written;
the Rust harnesses under each `rs/` depend on the crate by relative path.

---

## Sources

**Lean**: lean-lang.org/doc/reference/latest/releases/{v4.22.0, v4.28.0,
v4.29.0, v4.30.0, v4.31.0, v4.33.0, v4.34.0} · reference manual
"Floating-Point Numbers" and "Validating Proofs" · github.com/leanprover/lean4
(`src/Init/Data/Float/{Float,Model}.lean`, `src/Init/Data/OfScientific.lean`,
`src/CMakeLists.txt`, PRs #14079 #14091 #14110, RFC #12216) ·
juliahimmel.de/blog/float-qanda (Lean FRO, 2026-06-19) · lean-lang.org/fro/roadmap/{y3,y4-1} ·
github.com/leanprover/comparator · arXiv 2609.19352 (FloatLib, verified
IEEE-754 library) · github.com/Beneficial-AI-Foundation/FloatSpec ·
github.com/opencompl/fp-lean · github.com/kim-em/lean-crypto-hash ·
github.com/nielsvoss/lean-pitfalls.
**Rust ↔ Lean**: github.com/AeneasVerif/{aeneas,charon} (issues #828, #838,
#1372; charon #142) · arXiv 2609.15648 "Scaling Verification of
Cryptographic Software with Aeneas, Rust, and Lean" (SymCrypt, Sept 2026) ·
github.com/cryspen/hax · cryspen.com/post/strengths-and-limitations ·
github.com/verus-lang/verus · amazon.science (Verus, 2026-08-31) ·
github.com/creusot-rs/creusot · arXiv 2607.01504 (Kani) ·
arXiv 2407.01688 "How We Built Cedar: A Verification-Guided Approach"
(FSE 2024) · cedar-policy.github.io/rfcs/0032 · github.com/cedar-policy/cedar-spec ·
lean-lang.org/use-cases/{cedar,aeneas} · arXiv 2605.30106 (Runtime
Verification Rust-to-Lean report) · github.com/Verified-zkEVM/leanerVM.
**Prior art**: sel4.systems (SOSP 2009; TOCS 2014 "Comprehensive formal
verification of an OS microkernel"; whitepaper; FAQ) · Matichuk et al.,
ICSE 2015 (proof size vs statement size) · xavierleroy.org/publi/compcert-CACM.pdf ·
Regehr et al., PLDI 2011 (Csmith) · Fiat-Crypto, IEEE S&P 2019, and
google/boringssl `third_party/fiat/README.md` · Go CL 14c3d2aa ·
eprint.iacr.org/2017/536 (HACL\*) · project-everest.github.io
(perspectives 2025) · cakeml.org/jfp19.pdf · Lamport, CACM 2015 ·
hillelwayne.com (2019) and Pragmatic Engineer (2026-07-29) ·
martin.kleppmann.com/2025/12/08 · aws.amazon.com/blogs/security (2024-10-17) ·
allthingsdistributed.com (2026-02-17) · ranjitjhala.github.io/static/oopsla25-formal.pdf
(interview study of Dafny users).
**AI code review and verification evidence** (arXiv ids with the claim
each supports): metr.org (2025-07-10 RCT; 2026-02-24 update) ·
cloud.google.com DORA 2025 announcement · infoq.com (DORA ROI report,
2026-05) · faros.ai (2025-07 telemetry) · 2607.01904 (2× merged-PR
mandate, 802 developers) · 2607.13196 (1.02M reviewed PRs, review speed
vs quality) · 2605.02273 (AI PRs mostly unreviewed / agent-reviewed) ·
2605.22534 (11,048 agentic PRs; rejection rationale; merge feedback) ·
2603.15911 (human vs AI reviewer contributions) · 2608.21311 (AI-to-AI
review loops) · 2604.13277 (comprehension-debt diary study) · 2603.28592
(static-analysis issues in AI commits) · 2609.12708 (CQBench human/AI
function pairs) · 2605.21384 (SpecBench) · 2509.22908 (Vericoding) ·
2505.23135 (VERINA) · 2605.23772 (agentic Claude Code on CLEVER) and
2505.13938 (CLEVER) · 2602.09464 (AlgoVeri) · 2608.13522 (Vero) ·
2609.21190 (SWE-Proof) · 2605.26457 (Verus-SpecBench) · 2605.30914
(specification hacking) · 2608.28639 (spurious proofs) · 2606.29493
(Lean benchmark defects) · 2608.14673 (audit of OpenAI's Lean results) ·
2606.13925 (expert review of an LLM formalization) · 2602.20082 (CertiCoq
ANF pass by Claude Code) · 2607.26306 (EVM contract refinement proofs) ·
2605.01660 (Rinard) · 2608.00965 (CryptoProver) · 2608.16753 (Tao,
"Mathematics in the age of AI", ICM 2026) · anthropic.com/research
(AI-assistance-coding-skills; formalizing-fermats-last-theorem) ·
xenaproject.wordpress.com (2026-09-04 post and 2026-09-08 comment) ·
terrytao.wordpress.com (2026-08-18, 2026-09-23) · harmonic.fun ·
logosresearch.ai · sonarsource.com (2026-01-08) · survey.stackoverflow.co/2025 ·
veracode.com (Spring 2026) · gitclear.com.
**This repo**: SPEC.md · PLAN.md · SPEC-GAPS.md · goldens/ · .github/workflows/ci.yml ·
docs/loop-engineering-research.md · docs/lean-spike/.
