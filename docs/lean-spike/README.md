# lean-spike — five beatcode modules in Lean 4, with their goldens as theorems

Companion artifact to [`../lean-spec-first-research.md`](../lean-spec-first-research.md).
Each directory is a transparent Lean 4 transcription of one Rust module,
checked against the repo's frozen goldens and against the Rust binary
bit-for-bit, plus the first theorems a spec-first `beatcode-lean` would
want. Written by Claude agents in one session (2026-09-27), each module
then re-run and audited by an independent agent; the audits' additions
(`rational/Converse.lean`, `prng/Gap.lean`) are included.

Toolchain: Lean **4.34.1** (2026-09-24), core + Std only — no Mathlib,
no network. `./run.sh` recompiles everything in dependency order
(about 90 s on four cores); set `LEAN_BIN` to the toolchain's `bin/` if
`lean` is not on `PATH`.

| dir | Rust source | what is kernel-checked (standard axioms only) | what is only tested |
|---|---|---|---|
| `prng/` | `src/prng.rs` (73 LOC) | all 87 `goldens/prng-vectors.jsonl` vectors as 374 theorems (`Vectors.lean`, `decide +kernel`, ~12 s); `fnv "" = offset basis`; noise-prefix property; `splitmix64` injective via an explicit inverse; SPEC §4.4's "`flt` reaches exactly 1.0" as a theorem about an explicit round-to-nearest model (`Proofs.lean`, `Gap.lean`) | that Lean's `UInt64.toFloat` agrees with that model (99 golden values + 3,293 fuzz inputs) |
| `decfmt/` | `src/decfmt.rs` (109 LOC) | the §12.5 top-dropped-bit algorithm equals round-half-away-from-zero of `m·10^n / 2^k` for every branch (`qprime_eq`); 33 value-rounding and 29 formatter golden lines by kernel `decide`; the one expected-to-diverge line pinned; formatter shape and integer round-trip | the final `q'/10^n` float division vs Rust on 70 inputs (also kernel-checkable now, see `kernel-checks/DecfmtFloatStep.lean`); the out-of-domain `Display` fallback is not modelled |
| `rational/` | `src/rational.rs` (106 LOC) | invariants as structure fields; every successful `add`/`mul`/`divr` equals core `Rat`'s field operation; `Overflow` is the only deviation (`mk?_cases`, and the converse in `Converse.lean`); `floor_i` is floor toward −∞; i64 sharp edges by `decide` | agreement with Rust on 108 table lines + a 22,000-case three-way diff (Rust, Lean, independent Python) |
| `sha256/` | `src/sha256.rs` (135 LOC) | FIPS vectors `""`, `"abc"`, 56-byte, 112-byte computed by the kernel (`DecideK_*.lean`, ~1–1.5 s per block); hex length 64; padding length facts; totality | the million-`a` vector (`native_decide`); 500 KB / 2 MB inputs vs Rust and `sha256sum` |
| `synthb/` | `src/synth.rs` (kick, hat, `sin_p`, `note_freq`) + noise | buffer lengths 13230 / 3307; noise length and prefix; xorshift-31 inverse; a cache-transparency model (overstated — see the research doc) | Lean `Float` vs Rust: 200/200 `sin_p` inputs, 128/128 `note_freq`, full kick and hat buffers bit-identical (same sha256s); 28 literals correctly rounded; `-ffp-contract=off`, 0 FMA instructions |
| `kernel-checks/` | — | the facts the write-up leans on, proved in the kernel on Lean 4.34.1 with `Float.Model`: u64::MAX/2^64 = 1.0 and its tie band; `50 × 1.15 < 57.5` (the accent tie); `(49/12)·(60/128) < 1.9140625` (the tempo-128 tie); a Horner sine polynomial bit pattern; decfmt's final division for two golden rows | — |

How to read a directory: the `.lean` file named after the module is the
definition; `Proofs`/`Converse`/`Gap` are theorems about it;
`Vectors`/`Table`/`RatTable` are generated golden tables (the `gen_*.py`
scripts regenerate them from `goldens/`); `rs/` is a scratch Cargo
project depending on the crate by path that prints the Rust side's bit
patterns for the differential comparisons.

Trust marker: `#print axioms` on every theorem. Standard is
`[propext, Classical.choice, Quot.sound]` or fewer. A theorem that lists a
`<name>._native.native_decide.ax_*` or `_native.bv_decide.ax_*` axiom
rests on Lean's compiled code. The sha256 `Tests.lean` vectors, the
`rotr`/`ch`/`maj` identities and `synthb`'s `xorshift31_inv` do, as do two
deliberately labelled one-line probes of the mechanism
(`rational/Rational.lean:Rat64.floor_i_neg_quarter_native`, beside its
kernel-`decide` twin, and `kernel-checks/Smoke.lean:sm_step`); nothing
else in `prng/`, `decfmt/`, `rational/` or `kernel-checks/` does. An
independent refuter re-scanned every constant in every module with
`Lean.collectAxioms` and found exactly those.
