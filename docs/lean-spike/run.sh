#!/usr/bin/env bash
# Re-check the Lean spike. Needs a Lean 4.34.x toolchain: set LEAN_BIN to its bin/ dir (or have lean on PATH).
# Usage: ./run.sh            (all modules)   |   ./run.sh prng   (one module)
set -euo pipefail
cd "$(dirname "$0")"
LEAN_BIN="${LEAN_BIN:-$(dirname "$(command -v lean)")}"
LIB="$(cd "$LEAN_BIN/../lib/lean" && pwd)"
run() { local dir=$1; shift; echo "== $dir"; ( cd "$dir"; mkdir -p build
  for f in "$@"; do printf '   %-22s' "$f"; s=$(date +%s.%N)
    LEAN_PATH="$PWD/build:$LIB" timeout 900 "$LEAN_BIN/lean" -o "build/${f%.lean}.olean" "$f" > "build/${f%.lean}.log" 2>&1 && ok=ok || ok=FAIL
    printf '%s  %.1fs\n' "$ok" "$(echo "$(date +%s.%N) - $s" | bc)"; [ "$ok" = ok ] || { cat "build/${f%.lean}.log"; exit 1; }
  done ) }
sel="${1:-all}"
[ "$sel" = all ] || [ "$sel" = prng ]     && run prng     Prng.lean Proofs.lean Vectors.lean Conj.lean Gap.lean
[ "$sel" = all ] || [ "$sel" = decfmt ]   && run decfmt   Decfmt.lean Table.lean Test.lean FloatTest.lean
[ "$sel" = all ] || [ "$sel" = rational ] && run rational Rational.lean RatTable.lean Converse.lean
[ "$sel" = all ] || [ "$sel" = sha256 ]   && run sha256   Sha256.lean Tests.lean DecideK_abc.lean DecideK_nist56.lean DecideK_nist112.lean DecideBytesK.lean
[ "$sel" = all ] || [ "$sel" = synthb ]   && run synthb   SynthB.lean Probe.lean
[ "$sel" = all ] || [ "$sel" = kernel-checks ] && run kernel-checks Smoke.lean KernelChecks.lean DecfmtFloatStep.lean
echo "all green"
