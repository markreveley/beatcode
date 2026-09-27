import Std.Tactic.BVDecide
/-!
# beatcode PRNG (SPEC §4) — executable Lean model

Everything here is a transparent, kernel-reducible definition (no
`implemented_by`, no `extern`), except `flt`/`noise` which go through
Lean's opaque `Float` and exist only to be *tested* (`#eval`) against the
integer/bit model `fltBits`/`noiseBits`.
-/

namespace Prng

/-! ## §4.1 fnv-1a (64-bit) over UTF-8 bytes -/

def fnvOffset : UInt64 := 0xCBF29CE484222325
def fnvPrime  : UInt64 := 0x100000001B3

def fnvStep (h : UInt64) (b : UInt8) : UInt64 := (h ^^^ b.toUInt64) * fnvPrime

def fnvBytes (bs : List UInt8) : UInt64 := bs.foldl fnvStep fnvOffset

/-- fnv-1a over the UTF-8 bytes of a string (Lean strings *are* UTF-8 byte arrays). -/
def fnv (s : String) : UInt64 := fnvBytes s.toUTF8.toList

/-! ## §4.2 the string-mixed key chain -/

inductive Part where
  | str (s : String)
  | int (i : Int)
deriving Repr, DecidableEq

/-- strings verbatim (no quotes); integers in decimal with `-` if negative. -/
def Part.render : Part → String
  | .str s => s
  | .int i => toString i          -- Int.repr: "-5", "0", "41"

/-- `acc` renders in UNSIGNED decimal: `toString (u : UInt64)` = `Nat.repr u.toNat`. -/
def chainStep (acc : UInt64) (p : Part) : UInt64 :=
  fnv (p.render ++ "|" ++ toString acc)

def fltKey (seed : UInt64) (parts : List Part) : UInt64 :=
  parts.foldl chainStep seed

/-! ## §4.3 splitmix64 finalizer (one-shot) -/

def gamma : UInt64 := 0x9E3779B97F4A7C15
def mulA  : UInt64 := 0xBF58476D1CE4E5B9
def mulB  : UInt64 := 0x94D049BB133111EB

def xorShr (s : UInt64) (z : UInt64) : UInt64 := z ^^^ (z >>> s)

def splitmix64 (key : UInt64) : UInt64 :=
  let s := key + gamma
  let z := xorShr 30 s * mulA
  let z := xorShr 27 z * mulB
  xorShr 31 z

/-! ## seed masking (SPEC §2.1): i128-ish → u64 two's complement -/

def maskSeed (seed : Int) : UInt64 := UInt64.ofNat (Int.emod seed (2 ^ 64)).toNat

/-! ## §4.4 u64 → f64: explicit Nat model of round-to-nearest-even to 53 bits -/

/-- fuel-based `log2` (structural, so the kernel can evaluate it). -/
def log2F : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n => if n ≥ 2 then log2F fuel (n / 2) + 1 else 0

def log2' (n : Nat) : Nat := log2F 64 n

/-- Round `x < 2^64` to the nearest multiple of `2^(log2 x − 52)` (53 significant
bits), ties to even. This is exactly what IEEE-754 `u64 → f64` does, and the
result may be `2^64` itself. -/
def roundToF64Nearest (x : Nat) : Nat :=
  if x < 2 ^ 53 then x else
  let s    := log2' x - 52          -- number of low bits dropped (≥ 1)
  let q    := x / 2 ^ s
  let r    := x % 2 ^ s
  let half := 2 ^ (s - 1)
  let q'   := if r < half then q else if half < r then q + 1
              else if q % 2 = 0 then q else q + 1
  q' * 2 ^ s

/-- IEEE-754 binary64 bit pattern of the value `(-1)^neg · m · 2^e` where `m`
already has ≤ 53 significant bits and is nonzero and the result is a normal
number (all our values are). `m = 0` encodes ±0. -/
def f64Bits (neg : Bool) (m : Nat) (e : Int) : UInt64 :=
  let sign : Nat := if neg then 1 <<< 63 else 0
  if m = 0 then UInt64.ofNat sign else
  let k := log2' m                          -- m ∈ [2^k, 2^(k+1))
  let mant := if k ≤ 52 then m <<< (52 - k) else m >>> (k - 52)
  let biased := (Int.ofNat k + e + 1023).toNat
  UInt64.ofNat (sign + (biased <<< 52) + (mant % (2 ^ 52)))

/-- bit pattern of `flt` = `round(out) / 2^64`, purely integer. -/
def fltBitsOfOut (out : UInt64) : UInt64 :=
  f64Bits false (roundToF64Nearest out.toNat) (-64)

def fltBits (seed : UInt64) (parts : List Part) : UInt64 :=
  fltBitsOfOut (splitmix64 (fltKey seed parts))

/-- bit pattern of `flt · 2 − 1` (§4.6): exact value `(r − 2^63)/2^63`, rounded once. -/
def noiseBitsOfOut (out : UInt64) : UInt64 :=
  let r : Int := roundToF64Nearest out.toNat
  let n : Int := r - 2 ^ 63
  f64Bits (decide (n < 0)) (roundToF64Nearest n.natAbs) (-63)

def noiseKey (tag : String) (i : Nat) : UInt64 :=
  fltKey (fnv ("sample|" ++ tag)) [.int i]

def noiseBits (tag : String) (n : Nat) : List UInt64 :=
  (List.range n).map fun i => noiseBitsOfOut (splitmix64 (noiseKey tag i))

/-! ## the Float-facing API (opaque to the kernel; test-only) -/

def two64 : Float := 18446744073709551616.0

def fltOfOut (out : UInt64) : Float := out.toFloat / two64

def flt (seed : UInt64) (parts : List Part) : Float :=
  fltOfOut (splitmix64 (fltKey seed parts))

def noiseAt (tag : String) (i : Nat) : Float :=
  flt (fnv ("sample|" ++ tag)) [.int i] * 2.0 - 1.0

def noise (tag : String) (n : Nat) : List Float :=
  (List.range n).map (noiseAt tag)

end Prng
