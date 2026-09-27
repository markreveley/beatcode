import Std.Tactic.BVDecide
/-!
# SynthB — the Class B/C float boundary of beatcode, transcribed to Lean 4

Mirrors `src/synth.rs` (sin_p, cos_p, note_freq, kick, hat) and the noise
stream of `src/prng.rs` (SPEC §4, §5.6, §8, §8.4).

Two layers, deliberately separated:
* **Executable Float layer** — `Float` is opaque to the kernel; nothing here
  is provable, only `#eval`-testable against the Rust reference bits.
* **Structural layer** — lengths, prefix property of noise streams, the
  cache being a transparent lookup, and the exact u64→f64 rounding model,
  all stated over Nat/List/UInt64 and PROVED.
-/
namespace SynthB

/-! ## §4 PRNG (byte-exact, Class A) -/

/-- fnv-1a 64 over the UTF-8 bytes of `s` (SPEC §4.1). -/
def fnv (s : String) : UInt64 :=
  s.toUTF8.foldl (fun h b => (h ^^^ b.toUInt64) * 0x100000001B3) 0xCBF29CE484222325

/-- splitmix64 finalizer (SPEC §4.3). -/
def splitmix64 (key : UInt64) : UInt64 :=
  let s := key + 0x9E3779B97F4A7C15
  let z := (s ^^^ (s >>> 30)) * 0xBF58476D1CE4E5B9
  let z := (z ^^^ (z >>> 27)) * 0x94D049BB133111EB
  z ^^^ (z >>> 31)

/-- Key-chain part (SPEC §4.2). -/
inductive Part where
  | str (s : String)
  | int (i : Int)

def Part.render : Part → String
  | .str s => s
  | .int i => toString i

/-- `acc` always renders in unsigned decimal — `toString : UInt64 → String` is unsigned. -/
def fltKey (seed : UInt64) (parts : List Part) : UInt64 :=
  parts.foldl (fun acc p => fnv s!"{p.render}|{acc}") seed

/-- u64 → f64 in [0,1] inclusive (SPEC §4.4). `UInt64.toFloat` is the C
`(double)` cast = round-to-nearest-even (verified in Probe.lean). -/
def flt (seed : UInt64) (parts : List Part) : Float :=
  (splitmix64 (fltKey seed parts)).toFloat / 18446744073709551616.0

/-- Noise stream (SPEC §4.6): keyed on (tag, i) only. -/
def noise (tag : String) (n : Nat) : List Float :=
  let base := fnv s!"sample|{tag}"
  (List.range n).map fun i => flt base [.int (Int.ofNat i)] * 2.0 - 1.0

/-! ## §8.4 pinned transcendentals -/

def SR : Float := 44100.0
def TAU : Float := 6.283185307179586        -- std::f64::consts::TAU
def FRAC_PI_2 : Float := 1.5707963267948966 -- std::f64::consts::FRAC_PI_2
def K_KICK_AMP : Float := 0.999802839117358
def K_KICK_SWEEP : Float := 0.9994332672296815
def K_HAT : Float := 0.9989207857728373

-- Same const-evaluated divisions as the Rust (each is one correctly-rounded IEEE div).
def C3 : Float := -1.0 / 6.0
def C5 : Float := 1.0 / 120.0
def C7 : Float := -1.0 / 5040.0
def C9 : Float := 1.0 / 362880.0
def C11 : Float := -1.0 / 39916800.0
def C13 : Float := 1.0 / 6227020800.0

/-- Pinned sine, identical association to `synth.rs::sin_p`. -/
def sin_p (x : Float) : Float :=
  let u := x * (1.0 / TAU)
  let u := u - (u + 0.5).floor
  let u := if u > 0.25 then 0.5 - u else if u < -0.25 then -0.5 - u else u
  let z := u * TAU
  let z2 := z * z
  z * (1.0 + z2 * (C3 + z2 * (C5 + z2 * (C7 + z2 * (C9 + z2 * (C11 + z2 * C13))))))

def cos_p (x : Float) : Float := sin_p (x + FRAC_PI_2)

/-- Pinned 2^(1/12) (SPEC §5.6). -/
def SEMITONE : Float := 1.0594630943592953

/-- `440 · r^(midi−69)`: multiply up / divide down, one rounding per step. -/
def note_freq (midi : Int) : Float :=
  let k := midi - 69
  if k ≥ 0 then
    Nat.fold k.toNat (fun _ _ f => f * SEMITONE) 440.0
  else
    Nat.fold (-k).toNat (fun _ _ f => f / SEMITONE) 440.0

/-! ## §8 kit buffers -/

/-- A state-threaded sample loop: `step s i = (sample_i, s')`. Structure
(length) is provable; the samples are opaque Floats. -/
def loop {σ : Type} (step : σ → Nat → Float × σ) : Nat → Nat → σ → List Float
  | 0, _, _ => []
  | n + 1, i, s => (step s i).1 :: loop step n (i + 1) (step s i).2

def KICK_LEN : Nat := 13230 -- trunc(0.30 · 44100)
def HAT_LEN : Nat := 3307   -- trunc(0.075 · 44100)

/-- `kick()` from synth.rs. State = (ph, sweep, env). -/
def kick : List Float :=
  let clickNoise := (noise "kick-click" 40).toArray
  loop (fun (st : Float × Float × Float) i =>
      let (ph, sweep, env) := st
      let f := 44.0 + sweep
      let ph := ph + TAU * f / SR
      let click := match clickNoise[i]? with
        | some n => n * 0.35 * (1.0 - i.toFloat / 40.0)
        | none => 0.0
      (sin_p ph * env * 0.95 + click, (ph, sweep * K_KICK_SWEEP, env * K_KICK_AMP)))
    KICK_LEN 0 (0.0, 76.0, 1.0)

/-- `hat()` from synth.rs. State = (y1, x1, env); noise consumed by index. -/
def hat : List Float :=
  let n := (noise "hat" HAT_LEN).toArray
  loop (fun (st : Float × Float × Float) i =>
      let (y1, x1, env) := st
      let x := n[i]?.getD 0.0
      let y := 0.92 * (y1 + x - x1)
      (y * 0.7 * env, (y, x, env * K_HAT)))
    HAT_LEN 0 (0.0, 0.0, 1.0)

/-! ## Structural theorems (provable: no Float value is inspected) -/

theorem loop_length {σ} (step : σ → Nat → Float × σ) (n i : Nat) (s : σ) :
    (loop step n i s).length = n := by
  induction n generalizing i s with
  | zero => simp [loop]
  | succ n ih => simp [loop, ih]

theorem kick_length : kick.length = 13230 := by
  unfold kick; exact loop_length _ _ _ _

theorem hat_length : hat.length = 3307 := by
  unfold hat; exact loop_length _ _ _ _

theorem noise_length (tag : String) (n : Nat) : (noise tag n).length = n := by
  simp [noise]

/-- SPEC §4.6 keyed property: a prefix of a longer stream IS the shorter stream. -/
theorem noise_prefix (tag : String) (n m : Nat) :
    (noise tag (n + m)).take n = noise tag n := by
  simp only [noise]
  rw [← List.map_take, List.take_range, Nat.min_eq_left (Nat.le_add_right n m)]

/-- Determinism of the stream given the model: element i depends only on (tag, i). -/
theorem noise_getElem (tag : String) (n i : Nat) (h : i < n) :
    (noise tag n)[i]'(by simp [noise_length]; exact h)
      = flt (fnv s!"sample|{tag}") [.int i] * 2.0 - 1.0 := by
  simp [noise]

theorem fnv_empty : fnv "" = 0xCBF29CE484222325 := by decide

-- fnv("kick") golden from SPEC §4.1
theorem fnv_kick : fnv "kick" = 17268634781200901759 := by decide

-- splitmix64's final xorshift is invertible. NOTE: my first draft claimed the
-- inverse was `y ^^^ (y >>> 31)`; bv_decide produced the counterexample
-- z = 2^64−1 in under a second. The correct inverse needs the third term.
theorem xorshift31_inv (z : UInt64) :
    let y := z ^^^ (z >>> 31); (y ^^^ (y >>> 31) ^^^ (y >>> 62)) = z := by
  intro y; bv_decide

/-- Hence splitmix64 is injective on the last step; full injectivity would also
need odd-multiply invertibility (mod-2^64 inverse), stated here for the record. -/
theorem odd_mul_inv (z : UInt64) :
    z * 0xBF58476D1CE4E5B9 * 0x96DE1B173F119089 = z := by
  -- bv_decide times out here (64-bit multiplier bit-blasting); algebra instead.
  rw [UInt64.mul_assoc]
  have h : (0xBF58476D1CE4E5B9 : UInt64) * 0x96DE1B173F119089 = 1 := by decide
  rw [h, UInt64.mul_one]

/-! ### The memo cache is lookup-only (SPEC §8): a transparent cache -/

inductive Key where
  | sample (k : Fin 4)
  | pluck (centi : Int)
  deriving DecidableEq

/-- Abstract synthesizer for a key; `synth` is the *specification* of the buffer. -/
structure KitModel where
  synth : Key → List Float
  cache : Key → Option (List Float)

def KitModel.wellFormed (m : KitModel) : Prop :=
  ∀ k b, m.cache k = some b → b = m.synth k

def KitModel.buffer (m : KitModel) (k : Key) : List Float × KitModel :=
  match m.cache k with
  | some b => (b, m)
  | none =>
    let b := m.synth k
    (b, { m with cache := fun k' => if k' = k then some b else m.cache k' })

/-- Cache transparency: a well-formed kit always returns the specified
buffer and stays well-formed, so the cache can never change the audio. -/
theorem KitModel.buffer_spec (m : KitModel) (hm : m.wellFormed) (k : Key) :
    (m.buffer k).1 = m.synth k ∧ (m.buffer k).2.wellFormed := by
  unfold KitModel.buffer
  cases h : m.cache k with
  | some b => exact ⟨hm k b h, hm⟩
  | none =>
    refine ⟨rfl, fun k' b' hb' => ?_⟩
    dsimp only at hb'
    split at hb'
    · rename_i heq; subst heq; injection hb' with hb'; exact hb'.symm
    · exact hm k' b' hb'

/-- The cached-synth function never changes under `buffer`. -/
theorem KitModel.buffer_synth (m : KitModel) (k : Key) : (m.buffer k).2.synth = m.synth := by
  unfold KitModel.buffer; cases m.cache k <;> rfl

/-! ### Exact model of the u64 → f64 conversion (SPEC §4.4) -/

/-- Round a Nat to 53 significant bits, ties-to-even (the IEEE cast). -/
def roundSig53 (n : Nat) : Nat :=
  let bits := Nat.log2 n + 1
  if bits ≤ 53 then n else
    let sh := bits - 53
    let q := n >>> sh
    let r := n % (1 <<< sh)
    let half := 1 <<< (sh - 1)
    let q := if r > half ∨ (r = half ∧ q % 2 = 1) then q + 1 else q
    q <<< sh

set_option maxRecDepth 100000 in
/-- Every u64 ≥ 2^64 − 2^10 rounds UP to 2^64, so `flt` reaches exactly 1.0. -/
theorem round_top_range : ∀ k, k < 1024 → roundSig53 (2^64 - 1024 + k) = 2^64 := by
  decide +kernel

/-- Just below that window, the cast lands on 2^64 − 2^11 (< 1.0 after division). -/
theorem round_below_window : roundSig53 (2^64 - 1025) = 2^64 - 2048 := by decide

/-- Exact tie 2^64 − 3·2^10 goes to even: 2^64 − 2^12. -/
theorem round_tie_even : roundSig53 (2^64 - 3072) = 2^64 - 4096 := by decide

/-! ## Executable dump for golden comparison -/

def hex16 (u : UInt64) : String :=
  let s := Nat.toDigits 16 u.toNat
  String.ofList (List.replicate (16 - s.length) '0' ++ s)

def sinInputs : List Float :=
  ((List.range 181).map fun i => -100.0 + i.toFloat * (100.0 / 90.0)) ++
  [0.0, -0.0, 1e-300, 1.5707963267948966, 3.141592653589793,
   6.283185307179586, 1e6, 1e10, 1e15, 1e16, 1e17, 1e20, 1e300,
   -1e300, 4503599627370496.0, 9007199254740992.0,
   0.7853981633974483, 123456.789, -98765.4321]

def main : IO Unit := do
  let h ← IO.FS.Handle.mk "lean_out.txt" .write
  h.putStrLn "SIN"
  for x in sinInputs do
    h.putStrLn s!"{hex16 x.toBits} {hex16 (sin_p x).toBits}"
  h.putStrLn "NOTE"
  for m in List.range 128 do
    h.putStrLn s!"{m} {hex16 (note_freq m).toBits}"
  h.putStrLn s!"KICK len={kick.length}"
  h.putStrLn s!"HAT len={hat.length}"
  h.putStrLn "KICKBITS"
  for x in kick do h.putStrLn (hex16 x.toBits)
  h.putStrLn "HATBITS"
  for x in hat do h.putStrLn (hex16 x.toBits)
  -- model-vs-Float agreement for the u64→f64 cast on the boundary window and some random keys
  let mut bad := 0
  for k in List.range 4096 do
    let u : UInt64 := UInt64.ofNat (2^64 - 4096 + k)
    let modelBits := (roundSig53 u.toNat).toFloat.toBits  -- roundSig53 result is exactly representable
    if u.toFloat.toBits != modelBits then bad := bad + 1
  for k in List.range 4096 do
    let u := splitmix64 (UInt64.ofNat k)
    if u.toFloat.toBits != (roundSig53 u.toNat).toFloat.toBits then bad := bad + 1
  h.putStrLn s!"CASTMODEL mismatches={bad} of 8192"
  h.flush

end SynthB

#print axioms SynthB.loop_length
#print axioms SynthB.kick_length
#print axioms SynthB.hat_length
#print axioms SynthB.noise_length
#print axioms SynthB.noise_prefix
#print axioms SynthB.noise_getElem
#print axioms SynthB.fnv_empty
#print axioms SynthB.fnv_kick
#print axioms SynthB.xorshift31_inv
#print axioms SynthB.odd_mul_inv
#print axioms SynthB.KitModel.buffer_spec
#print axioms SynthB.KitModel.buffer_synth
#print axioms SynthB.round_top_range
#print axioms SynthB.round_below_window
#print axioms SynthB.round_tie_even

def main : IO Unit := SynthB.main

