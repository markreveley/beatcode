import Prng
open Prng

/-! ## (a) fnv of the empty string is the offset basis -/
theorem fnv_empty : fnv "" = fnvOffset := by decide +kernel
theorem fnv_empty' : fnv "" = 14695981039346656037 := by decide +kernel

/-! ## key chain: definitional facts -/
theorem fltKey_nil (s : UInt64) : fltKey s [] = s := rfl
theorem fltKey_cons (s : UInt64) (p : Part) (ps : List Part) :
    fltKey s (p :: ps) = fltKey (fnv (p.render ++ "|" ++ toString s)) ps := rfl

/-! ## (b) noise-prefix property (SPEC §4.6): keyed ⇒ prefix of longer stream = shorter stream -/
theorem noise_prefix (tag : String) (n k : Nat) :
    noise tag n = (noise tag (n + k)).take n := by
  unfold noise
  rw [List.range_add, List.map_append, List.take_left' (by simp)]

theorem noiseBits_prefix (tag : String) (n k : Nat) :
    noiseBits tag n = (noiseBits tag (n + k)).take n := by
  unfold noiseBits
  rw [List.range_add, List.map_append, List.take_left' (by simp)]

/-! ## (c) splitmix64 is injective, via an explicit inverse.
The xorshift inverses are proved by bit extensionality (no SAT, no native axiom);
`bv_decide` also proves them in <1 s but adds a per-theorem `_native.bv_decide` axiom. -/
def mulAInv : UInt64 := 0x96DE1B173F119089
def mulBInv : UInt64 := 0x319642B2D24D8EC3

def invXorShr30 (z : UInt64) : UInt64 := z ^^^ (z >>> 30) ^^^ (z >>> 60)
def invXorShr27 (z : UInt64) : UInt64 := z ^^^ (z >>> 27) ^^^ (z >>> 54)
def invXorShr31 (z : UInt64) : UInt64 := z ^^^ (z >>> 31) ^^^ (z >>> 62)

def splitmix64Inv (out : UInt64) : UInt64 :=
  let z := invXorShr31 out
  let z := invXorShr27 (z * mulBInv)
  let s := invXorShr30 (z * mulAInv)
  s - gamma

theorem inv30 (z : UInt64) : invXorShr30 (xorShr 30 z) = z := by
  unfold invXorShr30 xorShr
  apply UInt64.toBitVec_inj.mp
  simp only [UInt64.toBitVec_xor, UInt64.toBitVec_shiftRight]
  have ha : (UInt64.toBitVec 30 % 64 : BitVec 64).toNat = 30 := by decide
  have hb : (UInt64.toBitVec 60 % 64 : BitVec 64).toNat = 60 := by decide
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.ushiftRight_eq', ha, hb, BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight]
  rw [BitVec.getLsbD_of_ge z.toBitVec (30 + (60 + i)) (by omega)]
  have hab : 30 + (30 + i) = 60 + i := by omega
  rw [hab]
  simp [Bool.xor_assoc, Bool.xor_left_comm]
theorem inv27 (z : UInt64) : invXorShr27 (xorShr 27 z) = z := by
  unfold invXorShr27 xorShr
  apply UInt64.toBitVec_inj.mp
  simp only [UInt64.toBitVec_xor, UInt64.toBitVec_shiftRight]
  have ha : (UInt64.toBitVec 27 % 64 : BitVec 64).toNat = 27 := by decide
  have hb : (UInt64.toBitVec 54 % 64 : BitVec 64).toNat = 54 := by decide
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.ushiftRight_eq', ha, hb, BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight]
  rw [BitVec.getLsbD_of_ge z.toBitVec (27 + (54 + i)) (by omega)]
  have hab : 27 + (27 + i) = 54 + i := by omega
  rw [hab]
  simp [Bool.xor_assoc, Bool.xor_left_comm]
theorem inv31 (z : UInt64) : invXorShr31 (xorShr 31 z) = z := by
  unfold invXorShr31 xorShr
  apply UInt64.toBitVec_inj.mp
  simp only [UInt64.toBitVec_xor, UInt64.toBitVec_shiftRight]
  have ha : (UInt64.toBitVec 31 % 64 : BitVec 64).toNat = 31 := by decide
  have hb : (UInt64.toBitVec 62 % 64 : BitVec 64).toNat = 62 := by decide
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.ushiftRight_eq', ha, hb, BitVec.getLsbD_xor, BitVec.getLsbD_ushiftRight]
  rw [BitVec.getLsbD_of_ge z.toBitVec (31 + (62 + i)) (by omega)]
  have hab : 31 + (31 + i) = 62 + i := by omega
  rw [hab]
  simp [Bool.xor_assoc, Bool.xor_left_comm]

theorem mulA_mul_inv : mulA * mulAInv = 1 := by decide
theorem mulB_mul_inv : mulB * mulBInv = 1 := by decide

theorem mulA_cancel (x : UInt64) : x * mulA * mulAInv = x := by
  rw [UInt64.mul_assoc, mulA_mul_inv, UInt64.mul_one]
theorem mulB_cancel (x : UInt64) : x * mulB * mulBInv = x := by
  rw [UInt64.mul_assoc, mulB_mul_inv, UInt64.mul_one]

theorem splitmix64_leftInverse (k : UInt64) : splitmix64Inv (splitmix64 k) = k := by
  simp only [splitmix64, splitmix64Inv, inv31, mulB_cancel, inv27, mulA_cancel, inv30]
  bv_decide

theorem splitmix64_injective : Function.Injective splitmix64 := by
  intro a b h
  have := congrArg splitmix64Inv h
  rwa [splitmix64_leftInverse, splitmix64_leftInverse] at this

/-! ## §4.4 the u64 → f64 model -/
theorem round_max : roundToF64Nearest (2 ^ 64 - 1) = 2 ^ 64 := by decide

set_option maxRecDepth 100000 in
theorem round_top_bounded : ∀ d, d < 2 ^ 10 → roundToF64Nearest (2 ^ 64 - 2 ^ 10 + d) = 2 ^ 64 := by
  decide +kernel

theorem round_top (x : Nat) (h1 : 2 ^ 64 - 2 ^ 10 ≤ x) (h2 : x < 2 ^ 64) :
    roundToF64Nearest x = 2 ^ 64 := by
  have hx : x = 2 ^ 64 - 2 ^ 10 + (x - (2 ^ 64 - 2 ^ 10)) := by omega
  rw [hx]
  exact round_top_bounded _ (by omega)

/-- the model, not just the Float, says 2^64−1 ↦ exactly 1.0 (bits 0x3FF0…) -/
theorem model_max_is_one : fltBitsOfOut 18446744073709551615 = 0x3FF0000000000000 := by decide +kernel

#print axioms fnv_empty
#print axioms fltKey_cons
#print axioms noise_prefix
#print axioms noiseBits_prefix
#print axioms inv30
#print axioms mulA_cancel
#print axioms splitmix64_leftInverse
#print axioms splitmix64_injective
#print axioms round_max
#print axioms round_top_bounded
#print axioms round_top
#print axioms model_max_is_one
