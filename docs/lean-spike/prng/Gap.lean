import Proofs
open Prng
-- Does round_top connect to the executable model's full encoding? (builder only proved model_max_is_one for the single max value)
theorem band_is_one (u : UInt64) (h : 2 ^ 64 - 2 ^ 10 ≤ u.toNat) : fltBitsOfOut u = 0x3FF0000000000000 := by
  unfold fltBitsOfOut
  rw [round_top u.toNat h (UInt64.toNat_lt_size u)]
  decide +kernel
#print axioms band_is_one
-- Does the Float side agree at the band edges? (test only)
#eval (fltOfOut (2^64 - 2^10)).toBits == 0x3FF0000000000000
#eval (fltOfOut (2^64 - 2^10 - 1)).toBits == 0x3FF0000000000000
#eval (fltOfOut (2^64 - 2^10 - 1)).toBits
#eval fltBitsOfOut (2^64 - 2^10 - 1)
-- formatting trust points used by chainStep
#eval toString (18446744073709551615 : UInt64)
#eval toString (-5 : Int)
#eval toString (0 : Int)
#eval maskSeed (2^64)
#eval maskSeed (2^64 + 1)
#eval maskSeed (-(2^64) - 1)
#eval (Prng.Part.int (-5)).render ++ "|" ++ toString (3 : UInt64)
-- does splitmix64_leftInverse's bv_decide really avoid the native axiom? cross-check with bv_normalize
theorem li' (k : UInt64) : splitmix64Inv (splitmix64 k) = k := by
  simp only [splitmix64, splitmix64Inv, inv31, mulB_cancel, inv27, mulA_cancel, inv30]
  bv_normalize
#print axioms li'
-- statement sanity: does fltKey really equal the SPEC fold (Rust flt_key)? Check unfold shape.
example (s : UInt64) : fltKey s [.str "kick", .int 3] = fnv ("3|" ++ toString (fnv ("kick|" ++ toString s))) := rfl
