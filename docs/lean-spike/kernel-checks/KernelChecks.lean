set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
-- SPEC §4.4: u64::MAX as f64 / 2^64 = 1.0 (kernel, Float.Model)
theorem g1 : (18446744073709551615 : UInt64).toFloat / 18446744073709551616.0 = (1.0 : Float) := by decide
-- exact midpoint 2^64-2^10 rounds up (ties-to-even) → 1.0; one below stays < 1.0
theorem g3 : (18446744073709550592 : UInt64).toFloat / 18446744073709551616.0 = (1.0 : Float) := by decide
theorem g4 : (18446744073709550591 : UInt64).toFloat / 18446744073709551616.0 < (1.0 : Float) := by decide
-- accent tie: f64 50*1.15 < 57.5, so f64-round gives 57 while exact half-away gives 58
theorem acc : ((50 : Float) * 1.15) < 57.5 := by decide
-- tempo 128, grid 49/12: f64 pipeline lands below the exact dyadic tie 1.9140625
theorem t128 : ((49 : Float) / 12.0) * (60.0 / 128.0) < 1.9140625 := by decide
def C3 : Float := -1.0 / 6.0
def C5 : Float := 1.0 / 120.0
def C7 : Float := -1.0 / 5040.0
def C9 : Float := 1.0 / 362880.0
def C11 : Float := -1.0 / 39916800.0
def C13 : Float := 1.0 / 6227020800.0
def poly (z : Float) : Float :=
  let z2 := z * z
  z * (1.0 + z2 * (C3 + z2 * (C5 + z2 * (C7 + z2 * (C9 + z2 * (C11 + z2 * C13))))))
theorem p1 : (poly 0.5).toBits = 4602308182625945072 := by decide
#print axioms g1
#print axioms g4
#print axioms acc
#print axioms t128
#print axioms p1
