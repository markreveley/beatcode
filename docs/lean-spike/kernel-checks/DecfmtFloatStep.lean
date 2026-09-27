set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
-- decfmt's final Float step is now kernel-checkable too (Float.Model):
-- round_dec(2.675,2): q'=267 → 267/100 must be the f64 nearest 2.67
theorem d1 : ((267 : Nat).toFloat / (100 : Nat).toFloat).toBits = (2.67 : Float).toBits := by decide
-- round_dec(0.0078125,6): q'=7813 → 7813/10^6 = f64 nearest 0.007813
theorem d2 : ((7813 : Nat).toFloat / (1000000 : Nat).toFloat).toBits = (0.007813 : Float).toBits := by decide
-- Rust's domain guard constant is a rounded Float division, not the exact rational (n=2)
#print axioms d1
#print axioms d2
