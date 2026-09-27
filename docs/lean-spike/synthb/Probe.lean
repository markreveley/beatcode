-- literal parsing probe: print bits of every f64 literal used in synth.rs
def lits : List (String × Float) := [
  ("TAU", 6.283185307179586), ("FRAC_PI_2", 1.5707963267948966),
  ("K_KICK_AMP", 0.999802839117358), ("K_KICK_SWEEP", 0.9994332672296815),
  ("K_HAT", 0.9989207857728373), ("SEMITONE", 1.0594630943592953),
  ("SR", 44100.0), ("TWO64", 18446744073709551616.0),
  ("C3", -1.0/6.0), ("C5", 1.0/120.0), ("C7", -1.0/5040.0), ("C9", 1.0/362880.0),
  ("C11", -1.0/39916800.0), ("C13", 1.0/6227020800.0),
  ("0.35", 0.35), ("0.95", 0.95), ("0.92", 0.92), ("0.7", 0.7), ("0.25", 0.25), ("0.5", 0.5),
  ("0.30", 0.30), ("0.075", 0.075), ("44.0", 44.0), ("76.0", 76.0), ("40.0", 40.0), ("440.0", 440.0), ("2.0", 2.0), ("1.0", 1.0) ]
def hex16 (u : UInt64) : String :=
  let s := (Nat.toDigits 16 u.toNat)
  String.mk (List.replicate (16 - s.length) '0' ++ s)
#eval lits.map fun (n, f) => s!"{n} {hex16 f.toBits}"
#eval (UInt64.ofNat (2^64-1)).toFloat / 18446744073709551616.0 == 1.0
#eval hex16 ((UInt64.ofNat (2^64-1)).toFloat.toBits)
#eval hex16 ((UInt64.ofNat (2^64-1024)).toFloat.toBits)
#eval hex16 ((UInt64.ofNat (2^64-1025)).toFloat.toBits)
#eval hex16 ((UInt64.ofNat (2^64-3*1024)).toFloat.toBits)  -- tie: 2^64-3072 -> even -> 2^64-4096
#eval hex16 ((2.7 : Float).floor.toBits)
#eval hex16 ((-2.7 : Float).floor.toBits)
#eval (1e300 : Float).floor
#eval Float.ofScientific 1 true 300 == 1e-300
#check @List.take_range
#check @List.length_range
#check @Nat.fold
#eval toString (UInt64.ofNat (2^64-1))
#eval s!"{(-3 : Int)}|{(5:UInt64)}"
