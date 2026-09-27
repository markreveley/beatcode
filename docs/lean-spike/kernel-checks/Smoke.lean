import Std.Tactic.BVDecide
open Std

def fnv1a (s : String) : UInt64 :=
  s.toUTF8.foldl (fun h b => (h ^^^ b.toUInt64) * 0x100000001B3) 0xCBF29CE484222325

#eval fnv1a ""
#eval fnv1a "kick"
example : fnv1a "" = 14695981039346656037 := by decide
example : fnv1a "kick" = 17268634781200901759 := by native_decide
example : fnv1a "kick" = 17268634781200901759 := by decide

theorem sm_step (s : UInt64) : (s ^^^ (s >>> 30)) ^^^ (s >>> 30) = s := by bv_decide

#eval (2.675 : Float).toBits
#eval Float.ofBits 0x4005666666666666
#eval (18446744073709551615 : UInt64).toFloat
#eval ((18446744073709551615 : UInt64).toFloat / 18446744073709551616.0)
#eval (18446744073709551615 : Nat).toFloat
#check Float.toBits
#check Float.ofBits
#check Float.ofScientific
#check UInt64.toFloat
#check Float.frExp
#check Float.toString
#print axioms sm_step
