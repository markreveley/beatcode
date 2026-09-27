import Rational
open Rat64
/-- The direction `mk?_cases` does NOT state: when the normalized pair fits the window, `mk?` succeeds.
Without it, `fun _ _ => .error .Overflow` would also satisfy `mk?_cases`. -/
theorem mk?_ok_of_fits {n d : Int} (hd : d ≠ 0)
    (h : inI64 (normPair n d).1 ∧ inI64 (normPair n d).2) : ∃ r, mk? n d = .ok r := by
  simp only [mk?, hd, dite_false]
  rw [dif_pos h]
  exact ⟨_, rfl⟩
theorem mk?_overflow_iff {n d : Int} (hd : d ≠ 0) :
    mk? n d = .error .Overflow ↔ ¬ (inI64 (normPair n d).1 ∧ inI64 (normPair n d).2) := by
  constructor
  · intro h hf
    obtain ⟨r, hr⟩ := mk?_ok_of_fits hd hf
    rw [hr] at h; cases h
  · intro hf
    simp only [mk?, hd, dite_false]
    rw [dif_neg hf]
#print axioms mk?_ok_of_fits
#print axioms mk?_overflow_iff
