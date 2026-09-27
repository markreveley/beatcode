/-!
# Rational.lean — SPEC §3 "Exact rational time", Lean-first.

The mathematical object is a rational number (core `Rat`).  The SPEC's
invariants (`den > 0`, reduced by gcd) are carried *in the type*: a value of
`Rat64` cannot exist without them.  The `i64` width the Rust chose is a
*refinement* (`inI64`), also carried in the type, so `Rat64 → Int64` is
total.  The constructor `mk?` is the only place a value is born, and the
only ways it can fail are `ZeroDen` and `Overflow` (proved: `mk?_cases`).
-/

/-- The `i64` window.  SPEC §3 "Width": an implementation refinement, not
the mathematical intent.  Written with literals so `omega`/`decide` see it. -/
def inI64 (x : Int) : Prop := -9223372036854775808 ≤ x ∧ x ≤ 9223372036854775807

instance : DecidablePred inI64 := fun _ => inferInstanceAs (Decidable (_ ∧ _))

inductive RatError where
  | ZeroDen
  | Overflow
deriving DecidableEq, Repr

def RatError.msg : RatError → String
  | .ZeroDen => "rational division by zero"
  | .Overflow => "rational overflow"

/-- SPEC §3 representation with its invariants as fields. -/
structure Rat64 where
  num : Int
  den : Int
  den_pos : 0 < den
  reduced : Int.gcd num den = 1
  num_i64 : inI64 num
  den_i64 : inI64 den
deriving Repr, DecidableEq

namespace Rat64

theorem ext {a b : Rat64} (hn : a.num = b.num) (hd : a.den = b.den) : a = b := by
  cases a; cases b; simp only at hn hd; subst hn; subst hd; rfl

/-! ## Normalization: pure mathematics, no width -/

/-- SPEC §3 `new`: divide by the non-negative gcd, then carry the sign in the
numerator.  `gcd 0 d = |d|` so `0/d ↦ 0/1` falls out. -/
def normPair (n d : Int) : Int × Int :=
  let g : Int := Int.gcd n d
  if d / g < 0 then (-(n / g), -(d / g)) else (n / g, d / g)

theorem normPair_den_pos {n d : Int} (hd : d ≠ 0) : 0 < (normPair n d).2 := by
  have h1 : d / (Int.gcd n d : Int) ≠ 0 := Int.ediv_gcd_ne_zero_of_ne_zero_right n hd
  simp only [normPair]
  split <;> simp only <;> omega

theorem normPair_reduced {n d : Int} (hd : d ≠ 0) :
    Int.gcd (normPair n d).1 (normPair n d).2 = 1 := by
  have h := Int.gcd_ediv_gcd_ediv_gcd_of_ne_zero_right (n := n) hd
  simp only [normPair]
  split <;> simp [h]

/-- The semantic core: normalization preserves the rational value
(cross-multiplication). -/
theorem normPair_cross (n d : Int) : (normPair n d).1 * d = n * (normPair n d).2 := by
  simp only [normPair]
  by_cases hg : (Int.gcd n d : Int) = 0
  · simp [hg]
  · obtain ⟨n', hn'⟩ := Int.gcd_dvd_left n d
    obtain ⟨d', hd'⟩ := Int.gcd_dvd_right n d
    generalize hG : (Int.gcd n d : Int) = g at *
    subst hn'; subst hd'
    simp only [Int.mul_ediv_cancel_left _ hg]
    split <;> simp [Int.mul_comm, Int.mul_left_comm, Int.mul_neg]

/-! ## The constructor -/

/-- SPEC §3 `new` with the Rust `make` semantics: zero denominator is
`ZeroDen`; after reduction and sign normalization, anything outside `i64`
is `Overflow`.  The invariants are discharged *here*, once. -/
def mk? (n d : Int) : Except RatError Rat64 :=
  if hd : d = 0 then .error .ZeroDen
  else
    let p := normPair n d
    if h : inI64 p.1 ∧ inI64 p.2 then
      .ok { num := p.1, den := p.2
            den_pos := normPair_den_pos hd
            reduced := normPair_reduced hd
            num_i64 := h.1, den_i64 := h.2 }
    else .error .Overflow

/-! ## Operations (SPEC §3), each re-normalized through `mk?` -/

def add (a b : Rat64) : Except RatError Rat64 := mk? (a.num * b.den + b.num * a.den) (a.den * b.den)
def mul (a b : Rat64) : Except RatError Rat64 := mk? (a.num * b.num) (a.den * b.den)
def divr (a b : Rat64) : Except RatError Rat64 := mk? (a.num * b.den) (a.den * b.num)
/-- Floor division toward −∞.  `Int./` is Euclidean division; with `den > 0`
it coincides with floor (proved below). -/
def floor_i (r : Rat64) : Int := r.num / r.den
def is_int (r : Rat64) : Bool := r.den == 1
def to_s (r : Rat64) : String := s!"{r.num}/{r.den}"
/-- The only rational→float edge.  Opaque to the kernel; tested, not proved. -/
def to_f (r : Rat64) : Float := Float.ofInt r.num / Float.ofInt r.den

/-! ## (a) Invariants hold by construction -/

theorem mk?_den_pos {n d : Int} {r : Rat64} (_ : mk? n d = .ok r) : 0 < r.den := r.den_pos
theorem mk?_reduced {n d : Int} {r : Rat64} (_ : mk? n d = .ok r) : Int.gcd r.num r.den = 1 := r.reduced
theorem mk?_i64 {n d : Int} {r : Rat64} (_ : mk? n d = .ok r) : inI64 r.num ∧ inI64 r.den :=
  ⟨r.num_i64, r.den_i64⟩

/-! ## (b) `0/d` normalizes to `0/1` -/

def zero : Rat64 := ⟨0, 1, by decide, by decide, by decide, by decide⟩

theorem normPair_zero {d : Int} (hd : d ≠ 0) : normPair 0 d = (0, 1) := by
  simp only [normPair, Int.gcd_zero_left, Int.zero_ediv]
  generalize ha' : (d.natAbs : Int) = a
  have ha : a ≠ 0 := by omega
  have h : d = a ∨ d = -a := by omega
  rcases h with rfl | rfl
  · simp [Int.ediv_self ha]
  · simp [Int.neg_ediv_self a ha]

theorem mk?_zero {d : Int} (hd : d ≠ 0) : mk? 0 d = .ok zero := by
  simp [mk?, hd, normPair_zero hd, inI64, zero]

theorem mk?_zero_den {n : Int} : mk? n 0 = .error .ZeroDen := by simp [mk?]

/-! ## (c) floor -/

theorem floor_i_eq_fdiv (r : Rat64) : r.floor_i = Int.fdiv r.num r.den :=
  (Int.fdiv_eq_ediv_of_nonneg _ (Int.le_of_lt r.den_pos)).symm

/-- `floor_i r` is the greatest integer `≤ num/den`. -/
theorem floor_i_spec (r : Rat64) :
    r.floor_i * r.den ≤ r.num ∧ r.num < (r.floor_i + 1) * r.den :=
  ⟨Int.ediv_mul_le _ (Int.ne_of_gt r.den_pos), Int.lt_ediv_add_one_mul_self _ r.den_pos⟩

deriving instance DecidableEq for Except

theorem floor_i_neg_quarter : (mk? (-1) 4).map floor_i = .ok (-1) := by decide

/-! ## (d) Semantics: agreement with the unbounded rational `Rat` -/

def toQ (r : Rat64) : Rat := Rat.divInt r.num r.den

theorem mk?_ok_cross {n d : Int} {r : Rat64} (h : mk? n d = .ok r) :
    d ≠ 0 ∧ r.num * d = n * r.den := by
  simp only [mk?] at h
  split at h
  · exact absurd h (by simp)
  · rename_i hd
    split at h
    · simp only [Except.ok.injEq] at h
      subst h
      exact ⟨hd, normPair_cross n d⟩
    · exact absurd h (by simp)

/-- Whenever `mk?` succeeds, the value is exactly the rational `n/d`. -/
theorem mk?_toQ {n d : Int} {r : Rat64} (h : mk? n d = .ok r) : toQ r = Rat.divInt n d := by
  obtain ⟨hd, hc⟩ := mk?_ok_cross h
  unfold toQ
  rw [Rat.divInt_eq_divInt_iff (Int.ne_of_gt r.den_pos) hd]
  exact hc

/-- Completeness: for a nonzero denominator the ONLY two outcomes are the
exact rational, or `Overflow`. No third behaviour exists. -/
theorem mk?_cases (n : Int) {d : Int} (hd : d ≠ 0) :
    (∃ r, mk? n d = .ok r ∧ toQ r = Rat.divInt n d) ∨ mk? n d = .error .Overflow := by
  by_cases h : ∃ r, mk? n d = .ok r
  · obtain ⟨r, hr⟩ := h
    exact .inl ⟨r, hr, mk?_toQ hr⟩
  · right
    simp only [mk?, hd, dite_false] at h ⊢
    split at h
    · exact absurd ⟨_, rfl⟩ h
    · rename_i hh; simp [hh]

theorem add_ok {a b r : Rat64} (h : add a b = .ok r) : toQ r = toQ a + toQ b := by
  unfold add at h
  rw [mk?_toQ h, toQ, toQ,
      Rat.divInt_add_divInt _ _ (Int.ne_of_gt a.den_pos) (Int.ne_of_gt b.den_pos)]

theorem mul_ok {a b r : Rat64} (h : mul a b = .ok r) : toQ r = toQ a * toQ b := by
  unfold mul at h
  rw [mk?_toQ h, toQ, toQ, Rat.divInt_mul_divInt]

theorem divr_ok {a b r : Rat64} (h : divr a b = .ok r) : toQ r = toQ a / toQ b := by
  unfold divr at h
  rw [mk?_toQ h, toQ, toQ, Rat.div_def, Rat.inv_divInt, Rat.divInt_mul_divInt]

theorem divr_zero (a b : Rat64) (hb : b.num = 0) : divr a b = .error .ZeroDen := by
  simp [divr, mk?, hb]

/-- The three ops never fail with `ZeroDen` on nonzero inputs and never
silently deviate: success is exact, failure is `Overflow`. -/
theorem add_cases (a b : Rat64) :
    (∃ r, add a b = .ok r ∧ toQ r = toQ a + toQ b) ∨ add a b = .error .Overflow := by
  rcases mk?_cases (a.num * b.den + b.num * a.den)
      (Int.mul_ne_zero (Int.ne_of_gt a.den_pos) (Int.ne_of_gt b.den_pos)) with ⟨r, hr, _⟩ | h
  · exact .inl ⟨r, hr, add_ok hr⟩
  · exact .inr h

theorem mul_cases (a b : Rat64) :
    (∃ r, mul a b = .ok r ∧ toQ r = toQ a * toQ b) ∨ mul a b = .error .Overflow := by
  rcases mk?_cases (a.num * b.num)
      (Int.mul_ne_zero (Int.ne_of_gt a.den_pos) (Int.ne_of_gt b.den_pos)) with ⟨r, hr, _⟩ | h
  · exact .inl ⟨r, hr, mul_ok hr⟩
  · exact .inr h

theorem divr_cases (a b : Rat64) (hb : b.num ≠ 0) :
    (∃ r, divr a b = .ok r ∧ toQ r = toQ a / toQ b) ∨ divr a b = .error .Overflow := by
  rcases mk?_cases (a.num * b.den) (Int.mul_ne_zero (Int.ne_of_gt a.den_pos) hb) with ⟨r, hr, _⟩ | h
  · exact .inl ⟨r, hr, divr_ok hr⟩
  · exact .inr h

/-- `Rat64` is literally core `Rat`'s canonical form: `toQ` preserves the
fields.  So `Rat64` is a *subset* of `Rat` (the `i64` window), not a
coarser quotient. -/
theorem toQ_num_den (r : Rat64) : (toQ r).num = r.num ∧ ((toQ r).den : Int) = r.den := by
  unfold toQ
  have hpos := r.den_pos
  rcases Int.eq_nat_or_neg r.den with ⟨k, hk | hk⟩
  · have hk0 : k ≠ 0 := by omega
    have hc : Nat.gcd r.num.natAbs k = 1 := by
      have := r.reduced
      rw [Int.gcd_eq_natAbs_gcd_natAbs, hk] at this
      simpa using this
    rw [hk, Rat.divInt_ofNat, ← Rat.normalize_eq_mkRat hk0, Rat.normalize_eq_mk' r.num k hk0 hc]
    exact ⟨rfl, rfl⟩
  · omega

theorem toQ_injective {a b : Rat64} (h : toQ a = toQ b) : a = b := by
  obtain ⟨ha1, ha2⟩ := toQ_num_den a
  obtain ⟨hb1, hb2⟩ := toQ_num_den b
  apply Rat64.ext
  · rw [← ha1, ← hb1, h]
  · rw [← ha2, ← hb2, h]

/-! ## (e) Commutativity on the success path (in fact on all paths) -/

theorem add_comm (a b : Rat64) : add a b = add b a := by
  unfold add; rw [Int.add_comm, Int.mul_comm a.den b.den]

theorem mul_comm (a b : Rat64) : mul a b = mul b a := by
  unfold mul; rw [Int.mul_comm a.num b.num, Int.mul_comm a.den b.den]

/-! ## Printing helpers for the golden diff against Rust -/

def showR : Except RatError Rat64 → String
  | .ok r => s!"ok {r.to_s}"
  | .error .ZeroDen => "err ZeroDen"
  | .error .Overflow => "err Overflow"

def showBin (op : Rat64 → Rat64 → Except RatError Rat64) :
    Except RatError Rat64 → Except RatError Rat64 → String
  | .ok a, .ok b => showR (op a b)
  | _, _ => "BADOPERAND"

def hex16 (x : UInt64) : String :=
  let s := String.ofList (Nat.toDigits 16 x.toNat)
  "".pushn '0' (16 - s.length) ++ s

def showUn : Except RatError Rat64 → String
  | .ok q => s!"floor={q.floor_i} int={q.is_int} s={q.to_s} f=0x{hex16 q.to_f.toBits}"
  | _ => "BADOPERAND"

end Rat64

#print axioms Rat64.mk?
#print axioms Rat64.mk?_zero
#print axioms Rat64.floor_i_eq_fdiv
#print axioms Rat64.floor_i_spec
#print axioms Rat64.floor_i_neg_quarter
#print axioms Rat64.mk?_toQ
#print axioms Rat64.mk?_cases
#print axioms Rat64.add_ok
#print axioms Rat64.mul_ok
#print axioms Rat64.divr_ok
#print axioms Rat64.add_cases
#print axioms Rat64.mul_cases
#print axioms Rat64.divr_cases
#print axioms Rat64.divr_zero
#print axioms Rat64.add_comm
#print axioms Rat64.mul_comm
#print axioms Rat64.toQ_num_den
#print axioms Rat64.toQ_injective

-- Which evaluation route the concrete checks accept (reported in the result):
example : (Rat64.mk? (-1) 4).map Rat64.floor_i = .ok (-1) := by rfl
example : (Rat64.mk? (-1) 4).map Rat64.floor_i = .ok (-1) := by decide +kernel
theorem Rat64.floor_i_neg_quarter_native : (Rat64.mk? (-1) 4).map Rat64.floor_i = .ok (-1) := by native_decide
#print axioms Rat64.floor_i_neg_quarter_native

-- i64 edges, by kernel `decide`:
theorem Rat64.min_over_neg_one_overflows : Rat64.mk? (-9223372036854775808) (-1) = .error .Overflow := by decide
theorem Rat64.max_over_min_overflows : Rat64.mk? 9223372036854775807 (-9223372036854775808) = .error .Overflow := by decide
theorem Rat64.min_over_min_is_one : (Rat64.mk? (-9223372036854775808) (-9223372036854775808)).map Rat64.to_s = .ok "1/1" := by decide
theorem Rat64.six_over_neg_nine : (Rat64.mk? 6 (-9)).map Rat64.to_s = .ok "-2/3" := by decide
#print axioms Rat64.min_over_neg_one_overflows
#print axioms Rat64.six_over_neg_nine

