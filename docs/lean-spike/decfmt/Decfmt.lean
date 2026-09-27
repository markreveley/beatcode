/-!
# Decfmt — exact decimal rounding / formatting (beatcode SPEC §6.10, §7, §12.5)

Model: a finite f64 is `±m·2^e` (`F`).  `decompose` reads that off the raw
IEEE-754 bits with pure UInt64/Nat arithmetic.  `roundDecQ` is the integer
core of §12.5 (N = m·10^n, k = −e, top-dropped-bit test).  `qprime` adds the
domain check (spec: |x| ≤ 2^53/10^n, as an exact rational inequality) and the
e > 0 corner.  `roundDecModel` adds the sign-of-zero rule (§6.10 / probe E1).
`formatChars`/`formatDec` are the §7/§12.5 formatter.

The ONLY Float step is `roundDecFloat`: `q'.toFloat / (10^n).toFloat`.
Lean's `Float` is opaque to the kernel, so nothing is proved about it — it is
only #eval-tested against the Rust `bc::decfmt::round_dec` bit patterns.
-/
namespace Decfmt

/-! ## f64 decomposition (bit level, provable) -/

/-- x = (if neg then -1 else 1) · m · 2^e -/
structure F where
  neg : Bool
  m : Nat
  e : Int
deriving Repr, DecidableEq

def expField (bits : UInt64) : UInt64 := (bits >>> 52) &&& 0x7FF
def fracField (bits : UInt64) : UInt64 := bits &&& 0xFFFFFFFFFFFFF
def signBit (bits : UInt64) : Bool := (bits >>> 63) == 1

/-- exponent field all ones = Inf/NaN -/
def isFinite (bits : UInt64) : Bool := expField bits != 0x7FF

/-- SPEC §12.5 / decfmt.rs `decompose`: subnormals (and zero) are `frac·2^-1074`,
normals are `(2^52 ∨ frac)·2^(exp−1075)`. -/
def decompose (bits : UInt64) : F :=
  if expField bits == 0 then
    { neg := signBit bits, m := (fracField bits).toNat, e := -1074 }
  else
    { neg := signBit bits,
      m := (fracField bits ||| 0x10000000000000).toNat,
      e := ((expField bits).toNat : Int) - 1075 }

theorem fracField_toNat_lt (bits : UInt64) : (fracField bits).toNat < 2^52 := by
  unfold fracField
  rw [UInt64.toNat_and]
  exact Nat.and_lt_two_pow _ (by decide)

theorem decompose_m_lt (bits : UInt64) : (decompose bits).m < 2^53 := by
  unfold decompose
  split
  · show (fracField bits).toNat < 2^53
    have := fracField_toNat_lt bits
    omega
  · show (fracField bits ||| 0x10000000000000).toNat < 2^53
    rw [UInt64.toNat_or]
    exact Nat.or_lt_two_pow (by have := fracField_toNat_lt bits; omega) (by decide)

theorem decompose_e_ge (bits : UInt64) : -1074 ≤ (decompose bits).e := by
  unfold decompose
  split
  · show -1074 ≤ (-1074 : Int); omega
  · show -1074 ≤ ((expField bits).toNat : Int) - 1075
    have h : ¬ (expField bits == 0) = true := by assumption
    have h1 : expField bits ≠ 0 := by simpa using h
    have h2 : 0 < (expField bits).toNat := by
      rcases Nat.eq_zero_or_pos (expField bits).toNat with h0 | h0
      · have h0' : (expField bits).toNat = (0 : UInt64).toNat := h0
        exact absurd (UInt64.toNat_inj.mp h0') h1
      · exact h0
    omega

/-! ## The integer core (SPEC §12.5) -/

/-- "r ≥ ½?" — the top-dropped-bit test, exactly as written in §12.5 / decfmt.rs. -/
def roundUpBit (N k : Nat) : Bool :=
  if k = 0 then false
  else if k ≤ 127 then decide ((N >>> (k - 1)) &&& 1 = 1)
  else if k = 128 then decide (N ≥ 2^127)
  else false

/-- q' for N = m·10^n and k = −e ≥ 0: integer part plus the round-up bit. -/
def roundDecQ (N k : Nat) : Nat :=
  (if k < 128 then N >>> k else 0) + (if roundUpBit N k then 1 else 0)

/-- Spec domain |x| ≤ 2^53 / 10^n as an exact rational inequality
(m·2^e·10^n ≤ 2^53).  NOTE: decfmt.rs tests `x.abs() > (2^53 as f64) / (10^n as f64)`,
i.e. against a Float-ROUNDED threshold; see the "surprises" report. -/
def inDomain (f : F) (n : Nat) : Bool :=
  if f.e ≥ 0 then decide (f.m * 2^f.e.toNat * 10^n ≤ 2^53)
  else decide (f.m * 10^n ≤ 2^53 * 2^(-f.e).toNat)

/-- §12.5 `q'`: `none` outside the exactness domain. -/
def qprime (f : F) (n : Nat) : Option Nat :=
  if !inDomain f n then none
  else if f.e > 0 then some ((f.m * 10^n) <<< f.e.toNat)   -- only |x| = 2^53, n = 0
  else some (roundDecQ (f.m * 10^n) (-f.e).toNat)

/-! ## Value rounding (sign-of-zero rule, §6.10 / probe E1) -/

inductive RoundResult
  | keep                        -- out of domain: value kept unchanged (SPEC-GAPS §1, probe E8)
  | zero (neg : Bool)           -- ±0.0; neg = true only for an exactly −0.0 input
  | val (neg : Bool) (q : Nat)  -- sign · q / 10^n  (the division is the untrusted Float step)
deriving Repr, DecidableEq

def roundDecModel (bits : UInt64) (n : Nat) : RoundResult :=
  let f := decompose bits
  match qprime f n with
  | none => .keep
  | some 0 => .zero (f.neg && f.m == 0)
  | some q => .val f.neg q

/-- UNTRUSTED (Float): the one Float step of §12.5, `sign · (q' as f64) / (10^n as f64)`. -/
def roundDecFloat (x : Float) (n : Nat) : Float :=
  match roundDecModel x.toBits n with
  | .keep => x
  | .zero neg => if neg then -0.0 else 0.0
  | .val neg q =>
    let v := q.toFloat / (10^n).toFloat
    if neg then -v else v

/-! ## Formatting (§7 / §12.5): digits of q' padded to n+1, split n from the right -/

/-- LSB-first decimal digits, fuel-structured so `decide` can evaluate it. -/
def digitsAux : Nat → Nat → List Nat
  | 0, _ => []
  | fuel+1, q => if q = 0 then [] else (q % 10) :: digitsAux fuel (q / 10)
def digitsLSB (q : Nat) : List Nat := digitsAux q q
def ofDigitsLSB : List Nat → Nat
  | [] => 0
  | d :: ds => d + 10 * ofDigitsLSB ds

/-- zero-padded (at the MSB end) to at least n+1 digits, LSB first -/
def paddedLSB (q n : Nat) : List Nat :=
  digitsLSB q ++ List.replicate (n + 1 - (digitsLSB q).length) 0
def fracLSB (q n : Nat) : List Nat := (paddedLSB q n).take n
def intLSB (q n : Nat) : List Nat := (paddedLSB q n).drop n
/-- trailing zeros trimmed (LSB first: leading zeros dropped), keeping ≥ 1 digit -/
def fracTrimLSB (q n : Nat) : List Nat :=
  let t := (fracLSB q n).dropWhile (· == 0)
  if t.isEmpty then [0] else t

def digitChar (d : Nat) : Char := Char.ofNat (48 + d)
def charDigit (c : Char) : Nat := c.toNat - 48

def formatChars (neg : Bool) (q n : Nat) : List Char :=
  (if neg then ['-'] else []) ++ (intLSB q n).reverse.map digitChar
    ++ ['.'] ++ (fracTrimLSB q n).reverse.map digitChar

/-- §7 formatter on the model.  `none` outside the domain, where decfmt.rs falls
back to Rust's shortest-round-trip `Display` (not modelled here). -/
def formatDec (bits : UInt64) (n : Nat) : Option String :=
  match qprime (decompose bits) n with
  | none => none
  | some q => some (String.ofList (formatChars (decompose bits).neg q n))

/-- parse MSB-first digit chars back to a Nat -/
def parseChars (cs : List Char) : Nat := ofDigitsLSB (cs.reverse.map charDigit)


/-! ## Theorems: the integer core -/

/-- The heart: `N/2^(j+1) + bit_j(N) = (2N + 2^(j+1)) / 2^(j+2)` (round half up of N/2^(j+1)). -/
theorem core_bit (N j : Nat) :
    N / 2^(j+1) + (N / 2^j) % 2 = (2*N + 2^(j+1)) / 2^(j+1+1) := by
  have hP := Nat.two_pow_pos j
  rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_succ]
  generalize 2^j = P at *
  have h1 : N / (P*2) = N / P / 2 := by rw [Nat.div_div_eq_div_mul]
  have h2 : (2*N + P*2) / (P*2*2) = (N / P + 1) / 2 := by
    rw [← Nat.div_div_eq_div_mul]
    have : 2 * N + P * 2 = 2 * (N + P) := by omega
    have : P * 2 = 2 * P := by omega
    rw [‹2 * N + P * 2 = 2 * (N + P)›, this, Nat.mul_div_mul_left _ _ (by omega : 0 < 2),
      Nat.add_div_right _ hP]
  rw [h1, h2]
  generalize N / P = t
  omega

/-- CENTRAL THEOREM (integer core of §12.5): for every N < 2^128 (the u128 the Rust
uses) and every k, the top-dropped-bit algorithm computes exactly
round-half-away-from-zero(N / 2^k) = ⌊(2N + 2^k) / 2^(k+1)⌋. -/
theorem roundDecQ_eq (N k : Nat) (hN : N < 2^128) :
    roundDecQ N k = (2*N + 2^k) / 2^(k+1) := by
  unfold roundDecQ roundUpBit
  rcases Nat.lt_or_ge k 128 with hk | hk
  · rcases Nat.eq_zero_or_pos k with hk0 | hk0
    · subst hk0
      simp only [Nat.shiftRight_zero, Nat.pow_zero, ↓reduceIte, Nat.zero_lt_succ,
        Bool.false_eq_true]
      omega
    · have hk1 : ¬ k = 0 := by omega
      have hk2 : k ≤ 127 := by omega
      rw [if_pos hk, if_neg hk1, if_pos hk2]
      obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      rw [Nat.add_sub_cancel, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow,
        Nat.and_one_is_mod, ← core_bit N j]
      split <;> rename_i h <;> simp only [decide_eq_true_eq] at h <;> omega
  · have hk' : ¬ k < 128 := by omega
    have hk1 : ¬ k = 0 := by omega
    have hk2 : ¬ k ≤ 127 := by omega
    rw [if_neg hk', if_neg hk1, if_neg hk2]
    rcases Nat.eq_or_lt_of_le hk with hk128 | hk129
    · subst hk128
      rw [if_pos rfl]
      simp only [Nat.reduceAdd, Nat.reducePow, decide_eq_true_eq] at hN ⊢
      split <;> omega
    · have hk3 : ¬ k = 128 := by omega
      rw [if_neg hk3]
      obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
      have hj : 2^128 ≤ 2^j := Nat.pow_le_pow_right (by decide) (by omega)
      simp only [Bool.false_eq_true, ↓reduceIte, Nat.add_zero, Nat.pow_succ]
      have : 2 * N + 2^j * 2 < 2^j * 2 * 2 := by omega
      rw [Nat.div_eq_of_lt this]

theorem m_pow10_lt (m n : Nat) (hm : m < 2^53) (hn : n ≤ 6) : m * 10^n < 2^128 := by
  have h1 : 10^n ≤ 10^6 := Nat.pow_le_pow_right (by decide) hn
  have h2 : m * 10^n ≤ m * 10^6 := Nat.mul_le_mul_left m h1
  have h3 : m * 10^6 < 2^53 * 10^6 := Nat.mul_lt_mul_of_pos_right hm (by decide)
  have h4 : 2^53 * 10^6 < 2^128 := by decide
  omega

/-- CENTRAL THEOREM at the `qprime` level: for finite x = ±m·2^e with e ≤ 0 (k = −e)
in the domain, `qprime` returns exactly round-half-away-from-zero((m·10^n)/2^k). -/
theorem qprime_eq (f : F) (n : Nat) (he : f.e ≤ 0) (hdom : inDomain f n = true)
    (hm : f.m < 2^53) (hn : n ≤ 6) :
    qprime f n = some ((2 * (f.m * 10^n) + 2^(-f.e).toNat) / 2^((-f.e).toNat + 1)) := by
  unfold qprime
  rw [hdom]
  simp only [Bool.not_true, Bool.false_eq_true, if_false]
  rw [if_neg (by omega)]
  rw [roundDecQ_eq _ _ (m_pow10_lt _ _ hm hn)]

/-- Corollary for the real thing: bits → q'. -/
theorem roundDec_bits_eq (bits : UInt64) (n : Nat) (hn : n ≤ 6)
    (he : (decompose bits).e ≤ 0) (hdom : inDomain (decompose bits) n = true) :
    qprime (decompose bits) n =
      some ((2 * ((decompose bits).m * 10^n) + 2^(-(decompose bits).e).toNat)
              / 2^((-(decompose bits).e).toNat + 1)) :=
  qprime_eq _ n he hdom (decompose_m_lt bits) hn

/-! ## (c) Idempotence, integer half: any binary value within half a decimal unit of
q/10^n rounds to q.  (Concretely: x = M·2^-k, N = M·10^n; the hypothesis says
q − ½ ≤ N/2^k < q + ½, i.e. |x·10^n − q| ≤ ½ with the tie only on the low side.)
The Float half — that fl(q/10^n) lands within half an ulp, which is < ½·10^-n
whenever q < 2^52 — is a statement about IEEE division and is NOT provable in Lean
(Float is opaque); it is only tested.  Note the argument needs q < 2^52, not the
spec's 2^53, so §7's "format-after-round is idempotent" is proved only there. -/
theorem roundDecQ_of_close (N k q : Nat) (hN : N < 2^128)
    (hlo : q * 2^(k+1) ≤ 2*N + 2^k) (hhi : 2*N + 2^k < (q+1) * 2^(k+1)) :
    roundDecQ N k = q := by
  rw [roundDecQ_eq N k hN]
  exact Nat.div_eq_of_lt_le hlo hhi

/-! ## Digit lemmas -/

theorem ofDigits_digitsAux (fuel : Nat) : ∀ q, q < 10^fuel → ofDigitsLSB (digitsAux fuel q) = q := by
  induction fuel with
  | zero => intro q h; simp only [Nat.pow_zero] at h; have : q = 0 := by omega
            subst this; rfl
  | succ f ih =>
    intro q h
    unfold digitsAux
    split
    · rename_i h0; subst h0; rfl
    · simp only [ofDigitsLSB]
      rw [Nat.pow_succ] at h
      rw [ih (q/10) (by omega)]
      omega

theorem ofDigits_digitsLSB (q : Nat) : ofDigitsLSB (digitsLSB q) = q :=
  ofDigits_digitsAux q q (Nat.lt_pow_self (by decide))

theorem ofDigits_append (l1 l2 : List Nat) :
    ofDigitsLSB (l1 ++ l2) = ofDigitsLSB l1 + 10^l1.length * ofDigitsLSB l2 := by
  induction l1 with
  | nil => simp [ofDigitsLSB]
  | cons d ds ih =>
    simp only [List.cons_append, ofDigitsLSB, ih, List.length_cons, Nat.pow_succ, Nat.mul_add]
    generalize ofDigitsLSB l2 = X
    generalize 10^ds.length = P
    rw [Nat.mul_comm P 10, Nat.mul_assoc]
    omega

theorem ofDigits_replicate_zero (z : Nat) : ofDigitsLSB (List.replicate z 0) = 0 := by
  induction z with
  | zero => rfl
  | succ z ih => simp [List.replicate_succ, ofDigitsLSB, ih]

theorem ofDigits_padded (q n : Nat) : ofDigitsLSB (paddedLSB q n) = q := by
  unfold paddedLSB
  rw [ofDigits_append, ofDigits_replicate_zero, ofDigits_digitsLSB]
  simp

theorem length_padded (q n : Nat) : n + 1 ≤ (paddedLSB q n).length := by
  unfold paddedLSB
  simp only [List.length_append, List.length_replicate]
  omega

theorem int_frac_split (q n : Nat) :
    ofDigitsLSB (fracLSB q n) + 10^n * ofDigitsLSB (intLSB q n) = q := by
  have h := ofDigits_padded q n
  rw [← List.take_append_drop n (paddedLSB q n), ofDigits_append] at h
  have hl : (List.take n (paddedLSB q n)).length = n := by
    rw [List.length_take]; have := length_padded q n; omega
  rw [hl] at h
  exact h

theorem length_dropWhile_le {α} (p : α → Bool) (l : List α) : (l.dropWhile p).length ≤ l.length := by
  induction l with
  | nil => simp
  | cons a l ih => simp only [List.dropWhile_cons]; split <;> simp <;> omega

theorem mem_of_mem_dropWhile {α} (p : α → Bool) (l : List α) (a : α) :
    a ∈ l.dropWhile p → a ∈ l := by
  induction l with
  | nil => simp
  | cons b l ih =>
    simp only [List.dropWhile_cons]
    split
    · intro h; exact List.mem_cons_of_mem _ (ih h)
    · exact id

theorem ofDigits_dropWhile (l : List Nat) :
    ofDigitsLSB l = 10^(l.length - (l.dropWhile (· == 0)).length) * ofDigitsLSB (l.dropWhile (· == 0)) := by
  induction l with
  | nil => simp [ofDigitsLSB]
  | cons d ds ih =>
    simp only [List.dropWhile_cons]
    split
    · rename_i hd
      have hd0 : d = 0 := by simpa using hd
      subst hd0
      simp only [ofDigitsLSB, List.length_cons]
      have hle := length_dropWhile_le (· == 0) ds
      rw [show ds.length + 1 - (ds.dropWhile (· == 0)).length
            = (ds.length - (ds.dropWhile (· == 0)).length) + 1 by omega, Nat.pow_succ]
      rw [ih, Nat.zero_add]
      ac_rfl
    · simp

/-- (b) at the digit level: int·10^n + frac·10^(n − |frac|) = q. -/
theorem roundtrip_digits (q n : Nat) :
    ofDigitsLSB (intLSB q n) * 10^n
      + ofDigitsLSB (fracTrimLSB q n) * 10^(n - (fracTrimLSB q n).length) = q := by
  have hs := int_frac_split q n
  rw [Nat.mul_comm] at hs
  have hlen : (fracLSB q n).length = n := by
    unfold fracLSB; rw [List.length_take]; have := length_padded q n; omega
  have hd := ofDigits_dropWhile (fracLSB q n)
  unfold fracTrimLSB
  simp only []
  split
  · rename_i he
    have he' : (fracLSB q n).dropWhile (· == 0) = [] := List.isEmpty_iff.mp he
    rw [he'] at hd
    simp only [ofDigitsLSB, Nat.mul_zero] at hd
    simp only [ofDigitsLSB, Nat.zero_mul, Nat.mul_zero, Nat.add_zero]
    omega
  · rw [hlen, Nat.mul_comm] at hd
    omega

/-! ## Char level -/

theorem charDigit_digitChar : ∀ d, d < 10 → charDigit (digitChar d) = d := by decide
theorem digitChar_ne_dot : ∀ d, d < 10 → digitChar d ≠ '.' := by decide

theorem digitsAux_lt (fuel : Nat) : ∀ q, ∀ d ∈ digitsAux fuel q, d < 10 := by
  induction fuel with
  | zero => intro q d h; simp [digitsAux] at h
  | succ f ih =>
    intro q d hd
    unfold digitsAux at hd
    split at hd
    · simp at hd
    · simp only [List.mem_cons] at hd
      rcases hd with rfl | hd
      · exact Nat.mod_lt _ (by decide)
      · exact ih _ _ hd

theorem padded_lt (q n : Nat) : ∀ d ∈ paddedLSB q n, d < 10 := by
  intro d hd
  unfold paddedLSB at hd
  rw [List.mem_append] at hd
  rcases hd with hd | hd
  · exact digitsAux_lt _ _ _ hd
  · rw [List.mem_replicate] at hd; omega

theorem fracTrim_lt (q n : Nat) : ∀ d ∈ fracTrimLSB q n, d < 10 := by
  intro d hd
  unfold fracTrimLSB at hd
  simp only [] at hd
  split at hd
  · simp at hd; omega
  · exact padded_lt q n d (List.mem_of_mem_take (mem_of_mem_dropWhile _ _ _ hd))

theorem int_lt (q n : Nat) : ∀ d ∈ intLSB q n, d < 10 := fun d hd =>
  padded_lt q n d (List.mem_of_mem_drop hd)

theorem parseChars_digits (l : List Nat) (hl : ∀ d ∈ l, d < 10) :
    parseChars (l.reverse.map digitChar) = ofDigitsLSB l := by
  unfold parseChars
  simp only [List.map_reverse, List.reverse_reverse, List.map_map]
  have : List.map (charDigit ∘ digitChar) l = List.map id l :=
    List.map_congr_left (fun d hd => charDigit_digitChar d (hl d hd))
  rw [this, List.map_id]

theorem fracTrim_ne_nil (q n : Nat) : fracTrimLSB q n ≠ [] := by
  unfold fracTrimLSB
  simp only []
  split
  · simp
  · rename_i h; simpa [List.isEmpty_iff] using h

theorem dot_not_mem_digits (l : List Nat) (hl : ∀ d ∈ l, d < 10) :
    '.' ∉ l.reverse.map digitChar := by
  intro h
  rw [List.mem_map] at h
  obtain ⟨d, hd, hdc⟩ := h
  exact digitChar_ne_dot d (hl d (List.mem_reverse.mp hd)) hdc

/-- (a) + (b): the formatted characters are `[sign] ++ intChars ++ '.' :: fracChars` with
fracChars nonempty, no '.' among the digit chars, and parsing the two digit
groups back recovers q' exactly: int·10^n + frac·10^(n − |frac|) = q'. -/
theorem format_shape_roundtrip (neg : Bool) (q n : Nat) :
    ∃ ic fc : List Char,
      formatChars neg q n = (if neg then ['-'] else []) ++ ic ++ '.' :: fc ∧
      fc ≠ [] ∧ '.' ∉ ic ∧ '.' ∉ fc ∧
      parseChars ic * 10^n + parseChars fc * 10^(n - fc.length) = q := by
  refine ⟨(intLSB q n).reverse.map digitChar, (fracTrimLSB q n).reverse.map digitChar, ?_, ?_, ?_, ?_, ?_⟩
  · unfold formatChars; simp [List.append_assoc]
  · simp [fracTrim_ne_nil]
  · exact dot_not_mem_digits _ (int_lt q n)
  · exact dot_not_mem_digits _ (fracTrim_lt q n)
  · rw [parseChars_digits _ (int_lt q n), parseChars_digits _ (fracTrim_lt q n),
      List.length_map, List.length_reverse]
    exact roundtrip_digits q n

/-- The String-level formatter is exactly those characters. -/
theorem formatDec_eq (bits : UInt64) (n q : Nat) (h : qprime (decompose bits) n = some q) :
    formatDec bits n = some (String.ofList (formatChars (decompose bits).neg q n)) := by
  unfold formatDec; rw [h]

#print axioms roundDecQ_eq
#print axioms qprime_eq
#print axioms roundDec_bits_eq
#print axioms roundDecQ_of_close
#print axioms format_shape_roundtrip
#print axioms formatDec_eq
#print axioms decompose_m_lt
#print axioms decompose_e_ge

end Decfmt
