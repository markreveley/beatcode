import Std.Tactic.BVDecide
/-!
# SHA-256 (FIPS 180-4) in core Lean 4

Transparent implementation: no `implemented_by`, no `partial`, no
`unsafe`, no `sorry`, no Mathlib. Fixed-size data is carried in
`Vector` so that the well-formedness facts (8-word state, 64-word
schedule, 32-byte digest, 64-char hex) are structural.

**Nothing about cryptographic correctness is proved.** What is proved:
  * the hex receipt always has exactly 64 characters;
  * the padding rule (§5.1.1): padded length ≡ 0 (mod 64), ≥ len+9;
  * the shift-based ROTR / Ch / Maj primitives equal their FIPS
    definitions (bit-blasted via `bv_decide`);
  * schedule / compress / sha256 are total (structural recursion only;
    `#print axioms` shows no `sorryAx`).
The FIPS 180-4 test vectors, checked below with `#eval` and
`native_decide`, are the ONLY evidence that the hash is SHA-256 — the
same evidence the Rust implementation has.
-/
namespace Sha256

/-- FIPS 180-4 §4.2.2 round constants. -/
def K : Vector UInt32 64 := #v[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

/-- FIPS 180-4 §5.3.3 initial hash value. -/
def H0 : Vector UInt32 8 := #v[
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

/-! ## §4.1.2 primitives (shift-based, as in the Rust) -/

/-- ROTR^n(x) for 0 < n < 32, via two shifts (UInt32 shifts are mod 32). -/
@[inline] def rotr (x : UInt32) (n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))

@[inline] def ch  (e f g : UInt32) : UInt32 := (e &&& f) ^^^ ((~~~e) &&& g)
@[inline] def maj (a b c : UInt32) : UInt32 := (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)
@[inline] def bigSigma0 (a : UInt32) : UInt32 := rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22
@[inline] def bigSigma1 (e : UInt32) : UInt32 := rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25
@[inline] def smallSigma0 (x : UInt32) : UInt32 := rotr x 7 ^^^ rotr x 18 ^^^ (x >>> 3)
@[inline] def smallSigma1 (x : UInt32) : UInt32 := rotr x 17 ^^^ rotr x 19 ^^^ (x >>> 10)

/-! ## §5.1.1 padding -/

/-- Number of zero bytes between the 0x80 byte and the 8-byte length. -/
def zeroCount (len : Nat) : Nat := (119 - len % 64) % 64

/-- Length of the padded message, as a function of the input length. -/
def paddedLen (len : Nat) : Nat := len + 1 + zeroCount len + 8

/-- Big-endian bytes of the 64-bit bit-length. -/
def lenBytes (bitLen : UInt64) : Vector UInt8 8 :=
  Vector.ofFn fun (i : Fin 8) => (bitLen >>> (8 * (7 - i.val.toUInt64))).toUInt8

def pad (m : ByteArray) : ByteArray :=
  let bitLen : UInt64 := m.size.toUInt64 * 8
  ((m.push 0x80) ++ ⟨Array.replicate (zeroCount m.size) 0⟩) ++ ⟨(lenBytes bitLen).toArray⟩

/-! ## §6.2.2 message schedule and compression -/

/-- Big-endian 32-bit word at byte offset `i` (out of range reads as 0; never
    happens for in-range callers but keeps the function total). -/
@[inline] def be32 (m : ByteArray) (i : Nat) : UInt32 :=
  (m.get! i).toUInt32 <<< 24 ||| (m.get! (i+1)).toUInt32 <<< 16 |||
  (m.get! (i+2)).toUInt32 <<< 8 ||| (m.get! (i+3)).toUInt32

/-- W_0..W_63 for the 64-byte block starting at byte offset `off` of `m`. -/
def schedule (m : ByteArray) (off : Nat) : Vector UInt32 64 := Id.run do
  let mut w : Vector UInt32 64 := Vector.replicate 64 0
  for h : i in [0:16] do
    have hi : i < 64 := by have : i < 16 := h.upper; omega
    w := w.set i (be32 m (off + 4 * i))
  for h : i in [16:64] do
    have hi : i < 64 := by have : i < 64 := h.upper; omega
    let v := w[i-16] + smallSigma0 w[i-15] + w[i-7] + smallSigma1 w[i-2]
    w := w.set i v
  return w

/-- One compression: 64 rounds then feed-forward. -/
def compress (st : Vector UInt32 8) (w : Vector UInt32 64) : Vector UInt32 8 := Id.run do
  let mut a := st[0]; let mut b := st[1]; let mut c := st[2]; let mut d := st[3]
  let mut e := st[4]; let mut f := st[5]; let mut g := st[6]; let mut h := st[7]
  for hi : i in [0:64] do
    have : i < 64 := hi.upper
    let t1 := h + bigSigma1 e + ch e f g + K[i] + w[i]
    let t2 := bigSigma0 a + maj a b c
    h := g; g := f; f := e; e := d + t1
    d := c; c := b; b := a; a := t1 + t2
  return #v[st[0]+a, st[1]+b, st[2]+c, st[3]+d, st[4]+e, st[5]+f, st[6]+g, st[7]+h]

/-- Final state as 32 big-endian bytes. -/
def digestBytes (st : Vector UInt32 8) : Vector UInt8 32 :=
  Vector.ofFn fun (i : Fin 32) =>
    (st[i.val / 4]'(by omega) >>> (8 * (3 - (i.val % 4)).toUInt32)).toUInt8

/-- SHA-256 digest of `m`. -/
def sha256 (m : ByteArray) : Vector UInt8 32 :=
  let p := pad m
  let nblocks := p.size / 64
  let st := Nat.fold nblocks (fun b _ st => compress st (schedule p (64 * b))) H0
  digestBytes st

/-! ## Lowercase hex -/

def hexDigit (n : UInt8) : Char :=
  if n < 10 then Char.ofNat (48 + n.toNat) else Char.ofNat (87 + n.toNat)

def hexChars : List UInt8 → List Char
  | [] => []
  | b :: bs => hexDigit (b >>> 4) :: hexDigit (b &&& 0xf) :: hexChars bs

/-- `sha256` of `data` as lowercase hex — the determinism receipt. -/
def hex (m : ByteArray) : String := String.ofList (hexChars (sha256 m).toList)

/-! ## Theorems (what is cheaply provable) -/

theorem hexChars_length (l : List UInt8) : (hexChars l).length = 2 * l.length := by
  induction l with
  | nil => rfl
  | cons b bs ih => simp [hexChars, ih]; omega

/-- (a) The receipt is always exactly 64 characters. -/
theorem hex_length (m : ByteArray) : (hex m).length = 64 := by
  simp [hex, String.length_ofList, hexChars_length]

/-- (b) padding: the padded length is a multiple of 64 ... -/
theorem paddedLen_mod (len : Nat) : paddedLen len % 64 = 0 := by
  unfold paddedLen zeroCount; omega

/-- ... and at least len + 9 (one 0x80 byte plus the 8-byte length). -/
theorem paddedLen_ge (len : Nat) : paddedLen len ≥ len + 9 := by
  unfold paddedLen; omega

/-- ... and at most len + 72 (so at most one extra block). -/
theorem paddedLen_le (len : Nat) : paddedLen len ≤ len + 72 := by
  unfold paddedLen zeroCount; omega

/-- `pad` really produces `paddedLen` bytes. -/
theorem size_pad (m : ByteArray) : (pad m).size = paddedLen m.size := by
  unfold pad paddedLen
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_push]
  show _ + (Array.replicate _ 0).size + (lenBytes _).toArray.size = _
  simp

theorem size_pad_mod (m : ByteArray) : (pad m).size % 64 = 0 := by
  rw [size_pad]; exact paddedLen_mod _

theorem size_pad_ge (m : ByteArray) : (pad m).size ≥ m.size + 9 := by
  rw [size_pad]; exact paddedLen_ge _

/-- The byte right after the message is 0x80, as FIPS §5.1.1 says. -/
theorem pad_get_first (m : ByteArray) :
    (pad m)[m.size]'(by rw [size_pad]; have := paddedLen_ge m.size; omega) = 0x80 := by
  unfold pad
  rw [ByteArray.getElem_append_left, ByteArray.getElem_append_left]
  · simp only [ByteArray.getElem_eq_getElem_data, ByteArray.push]
    exact Array.getElem_push_eq (xs := m.data) (x := 128)
  · rw [ByteArray.size_push]; omega
  · rw [ByteArray.size_append, ByteArray.size_push]; omega

/-- (c-primitives) shift-based ROTR agrees with BitVec.rotateRight for every
    rotation amount the algorithm uses; Ch and Maj agree with the alternative
    (mux) forms. Bit-blasted with `bv_decide` (adds `Lean.ofReduceBool`). -/
theorem rotr_2  (x : UInt32) : rotr x 2  = ⟨x.toBitVec.rotateRight 2⟩  := by unfold rotr; bv_decide
theorem rotr_6  (x : UInt32) : rotr x 6  = ⟨x.toBitVec.rotateRight 6⟩  := by unfold rotr; bv_decide
theorem rotr_7  (x : UInt32) : rotr x 7  = ⟨x.toBitVec.rotateRight 7⟩  := by unfold rotr; bv_decide
theorem rotr_11 (x : UInt32) : rotr x 11 = ⟨x.toBitVec.rotateRight 11⟩ := by unfold rotr; bv_decide
theorem rotr_13 (x : UInt32) : rotr x 13 = ⟨x.toBitVec.rotateRight 13⟩ := by unfold rotr; bv_decide
theorem rotr_17 (x : UInt32) : rotr x 17 = ⟨x.toBitVec.rotateRight 17⟩ := by unfold rotr; bv_decide
theorem rotr_18 (x : UInt32) : rotr x 18 = ⟨x.toBitVec.rotateRight 18⟩ := by unfold rotr; bv_decide
theorem rotr_19 (x : UInt32) : rotr x 19 = ⟨x.toBitVec.rotateRight 19⟩ := by unfold rotr; bv_decide
theorem rotr_22 (x : UInt32) : rotr x 22 = ⟨x.toBitVec.rotateRight 22⟩ := by unfold rotr; bv_decide
theorem rotr_25 (x : UInt32) : rotr x 25 = ⟨x.toBitVec.rotateRight 25⟩ := by unfold rotr; bv_decide

theorem ch_mux (e f g : UInt32) : ch e f g = (e &&& f) ||| ((~~~e) &&& g) := by
  unfold ch; bv_decide
theorem maj_alt (a b c : UInt32) : maj a b c = (a &&& b) ||| (c &&& (a ||| b)) := by
  unfold maj; bv_decide

/-- (c) totality: these are ordinary (non-`partial`) definitions; the digest
    has 32 bytes by type. Stated as a theorem only so `#print axioms` can
    witness that nothing hidden is assumed. -/
theorem sha256_size (m : ByteArray) : (sha256 m).toArray.size = 32 := by simp

/-! ## FIPS 180-4 test vectors, kernel-checked

`decide +kernel` evaluates the whole hash inside the trusted kernel
(no `Lean.ofReduceBool`, no compiler trust): about 1-1.5 s per 64-byte
block. The one-million-'a' vector (15625 blocks) is out of kernel reach
and is covered by `native_decide` in Tests.lean. These vectors are the
ONLY evidence that this function is SHA-256. -/

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem vec_empty_kernel : (sha256 ⟨#[]⟩).toList = [0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55] := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem vec_abc_kernel : (sha256 ⟨#[97,98,99]⟩).toList = [0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad] := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem vec_nist56_kernel : (sha256 ⟨#[97,98,99,100,98,99,100,101,99,100,101,102,100,101,102,103,101,102,103,104,102,103,104,105,103,104,105,106,104,105,106,107,105,106,107,108,106,107,108,109,107,108,109,110,108,109,110,111,109,110,111,112,110,111,112,113]⟩).toList = [0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1] := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem vec_nist112_kernel : (sha256 ⟨#[97,98,99,100,101,102,103,104,98,99,100,101,102,103,104,105,99,100,101,102,103,104,105,106,100,101,102,103,104,105,106,107,101,102,103,104,105,106,107,108,102,103,104,105,106,107,108,109,103,104,105,106,107,108,109,110,104,105,106,107,108,109,110,111,105,106,107,108,109,110,111,112,106,107,108,109,110,111,112,113,107,108,109,110,111,112,113,114,108,109,110,111,112,113,114,115,109,110,111,112,113,114,115,116,110,111,112,113,114,115,116,117]⟩).toList = [0xcf,0x5b,0x16,0xa7,0x78,0xaf,0x83,0x80,0x03,0x6c,0xe5,0x9e,0x7b,0x04,0x92,0x37,0x0b,0x24,0x9b,0x11,0xe8,0xf0,0x7a,0x51,0xaf,0xac,0x45,0x03,0x7a,0xfe,0xe9,0xd1] := by decide +kernel

end Sha256
