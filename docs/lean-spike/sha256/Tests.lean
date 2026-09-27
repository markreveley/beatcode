import Sha256
open Sha256

def s (x : String) : ByteArray := x.toUTF8
def nist56 := "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
def nist112 := "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
def mod251 (n : Nat) : ByteArray := ⟨Array.ofFn fun (i : Fin n) => (i.val % 251).toUInt8⟩

-- #eval: run in the interpreter and compare to the goldens (from tests/sha256_wav.rs / rs/ref.txt)
#eval hex (s "") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
#eval hex (s "abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
#eval hex (s nist56) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
#eval hex (s nist112) == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
#eval hex ⟨Array.replicate 1000000 97⟩ == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
-- padding-seam lengths, goldens from the Rust reference program
#eval hex (mod251 55) == "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"
#eval hex (mod251 56) == "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"
#eval hex (mod251 63) == "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"
#eval hex (mod251 64) == "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"
#eval hex (mod251 119) == "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"
#eval hex (mod251 120) == "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"
#eval hex (mod251 128) == "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"
-- streaming-equivalence analogue: hash of ragged concat equals one-shot (trivial here: one-shot only)

-- native_decide: the compiled code is the evidence (Lean.ofReduceBool axiom)
theorem vec_empty : hex (s "") = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" := by native_decide
theorem vec_abc : hex (s "abc") = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" := by native_decide
theorem vec_nist56 : hex (s nist56) = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" := by native_decide
theorem vec_nist112 : hex (s nist112) = "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" := by native_decide
theorem vec_million : hex ⟨Array.replicate 1000000 97⟩ = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" := by native_decide
#print axioms vec_abc
#print axioms vec_million

-- axioms of the library theorems
#print axioms hex_length
#print axioms paddedLen_mod
#print axioms paddedLen_ge
#print axioms paddedLen_le
#print axioms size_pad
#print axioms size_pad_mod
#print axioms size_pad_ge
#print axioms pad_get_first
#print axioms rotr_7
#print axioms ch_mux
#print axioms maj_alt
#print axioms sha256_size
#print axioms hexChars_length
#print axioms vec_empty_kernel
#print axioms vec_abc_kernel
#print axioms vec_nist56_kernel
#print axioms vec_nist112_kernel
#print axioms rotr_2
#print axioms rotr_25
