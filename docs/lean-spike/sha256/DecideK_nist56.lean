import Sha256
open Sha256
set_option maxRecDepth 100000
set_option maxHeartbeats 0
-- nist56: 56 bytes, 2 block(s); expected digest computed by python hashlib
theorem vec_nist56_kernel : (sha256 ⟨#[97,98,99,100,98,99,100,101,99,100,101,102,100,101,102,103,101,102,103,104,102,103,104,105,103,104,105,106,104,105,106,107,105,106,107,108,106,107,108,109,107,108,109,110,108,109,110,111,109,110,111,112,110,111,112,113]⟩).toList = [0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1] := by decide +kernel
#print axioms vec_nist56_kernel
