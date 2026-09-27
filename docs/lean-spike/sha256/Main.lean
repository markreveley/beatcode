import Sha256

/-- Same LCG as the Rust reference program (rs/src/main.rs). -/
def lcgBytes (n : Nat) : ByteArray := Id.run do
  let mut s : UInt64 := 0x9E3779B97F4A7C15
  let mut out := ByteArray.emptyWithCapacity n
  for _ in [0:n] do
    s := s * 6364136223846793005 + 1442695040888963407
    out := out.push (s >>> 56).toUInt8
  return out

def timeIt (label : String) (act : IO String) : IO Unit := do
  let t0 ← IO.monoNanosNow
  let r ← act
  let t1 ← IO.monoNanosNow
  IO.println s!"{label} {r}"
  IO.println s!"{label}_secs {(t1 - t0).toFloat / 1e9}"

def main (args : List String) : IO Unit := do
  match args with
  | [path] =>
    let bytes ← IO.FS.readBinFile path
    let r ← IO.mkRef bytes
    timeIt s!"file({bytes.size})" (do let b ← r.get; pure (Sha256.hex b))
  | _ =>
    let r ← IO.mkRef (lcgBytes 500000)
    timeIt "lcg500k" (do let b ← r.get; pure (Sha256.hex b))
    let r2 ← IO.mkRef (ByteArray.mk (Array.replicate 1000000 97))
    timeIt "million_a" (do let b ← r2.get; pure (Sha256.hex b))
