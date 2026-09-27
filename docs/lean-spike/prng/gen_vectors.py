import os
REPO=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",".."))
import json,struct
S=os.path.dirname(os.path.abspath(__file__))
rows=[json.loads(l) for l in open(REPO+"/goldens/prng-vectors.jsonl") if l.strip()]
ref={}
for l in open(S+"/ref.txt"):
    t=l.split()
    if t[0]=="fnv": ref[("fnv",int(t[1]))]=int(t[2])
    elif t[0]=="flt": ref[("flt",int(t[1]))]=(int(t[2]),int(t[3]),int(t[4]),int(t[5],16))
    elif t[0]=="noise": ref[("noise",int(t[1]),int(t[2]))]=(int(t[3]),int(t[4]),int(t[5]),int(t[6],16))
def bits(s): return struct.unpack("<Q",struct.pack("<d",float(s)))[0]
def lstr(s): return json.dumps(s,ensure_ascii=False)
def lint(i): return f"({i} : Int)" if i<0 else str(i)
out=["import Prng","open Prng","set_option maxRecDepth 4000","set_option maxHeartbeats 4000000",""]
ev_flt=[]; ev_noise=[]; nfnv=nflt=nnoise=0; nthm=0
for i,r in enumerate(rows):
    if r["fn"]=="fnv":
        want=int(r["out"]); assert ref[("fnv",i)]==want
        out.append(f'theorem fnv_{i} : fnv {lstr(r["in"])} = {want} := by decide +kernel'); nfnv+=1; nthm+=1
    elif r["fn"]=="flt":
        seed,key,sm,rbits=ref[("flt",i)]
        gkey=int(r["key"]); gbits=bits(r["out"]); assert key==gkey and rbits==gbits, i
        parts=", ".join(f'.str {lstr(p)}' if isinstance(p,str) else f'.int {lint(p)}' for p in r["parts"])
        out.append(f'theorem flt_{i}_seed : maskSeed {lint(r["seed"])} = {seed} := by decide +kernel')
        out.append(f'theorem flt_{i}_key : fltKey {seed} [{parts}] = {gkey} := by decide +kernel')
        out.append(f'theorem flt_{i}_out : splitmix64 {gkey} = {sm} := by decide +kernel')
        out.append(f'theorem flt_{i}_bits : fltBitsOfOut {sm} = 0x{gbits:016x} := by decide +kernel')
        ev_flt.append(f'({sm}, 0x{gbits:016x})'); nflt+=1; nthm+=4
    elif r["fn"]=="noise":
        for j,w in enumerate(r["first8"]):
            base,key,sm,rbits=ref[("noise",i,j)]; gbits=bits(w); assert rbits==gbits,(i,j)
            out.append(f'theorem noise_{i}_{j}_key : noiseKey {lstr(r["tag"])} {j} = {key} := by decide +kernel')
            out.append(f'theorem noise_{i}_{j}_out : splitmix64 {key} = {sm} := by decide +kernel')
            out.append(f'theorem noise_{i}_{j}_bits : noiseBitsOfOut {sm} = 0x{gbits:016x} := by decide +kernel')
            ev_noise.append(f'({lstr(r["tag"])}, {j}, 0x{gbits:016x})'); nnoise+=1; nthm+=3
# extra §4.4 probes (from tests/prng_goldens.rs)
probes=[(2**64-1,bits("1.0")),(9007199254740993,bits("4.8828125e-4")),(6148914691236517205,bits("0.3333333333333333"))]
for k,(x,b) in enumerate(probes):
    out.append(f'theorem probe_{k} : fltBitsOfOut {x} = 0x{b:016x} := by decide +kernel'); nthm+=1
out.append("")
out.append("/-! ## Float agreement tests (opaque Float, #eval only) -/")
out.append("def fltTable : List (UInt64 × UInt64) := [" + ",\n  ".join(ev_flt) + "]")
out.append("def noiseTable : List (String × Nat × UInt64) := [" + ",\n  ".join(ev_noise) + "]")
out.append("def probeTable : List (UInt64 × UInt64) := [" + ", ".join(f'({x}, 0x{b:016x})' for x,b in probes) + "]")
out.append('''
def fltOk := fltTable.filter fun (sm, b) => (fltOfOut sm).toBits == b
def fltModelOk := fltTable.filter fun (sm, b) => fltBitsOfOut sm == b
def fltFloatVsModel := fltTable.filter fun (sm, _) => (fltOfOut sm).toBits == fltBitsOfOut sm
def noiseOk := noiseTable.filter fun (tag, i, b) => (noiseAt tag i).toBits == b
def noiseFloatVsModel := noiseTable.filter fun (tag, i, _) => (noiseAt tag i).toBits == noiseBitsOfOut (splitmix64 (noiseKey tag i))
def probeOk := probeTable.filter fun (x, b) => (fltOfOut x).toBits == b
#eval IO.println s!"flt Float==golden: {fltOk.length}/{fltTable.length}"
#eval IO.println s!"flt model==golden: {fltModelOk.length}/{fltTable.length}"
#eval IO.println s!"flt Float==model:  {fltFloatVsModel.length}/{fltTable.length}"
#eval IO.println s!"noise Float==golden: {noiseOk.length}/{noiseTable.length}"
#eval IO.println s!"noise Float==model:  {noiseFloatVsModel.length}/{noiseTable.length}"
#eval IO.println s!"probes Float==golden: {probeOk.length}/{probeTable.length}"
#eval IO.println s!"fltOfOut (2^64-1) == 1.0: {fltOfOut 18446744073709551615 == 1.0}  bits={(fltOfOut 18446744073709551615).toBits}"
#eval IO.println s!"noise snare 8 == (noise snare 64).take 8 (Float): {(noise "snare" 8).map Float.toBits == ((noise "snare" 64).take 8).map Float.toBits}"
''')
open(S+"/Vectors.lean","w").write("\n".join(out)+"\n")
print("theorems",nthm,"fnv",nfnv,"flt",nflt,"noise",nnoise)
# whole-conjunction variant for the 64 keys + 19 fnv
conj=["import Prng","open Prng","set_option maxRecDepth 4000","set_option maxHeartbeats 4000000","theorem all_keys_and_fnv :"]
cl=[]
for i,r in enumerate(rows):
    if r["fn"]=="fnv": cl.append(f'fnv {lstr(r["in"])} = {int(r["out"])}')
    elif r["fn"]=="flt":
        seed=ref[("flt",i)][0]
        parts=", ".join(f'.str {lstr(p)}' if isinstance(p,str) else f'.int {lint(p)}' for p in r["parts"])
        cl.append(f'fltKey (maskSeed {lint(r["seed"])}) [{parts}] = {int(r["key"])}')
conj.append("  " + " ∧\n  ".join(cl) + " := by decide +kernel")
open(S+"/Conj.lean","w").write("\n".join(conj)+"\n")
print("conj clauses",len(cl))
