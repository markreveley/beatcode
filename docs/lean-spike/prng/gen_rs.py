import os
REPO=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",".."))
import json,sys
S=os.path.dirname(os.path.abspath(__file__))
rows=[json.loads(l) for l in open(REPO+"/goldens/prng-vectors.jsonl") if l.strip()]
out=["use bc::prng::*;","fn main(){"]
for i,r in enumerate(rows):
    if r["fn"]=="fnv":
        out.append(f'  println!("fnv {i} {{}}", fnv({json.dumps(r["in"],ensure_ascii=False)}));')
    elif r["fn"]=="flt":
        ps=",".join(f'Part::Str({json.dumps(p)})' if isinstance(p,str) else f'Part::Int({p})' for p in r["parts"])
        out.append(f'  {{ let seed = mask_seed({r["seed"]}i128); let parts=[{ps}]; let k=flt_key(seed,&parts); let sm=splitmix64(k); let f=flt(seed,&parts); println!("flt {i} {{}} {{}} {{}} {{:016x}}", seed, k, sm, f.to_bits()); }}')
    elif r["fn"]=="noise":
        out.append(f'  {{ let base = fnv(&format!("sample|{{}}", {json.dumps(r["tag"])})); let v = noise({json.dumps(r["tag"])}, 8); for j in 0..8 {{ let k=flt_key(base,&[Part::Int(j as i64)]); let sm=splitmix64(k); println!("noise {i} {{}} {{}} {{}} {{}} {{:016x}}", j, base, k, sm, v[j].to_bits()); }} }}')
out.append("}")
open(S+"/rs/src/main.rs","w").write("\n".join(out)+"\n")
