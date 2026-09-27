import hashlib, struct
def parse(path):
    sec=None; d={}
    for line in open(path):
        line=line.rstrip('\n')
        if not line: continue
        if line.split()[0] in ("SIN","NOTE","KICKBITS","HATBITS") and len(line.split())==1:
            sec=line; d[sec]=[]; continue
        if line.startswith("KICK ") or line.startswith("HAT ") or line.startswith("CASTMODEL"):
            d[line.split()[0]]=line; continue
        d[sec].append(line)
    return d
R=parse('rust_ref.txt'); L=parse('lean_out.txt')
# SIN
rs=R['SIN']; ls=L['SIN']
assert len(rs)==len(ls)==200, (len(rs),len(ls))
inmis=sum(1 for a,b in zip(rs,ls) if a.split()[0]!=b.split()[0])
mis=[(a,b) for a,b in zip(rs,ls) if a!=b]
print("SIN inputs:",len(rs),"input-bit mismatches:",inmis,"output mismatches:",len(mis))
for a,b in mis[:10]: print("  rust",a,"lean",b)
# NOTE
rn=R['NOTE']; ln=L['NOTE']
nm=[(a,b) for a,b in zip(rn,ln) if a!=b]
print("NOTE 128 entries, mismatches:",len(nm)); print(nm[:5])
def sha(bits):
    h=hashlib.sha256()
    for x in bits: h.update(struct.pack('<Q', int(x,16)))
    return h.hexdigest()
for k in ("KICKBITS","HATBITS"):
    rk=R[k]; lk=L[k]
    km=sum(1 for a,b in zip(rk,lk) if a!=b)
    print(k, "len rust",len(rk),"lean",len(lk),"sample mismatches",km, "sha rust",sha(rk),"sha lean",sha(lk))
print(R['KICK']); print(R['HAT']); print(L['CASTMODEL'])
