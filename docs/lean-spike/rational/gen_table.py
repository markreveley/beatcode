# Single source of truth for the golden table: emits rs/src/main.rs and RatTable.lean
MAX = 2**63 - 1; MIN = -2**63
P62 = 2**62; P32 = 2**32; P31 = 2**31
new_cases = [
 (0,1),(0,-7),(0,5),(1,2),(2,4),(-1,4),(1,-4),(-1,-4),(6,-9),(-6,-9),(100,-1),
 (MAX,1),(MIN,1),(MIN,-1),(MAX,-1),(MIN,MIN),(MIN,2),(MIN,-2),(MAX,MAX),(MAX,MIN),
 (1,0),(0,0),(MIN,0),(MAX,3),(1,MIN),(2,MIN),(-2,MIN),(MAX,MAX-1),(MIN,MAX),(0,MIN),(12,18),(-12,18),
]
# (op, (an,ad), (bn,bd)) operands must construct OK
bin_cases = [
 ("add",(1,2),(1,3)),("add",(1,2),(-1,2)),("add",(-1,4),(1,4)),("add",(1,3),(2,3)),("add",(1,3),(1,6)),
 ("add",(MAX,1),(1,1)),("add",(MAX,1),(-MAX,1)),("add",(MAX,2),(MAX,2)),("add",(MAX,1),(MAX,1)),
 ("add",(MIN,1),(-1,1)),("add",(MIN,1),(MIN,1)),("add",(1,MAX),(1,MAX)),("add",(1,MAX),(1,MAX-1)),
 ("add",(1,P62),(1,P62)),("add",(-3,7),(-4,7)),("add",(0,1),(5,-7)),("add",(MIN,1),(0,1)),("add",(MAX,1),(0,1)),
 ("add",(MIN,P62),(MIN,P62)),("add",(MIN,1),(1,1)),("add",(MAX,1),(-1,1)),("add",(MIN,2),(MIN,2)),
 ("mul",(1,2),(2,3)),("mul",(-1,2),(-2,3)),("mul",(-1,2),(2,3)),("mul",(MAX,1),(1,MAX)),("mul",(MAX,1),(2,1)),
 ("mul",(MIN,1),(-1,1)),("mul",(MIN,1),(1,1)),("mul",(MIN,1),(1,2)),("mul",(MIN,1),(0,1)),("mul",(P32,1),(P31,1)),
 ("mul",(P32,1),(-P31,1)),("mul",(1,P32),(1,P31)),("mul",(1,P32),(1,P31-1)),("mul",(3,4),(4,3)),
 ("mul",(MAX,MAX-1),(MAX-1,MAX)),("mul",(MAX,2),(2,MAX)),("mul",(MIN,3),(3,-P62)),("mul",(0,-5),(MAX,1)),("mul",(MIN,1),(-1,2)),
 ("divr",(1,2),(1,3)),("divr",(1,2),(0,1)),("divr",(0,1),(0,1)),("divr",(0,1),(5,1)),("divr",(1,2),(-1,3)),("divr",(-1,2),(-1,3)),
 ("divr",(1,1),(MIN,1)),("divr",(1,1),(MAX,1)),("divr",(MIN,1),(-1,1)),("divr",(MIN,1),(1,1)),("divr",(MIN,1),(MIN,1)),
 ("divr",(MAX,1),(MAX,1)),("divr",(MAX,1),(1,MAX)),("divr",(1,MAX),(MAX,1)),("divr",(MIN,1),(2,1)),("divr",(MIN,1),(-2,1)),
 ("divr",(2,3),(4,9)),("divr",(1,3),(MIN,1)),("divr",(-1,4),(1,4)),("divr",(1,1),(-1,-P62)),("divr",(MAX,1),(-1,1)),
]
unary_cases = [(-1,4),(7,2),(-7,2),(1,1),(0,-3),(MIN,1),(MAX,1),(MIN,2),(-MAX,2),(5,-2),(1,3),(MAX,MAX-1),(1,-P62),(3,-MIN//3)]
def r(x): return f"({x})"
rs = ["use bc::rational::Rational;", "fn show(r: Result<Rational, bc::rational::RatError>) -> String { match r { Ok(q) => format!(\"ok {}\", q.to_s()), Err(e) => format!(\"err {:?}\", e) } }", "fn main() {"]
for n,d in new_cases:
    rs.append(f'  println!("new {n} {d} => {{}}", show(Rational::new({n}i64, {d}i64)));')
for op,(a,b),(c,d) in bin_cases:
    rs.append(f'  println!("{op} {a}/{b} {c}/{d} => {{}}", show(Rational::new({a}i64,{b}i64).unwrap().{op}(Rational::new({c}i64,{d}i64).unwrap())));')
for n,d in unary_cases:
    rs.append(f'  {{ let q = Rational::new({n}i64,{d}i64).unwrap(); println!("un {n} {d} => floor={{}} int={{}} s={{}} f=0x{{:016x}}", q.floor_i(), q.is_int(), q.to_s(), q.to_f().to_bits()); }}')
rs.append("}")
open("rs/src/main.rs","w").write("\n".join(rs)+"\n")
ln = ["import Rational", "open Rat64", "def main : IO Unit := do"]
for n,d in new_cases:
    ln.append(f'  IO.println s!"new {n} {d} => {{showR (mk? {r(n)} {r(d)})}}"')
for op,(a,b),(c,d) in bin_cases:
    ln.append(f'  IO.println s!"{op} {a}/{b} {c}/{d} => {{showBin {op} (mk? {r(a)} {r(b)}) (mk? {r(c)} {r(d)})}}"')
for n,d in unary_cases:
    ln.append(f'  IO.println s!"un {n} {d} => {{showUn (mk? {r(n)} {r(d)})}}"')
ln.append("#eval main")
open("RatTable.lean","w").write("\n".join(ln)+"\n")
print(len(new_cases), len(bin_cases), len(unary_cases))
