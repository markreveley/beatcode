-- kernel-checked (decide) golden checks of the integer model
import Table
namespace Decfmt

def roundOK : Bool := roundTable.all fun (b, n, e) => decide (roundDecModel b n = e)
def fmtOK : Bool := fmtTable.all fun (b, n, s) => decide (formatDec b n = some s)
def divergeOK : Bool := divergeTable.all fun (b, n, oracle, exact) =>
  decide (formatDec b n = some exact) && decide (oracle ≠ exact)

#eval roundTable.length
#eval fmtTable.length
#eval divergeTable.length

set_option maxRecDepth 100000 in
set_option exponentiation.threshold 2000 in
theorem round_golden : roundOK = true := by decide
set_option maxRecDepth 100000 in
set_option exponentiation.threshold 2000 in
theorem fmt_golden : fmtOK = true := by decide
set_option maxRecDepth 100000 in
set_option exponentiation.threshold 2000 in
theorem diverge_pinned : divergeOK = true := by decide

#print axioms round_golden
#print axioms fmt_golden
#print axioms diverge_pinned

end Decfmt
