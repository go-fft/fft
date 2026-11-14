package fft

import "testing"

// TestNoScalarRadix32: no default or FMA plan of the running CPU has a radix-32 pass
// that no kernel runs (it would fall back to passScalarFMA, 4.8 times slower), at any
// power of two, with the table as it is and with a table that asks for 32·8·8 where the
// route has no kernel for it (the planner then refuses it and plans skFactorize's).
func TestNoScalarRadix32(t *testing.T) {
	isolatePlans(t)
	defer func(f map[int]fmaChoice) { fmaTable = f }(fmaTable)
	bad := map[int]fmaChoice{}
	for k, v := range fmaTable {
		bad[k] = v
	}
	for e := 5; e <= 16; e++ {
		bad[1<<e] = fmaChoice{factors: append([]int{32}, skFactorize(1<<e/32)...), fuse: fmaAll}
	}
	for name, tab := range map[string]map[int]fmaChoice{"table": fmaTable, "asking for a first radix-32 pass everywhere": bad} {
		fmaTable = tab
		for e := 5; e <= 16; e++ {
			n := 1 << e
			for _, fma := range []bool{false, true} {
				p := newSKPlanOpt(n, fma)
				if hasScalarRadix32(p) {
					t.Errorf("%s: n=%d fma=%v has a radix-32 pass with no kernel", name, n, fma)
				}
			}
		}
	}
}
