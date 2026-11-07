package fft

import (
	"fmt"
	"testing"
)

// The instruction and traffic model behind the decision not to build a
// register-resident NEON leaf for 64 and 128 points (BENCHMARKS.md, Round 46).
// It extends radix16_predict_arm64_test.go: the same per-pass step counts
// (predStep), split into the part a leaf keeps and the part it removes.
//
// One pass step handles 2r points (two lanes of r streams): r VLD2, the
// butterfly, 5 instructions per twiddled output (one is the twiddle load), r
// stores (a last pass: 2r VZIPs plus r stores). A leaf of n points keeps the
// butterflies and the internal twiddle products (the same arithmetic), replaces
// the twiddle loads by constants, and does one load and one store per point:
// n VLD2 and n stores (with the 2n VZIPs of a final step) per 2n points, i.e.
// 0.5 + 1.5 instructions per point. Its 2n values of two lanes (re and im per
// point) exceed the 32 vector registers: with leafRegs usable (32 minus
// constants, pointers, scratch), 2n-leafRegs values make one round trip through
// the frame per memory stage (1 for 64 points, 2 for 128), each costing
// spillCost instructions per value and trip (2 with VST1/VLD1 one register at a
// time, 0.5 with the four-register ST1/LD1 forms).
const leafRegs = 26

// arithOf is the arithmetic (vector operations that are neither loads nor
// stores) per step of radix r, twiddled flag t.
func arithOf(r int, t bool) int {
	b := map[int]int{2: 4, 4: 16, 8: 57}[r]
	if t {
		b += 4 * (r - 1)
	}
	return b
}

// leafCost is the instructions per point of one untwiddled leaf replacing the
// passes f (all but the last twiddled in the old plan), spill per value and
// trip, trips memory stages, and the fraction cst of constants loaded per point.
func leafCost(f []int, spillCost float64, trips int) float64 {
	n := 1
	var arith float64
	for i, r := range f {
		n *= r
		arith += float64(arithOf(r, i != len(f)-1)) / float64(2*r)
	}
	spilled := float64(2*n - leafRegs)
	if spilled < 0 {
		spilled = 0
	}
	spill := spilled * spillCost * float64(trips) / float64(2*n)
	consts := float64(n-1) / float64(2*n) // one constant load per internal twiddle, bound
	return arith + 0.5 + 1.5 + spill + consts
}

// TestLeafPrediction prints, per row, the old per-point cost of the passes the
// leaf replaces, the leaf's cost range (cheap spill .. dear spill, one trip for
// 64 points, two for 128), the ratio of the whole transform (vector-loop
// instructions per point), and the main-data traffic in bytes per point (16 B
// read and 16 B written per pass; the leaf's frame traffic is L1 and counted in
// the instructions). Run with -v.
func TestLeafPrediction(t *testing.T) {
	rows := []struct {
		name  string
		n     int
		col   bool // column axis of a 2-D transform: no VZIP, lanes are columns
		group int  // points of the leaf at the end of the plan
	}{
		{"2-D 64x64 rows", 64, false, 64},
		{"2-D 128x128 rows", 128, false, 128},
		{"2-D 256x256 rows", 256, false, 128},
		{"2-D 1024x1024 rows", 1024, false, 64},
		{"RFFT 256 (complex 128)", 128, false, 128},
		{"RFFT 65,536 (complex 32,768)", 32768, false, 64},
		{"RFFT 1,048,576 (complex 524,288)", 524288, false, 64},
		{"2-D 64x64 columns", 64, true, 64},
		{"2-D 128x128 columns", 128, true, 128},
	}
	for _, r := range rows {
		f := skFactorize(r.n)
		// the tail of f whose product is r.group, if it is one
		k, p := len(f), 1
		for k > 0 && p < r.group {
			k--
			p *= f[k]
		}
		if p != r.group {
			t.Logf("%-34s %v: no tail with product %d", r.name, f, r.group)
			continue
		}
		old := predCost(f, 0)
		oldTail := predCost(f[k:], 0)
		if r.col { // batched column passes store without the VZIP pairing
			old -= 1.0
			oldTail -= 1.0
		}
		trips := 1
		if r.group == 128 {
			trips = 2
		}
		colAdj := 0.0
		if r.col {
			colAdj = 1.0 // leaf stores plainly too
		}
		lo := leafCost(f[k:], 0.5, 1) - colAdj
		hi := leafCost(f[k:], 2, trips) - colAdj
		passes := len(f)
		np := passes - (len(f) - k) + 1
		t.Logf("%-34s %v tail %v: old tail %.2f -> leaf %.2f..%.2f; row %.2f -> %.2f..%.2f = %.3f..%.3f; traffic %d -> %d B/pt (%d -> %d passes)",
			r.name, f, f[k:], oldTail, lo, hi, old, old-oldTail+lo, old-oldTail+hi,
			(old-oldTail+lo)/old, (old-oldTail+hi)/old, 32*passes, 32*np, passes, np)
	}
	// FFTW's published counts (BENCHMARKS.md, Round 38 bibliography): n1fv_64
	// 456 additions + 124 multiplications = 580 vector operations per 64 points
	// (9.06 per point, the model's arithmetic of two radix-8 passes is 8.88),
	// 456 with FMA (7.13).
	t.Log(fmt.Sprintf("n1fv_64 non-FMA %.2f per point, FMA %.2f", 580.0/64, 456.0/64))
}
