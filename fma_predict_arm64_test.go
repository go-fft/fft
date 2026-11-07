package fft

import "testing"

// The instruction model behind the decision not to build NEON FMA kernels for
// Options.FMA (BENCHMARKS.md, Round 47). It extends radix16_predict_arm64_test.go
// and leaf_predict_arm64_test.go (same predCost step counts).
//
// What FMA can still remove on arm64. The default NEON kernels already fuse
// every twiddle product: a twiddled output is VLD1, FMUL, FMLS, FMUL, FMLA (4
// arithmetic instructions, the count of FFTW's FMA complex product, against 6
// without FMA), because they reproduce the FMSUBD/FMADDD that gc emits for the
// Go reference (asmgen/arm64/gen.go header; internal/kernels/cmul_arm64.go). The
// radix-2 and radix-4 butterflies contain additions and ±i swaps only. Radix 8
// contains the one constant product: bfly8's ±45° and ±135° rotations, each a
// multiply by √2/2 followed by an add and a subtract (fmaPM in fma.go): per step
// of two lanes 4 multiplies and 8 add/subs become 8 FMAs, 57 → 53 arithmetic
// instructions (108 → 104 per 16 points).
const fmaSavedR8 = 4

// predCostFMA is predCost with each radix-8 step 4 instructions shorter.
func predCostFMA(f []int) float64 {
	c := predCost(f, 0)
	for _, r := range f {
		if r == 8 {
			c -= float64(fmaSavedR8) / float64(2*r)
		}
	}
	return c
}

// untangleBound is a generous upper bound on what fusing the RFFT untangle
// could save, per complex point: its products are already FMLS/FMLA
// (untangle_arm64.s), so only the 0.5 scalings could fuse; half an instruction
// per point is more than they hold.
const untangleBound = 0.5

// TestFMAPrediction prints, per behind row, the vector-loop instructions per
// point of the default plan, of the FMA plan, their ratio, and the ratio the
// row needs (1 / measured gap: 0.78 to 0.88).
func TestFMAPrediction(t *testing.T) {
	rows := []struct {
		name string
		n    int
		rfft bool
		gap  float64 // measured go-fft ÷ FFTW (Rounds 38 and 44), the worst of the range
	}{
		{"2-D 64x64 rows", 64, false, 1.18},
		{"2-D 128x128 rows", 128, false, 1.18},
		{"2-D 256x256 rows", 256, false, 1.28},
		{"2-D 1024x1024 rows", 1024, false, 1.28},
		{"RFFT 256 (complex 128)", 128, true, 1.18},
		{"RFFT 65,536 (complex 32,768)", 32768, true, 1.18},
		{"RFFT 1,048,576 (complex 524,288)", 524288, true, 1.18},
	}
	for _, r := range rows {
		f := skFactorize(r.n)
		old := predCost(f, 0)
		neu := predCostFMA(f)
		opt := neu
		if r.rfft {
			opt -= untangleBound
		}
		t.Logf("%-34s %v: %.2f -> %.2f instr/pt = %.3f (%.3f with the untangle bound); needs <= %.3f",
			r.name, f, old, neu, neu/old, opt/old, 1/r.gap)
		if opt/old < 1/r.gap {
			t.Errorf("%s: the model reaches the needed ratio; build the kernels", r.name)
		}
	}
}
