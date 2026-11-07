package fft

import (
	"fmt"
	"testing"
)

// The instruction model behind the decision not to build a NEON radix-16 pass
// (BENCHMARKS.md, Round 45). Every pass kernel runs one "step" per two lanes of
// r streams, i.e. 2r points; the step's instruction count follows from
// asmgen/fft/arm64/pass.go: r loads, the butterfly's vector operations, five
// instructions per twiddled output (one VLD1, FMUL, FMLS, FMUL, FMLA) and r
// stores. The model reproduces benchmarks/results/arm64-m4-gap-20261102/
// counts.txt: skPass4NEON has 90 vector and 38 memory instructions over three
// bodies (28 and ~11 each: 16 + 3·4, 4 + 3 + 4), skPass8NEON 270 and 74.
//
// A radix-16 step is the 4×4 split of bfly16 (radix16.go): 4 bfly4 (16 vector
// operations each), eight internal twiddle products by radix16Rot (4 each; the
// ninth, W16^4 = rotS, folds into the following adds), 4 bfly4, 15 outer
// twiddled outputs, 16 loads, 16 stores: 267 instructions. NEON has 32 vector
// registers, a 16-point butterfly holds 32 values of two lanes, so values must
// be spilled to the frame: spill is the number of extra VST1/VLD1 pairs per
// step (16 = all stage-1 values, 8 = half, 0 = the unreachable bound).
const (
	r16StepArith = 16 + 4*16 + 8*4 + 4*16 + 15*5 + 16 // 267
)

// predStep is the instructions of one twiddled step of radix r, and of a final
// (twiddle-free) step, which pairs two blocks with 2r VZIPs.
func predStep(r, spill int) (twiddled, last int) {
	var tw int
	switch r {
	case 2:
		tw = 2 + 4 + 5*1 + 2
	case 4:
		tw = 4 + 16 + 5*3 + 4
	case 8:
		tw = 8 + 57 + 5*7 + 8
	case 16:
		tw = r16StepArith + 2*spill
	default:
		panic("predStep")
	}
	return tw, tw - 5*(r-1) + 2*r
}

// predCost is a factorization's vector-loop instructions per point.
func predCost(f []int, spill int) float64 {
	var c float64
	for i, r := range f {
		tw, last := predStep(r, spill)
		s := tw
		if i == len(f)-1 {
			s = last
		}
		c += float64(s) / float64(2*r)
	}
	return c
}

// withRadix16 replaces each adjacent pair of radix-4 passes by one radix-16.
func withRadix16(f []int) []int {
	var g []int
	for i := 0; i < len(f); i++ {
		if f[i] == 4 && i+1 < len(f) && f[i+1] == 4 {
			g = append(g, 16)
			i++
			continue
		}
		g = append(g, f[i])
	}
	return g
}

// TestRadix16Prediction prints, for the rows still behind FFTW 3.3.11 on the
// Apple M4 Max, the factorization go-fft plans (skFactorize, the complex
// transform of an RFFT or the row axis of a 2-D transform), the same with every
// 4,4 pair a radix-16 pass, and the model's instruction ratio. Run with -v.
func TestRadix16Prediction(t *testing.T) {
	if got, _ := predStep(4, 0); got != 39 {
		t.Fatalf("radix-4 step = %d, want 39", got)
	}
	if got, _ := predStep(8, 0); got != 108 {
		t.Fatalf("radix-8 step = %d, want 108", got)
	}
	rows := []struct {
		name string
		n    int
	}{
		{"RFFT 256 (complex 128)", 128},
		{"RFFT 65,536 (complex 32,768)", 32768},
		{"RFFT 1,048,576 (complex 524,288)", 524288},
		{"2-D 64x64 rows", 64},
		{"2-D 128x128 rows", 128},
		{"2-D 256x256 rows", 256},
		{"2-D 1024x1024 rows", 1024},
	}
	for _, r := range rows {
		f := skFactorize(r.n)
		g := withRadix16(f)
		base := predCost(f, 0)
		line := fmt.Sprintf("%-34s %v -> %v  per point: %.2f", r.name, f, g, base)
		for _, spill := range []int{0, 8, 16} {
			line += fmt.Sprintf("  spill %2d: %.2f (%.3fx)", spill, predCost(g, spill), predCost(g, spill)/base)
		}
		t.Log(line)
	}
}
