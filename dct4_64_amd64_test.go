package fft

import (
	"math"
	"testing"
)

// The AVX2 pair kernels of the DCT-IV(64) (dct4_64_amd64.s): the tests of
// dct4_fused_test.go run on them through the plans of fusedPlans (the pair is the
// kernel wherever fmaAvailable); these add what only the kernels need.

// TestDCT4N64PairFallback: with the kernels off (a CPU without AVX2 or FMA3) the
// pair forms are the scalar kernel twice, and every pair test holds there too.
func TestDCT4N64PairFallback(t *testing.T) {
	withoutPairAsm(func() {
		for _, n := range fusedLens {
			for _, p := range fusedPlans(n) {
				testDCT4FusedPair(t, n, p)
				testDCT4FusedShortPanics(t, n, p)
			}
		}
	})
}

// pairImpulseAbsTol bounds the largest absolute error of any output of any
// impulse through the AVX2 pair kernels (the outputs are cosines of magnitude up
// to 2): see TestDCT4N64PairKernelImpulses for the measurement and the mutation
// that sets it.
const pairImpulseAbsTol = 2e-15

// TestDCT4N64PairKernelImpulses: every impulse through the pair kernels, one in
// each lane, in every norm and both precisions, against the double-double oracle,
// by the largest absolute error of any output. A relative RMS over 64 outputs
// dilutes an error that touches a few of them; the absolute maximum does not.
func TestDCT4N64PairKernelImpulses(t *testing.T) {
	if !dct4PairAsm {
		t.Skip("no AVX2 and FMA3 here: the pair kernels are not routed")
	}
	p := NewDCTPlan(64, 4)
	if !p.pair {
		t.Fatal("the plan does not route its pair forms to the kernels")
	}
	var worst, worst32 float64
	for _, norm := range accNorms {
		for j := 0; j < 64; j++ {
			k := (j*5 + 17) % 64 // another impulse in the other lane
			e0, e1 := make([]float64, 64), make([]float64, 64)
			e0[j], e1[k] = 1, 1
			w0 := oR2R(cddFromR(e0), true, 4, norm, false)
			w1 := oR2R(cddFromR(e1), true, 4, norm, false)
			a, b := p.DCTPair(make([]float64, 64), make([]float64, 64), e0, e1, norm)
			worst = math.Max(worst, math.Max(pairMaxAbsErr(a, w0), pairMaxAbsErr(b, w1)))
			f0, f1 := make([]float32, 64), make([]float32, 64)
			f0[j], f1[k] = 1, 1
			c, d := p.DCTMixedPair(make([]float32, 64), make([]float32, 64), f0, f1, norm)
			worst32 = math.Max(worst32, math.Max(pairMaxAbsErr32(c, w0), pairMaxAbsErr32(d, w1)))
		}
	}
	t.Logf("worst absolute error: float64 %.3g (bound %.3g), Mixed %.3g", worst, pairImpulseAbsTol, worst32)
	if worst > pairImpulseAbsTol {
		t.Errorf("pair kernel impulse absolute error %.3g exceeds %.3g", worst, pairImpulseAbsTol)
	}
	if worst32 > 2*0x1p-24 {
		t.Errorf("pair kernel Mixed impulse absolute error %.3g exceeds 2^-23 (the float32 spacing in [1, 2), twice the rounding error)", worst32)
	}
}

// pairMaxAbsErr is the largest absolute difference between got and the oracle's real
// parts.
func pairMaxAbsErr(got []float64, want []cdd) float64 {
	var m float64
	for i, g := range got {
		m = math.Max(m, math.Abs((g-want[i].re.hi)-want[i].re.lo))
	}
	return m
}

func pairMaxAbsErr32(got []float32, want []cdd) float64 {
	var m float64
	for i, g := range got {
		m = math.Max(m, math.Abs((float64(g)-want[i].re.hi)-want[i].re.lo))
	}
	return m
}
