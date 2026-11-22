package fft

import (
	"math"
	"testing"
)

// TestDCT32KernelRouted: on a CPU with AVX2 and FMA3 a DCTPlan32(64, 4) routes to
// the kernel, and with the kernel switched off (before the plan is built) it does
// not.
func TestDCT32KernelRouted(t *testing.T) {
	isolatePlans(t)
	// The routing switch, not fmaAvailable: other tests force the latter.
	if got := NewDCTPlan32(64, 4).quad; got != dct4F32Asm {
		t.Errorf("DCTPlan32(64, 4).quad = %v, dct4F32Asm = %v", got, dct4F32Asm)
	}
	if got := dct4F32Route(64, 4); got != dct4F32Asm {
		t.Errorf("dct4F32Route(64, 4) = %v, dct4F32Asm = %v", got, dct4F32Asm)
	}
	if dct4F32Route(128, 4) || dct4F32Route(64, 2) {
		t.Error("the kernel is routed for a size or type it does not compute")
	}
	old := dct4F32Asm
	dct4F32Asm = false
	defer func() { dct4F32Asm = old }()
	if NewDCTPlan32(64, 4).quad {
		t.Error("plan built with the kernel switched off still routes to it")
	}
}

// f32Bytes fills s from a deterministic stream of values in [-scale, scale].
func f32Fill(s []float32, state *uint64, scale float32) {
	for i := range s {
		*state = *state*6364136223846793005 + 1442695040888963407
		s[i] = scale * (float32(int64(*state>>11)-(1<<52)) / (1 << 52))
	}
}

// FuzzDCT32PairQuadKernel: the kernels against the generic float32 plan on arbitrary
// data of any magnitude (finite), both within the stated bound of each other (each
// is within eps32·log2(n) of the oracle, so twice that of one another, relative to
// the 2-norm of the result, with a floor for results that vanish).
func FuzzDCT32PairQuadKernel(f *testing.F) {
	f.Add(uint64(1), float32(1), uint8(0))
	f.Add(uint64(2), float32(1e-20), uint8(1))
	f.Add(uint64(3), float32(1e18), uint8(2))
	f.Add(uint64(4), float32(0), uint8(0))
	f.Fuzz(func(t *testing.T, seed uint64, scale float32, normSel uint8) {
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || math.Abs(float64(scale)) > 1e18 {
			t.Skip()
		}
		const n = 64
		norm := f32PairNorms[int(normSel)%len(f32PairNorms)]
		p := NewDCTPlan32(n, 4)
		var x, q, d [4][]float32
		st := seed
		for i := range x {
			x[i] = make([]float32, n)
			q[i] = make([]float32, n)
			d[i] = make([]float32, n)
			f32Fill(x[i], &st, scale)
		}
		p.DCTQuad(q, x, norm)
		r0, r1 := p.DCTPair(make([]float32, n), make([]float32, n), x[0], x[1], norm)
		pair := [][]float32{r0, r1}
		b := 2 * bound32(n)
		for i := range x {
			ref := make([]float32, n)
			if p.quad {
				r := *p
				r.quad = false
				r.DCT(ref, x[i], norm)
			} else {
				p.DCT(ref, x[i], norm)
			}
			check := func(what string, got []float32) {
				var num, den float64
				for j := range got {
					e := float64(got[j]) - float64(ref[j])
					num += e * e
					den += float64(ref[j]) * float64(ref[j])
				}
				if den == 0 {
					if num != 0 {
						t.Fatalf("%s stream %d: result %v on a zero reference", what, i, got[:4])
					}
					return
				}
				if e := math.Sqrt(num / den); e > b {
					t.Fatalf("%s stream %d: relative difference %.3g from the generic plan exceeds %.3g", what, i, e, b)
				}
			}
			check("quad", q[i])
			if i < 2 {
				check("pair", pair[i])
			}
		}
	})
}
