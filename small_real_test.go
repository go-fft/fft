package fft

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// smallSignals are simdSignals plus the signals the float64 untangle kernels
// are held to since Round 30: random draws from {±0} and from {±0, ±1} (a
// sum or difference of signed zeros keeps or loses its sign by the order of
// its operands), ±∞ mixed with finite values on both halves (an infinity
// against its mirror makes ∞ − ∞), and odd multiples of the smallest
// subnormal of both parities (a sum of two odd multiples halves exactly, an
// odd one rounds: only a mixed signal tells a halving rounded on its own from
// one fused or reordered).
func smallSignals(n int) [][]complex128 {
	r := rand.New(rand.NewPCG(30, uint64(n)))
	neg := math.Copysign(0, -1)
	pick := func(vs ...float64) float64 { return vs[r.IntN(len(vs))] }
	zeros := make([]complex128, n)
	ones := make([]complex128, n)
	infs := cmplxSignal(n)
	subs := make([]complex128, n)
	for i := range n {
		zeros[i] = complex(pick(0, neg), pick(0, neg))
		ones[i] = complex(pick(0, neg, 1, -1), pick(0, neg, 1, -1))
		if i%5 == 1 {
			infs[i] = complex(pick(math.Inf(1), math.Inf(-1)), imag(infs[i]))
		}
		if i%7 == 3 {
			infs[i] = complex(real(infs[i]), pick(math.Inf(1), math.Inf(-1)))
		}
		subs[i] = complex(float64(2*(i%7)+1)*5e-324*pick(1, -1), float64(i%4+1)*5e-324*pick(1, -1))
	}
	return append(simdSignals(n), zeros, ones, infs, subs)
}

// TestUntangleInPlace: the RealPlan untangles in place since Round 30 (the
// half transform writes dst, which the untangle then reads), so whatever
// kernel runs here must give the bits it gives out of place, at every
// half-length up to 300 and a few large ones.
func TestUntangleInPlace(t *testing.T) {
	sizes := []int{1024, 2048, 4097}
	for m := 1; m <= 300; m++ {
		sizes = append(sizes, m)
	}
	for _, m := range sizes {
		tw := NewRealPlan(2 * m).tw
		for s, z := range smallSignals(m) {
			out := make([]complex128, m+1)
			rfftUntangle(out, z, tw, m)
			in := append(slices.Clone(z), 0)
			rfftUntangle(in, in[:m], tw, m)
			for k := range out {
				if !sameBits(out[k], in[k]) {
					t.Fatalf("m=%d signal %d bin %d: in place %v, out of place %v", m, s, k, in[k], out[k])
				}
			}
		}
	}
}

// TestRealPlanInPlaceMatchesBuffered: since Round 30, RFFT writes the packed
// half spectrum into dst and untangles it there, and IRFFT builds it in dst
// and inverts it in place, where both used a pooled buffer. Both must give
// the bits of the buffered composition, for every engine the half plan can
// take (Stockham with an odd and an even pass count, with and without the
// scratch gap, Rader, Bluestein; the iterative kernel keeps the buffer), and
// for complete, short and over-long spectra.
func TestRealPlanInPlaceMatchesBuffered(t *testing.T) {
	for _, n := range []int{2, 4, 6, 8, 10, 16, 30, 64, 100, 128, 256, 512, 1000, 1024, 2048, 4096, 8192, 2 * 1009, 2 * 1201, 2 * 10007} {
		p := NewRealPlan(n)
		m := n / 2
		for s, x := range smallSignals(m) {
			src := make([]float64, n)
			for j, v := range x {
				src[2*j], src[2*j+1] = real(v), imag(v)
			}
			got := p.RFFT(make([]complex128, m+1), src)
			want := make([]complex128, m+1)
			Z := make([]complex128, m)
			if p.half.it != nil {
				p.half.it.transformRealPacked(Z, src)
			} else {
				p.half.execute(Z, asComplex(src), false)
			}
			rfftUntangle(want, Z, p.tw, m)
			for k := range want {
				if !sameBits(got[k], want[k]) {
					t.Fatalf("RFFT n=%d signal %d bin %d: %v, buffered %v", n, s, k, got[k], want[k])
				}
			}
			for _, spec := range [][]complex128{want, want[:m/2+1], append(slices.Clone(want), 7, 8)} {
				back := p.IRFFT(make([]float64, n), spec)
				if p.half.it != nil {
					continue // unchanged path
				}
				Zb := make([]complex128, m)
				p.packSpectrum(Zb, spec, 0.5*(1/float64(m)))
				ref := make([]float64, n)
				p.half.execute(asComplex(ref), Zb, true)
				for j := range ref {
					if math.Float64bits(back[j]) != math.Float64bits(ref[j]) && !(math.IsNaN(back[j]) && math.IsNaN(ref[j])) {
						t.Fatalf("IRFFT n=%d signal %d len %d sample %d: %v, buffered %v", n, s, len(spec), j, back[j], ref[j])
					}
				}
			}
		}
	}
}

// TestIRFFTInPlaceWhen pins smallIRFFTInPlace's rule.
func TestIRFFTInPlaceWhen(t *testing.T) {
	even := newSKPlanFactors(2048, []int{8, 8, 8, 4})
	odd3 := newSKPlanFactors(2048, []int{8, 16, 16})
	small := newSKPlanFactors(64, []int{4, 4, 4})
	casc := &skPlan{n: 1 << 16, cascT: 1}
	for _, c := range []struct {
		half *Plan
		want bool
	}{
		{&Plan{n: 128, it: &itPlan{}}, false},
		{&Plan{n: 1009, rader: &raderPlan{}}, false},
		{&Plan{n: 2048, sk: even}, true},
		{&Plan{n: 2048, sk: odd3}, false},
		{&Plan{n: 64, sk: small}, true},
		{&Plan{n: 1 << 16, sk: casc}, false},
	} {
		if got := smallIRFFTInPlace(c.half); got != c.want {
			t.Errorf("n=%d: smallIRFFTInPlace = %v, want %v", c.half.n, got, c.want)
		}
	}
}
