package fft

import (
	"math"
	"testing"
)

// irfftOddRef is the odd-length inverse as it was written before it took the
// plan's pooled buffer: two fresh length-n slices, full Hermitian spectrum, one
// inverse complex FFT, real part.
func irfftOddRef(p *RealPlan, src []complex128) []float64 {
	n := p.n
	full := make([]complex128, n)
	half := n/2 + 1
	for k := 0; k < half && k < len(src); k++ {
		full[k] = src[k]
		if k > 0 && k < n-k {
			full[n-k] = complexConj(src[k])
		}
	}
	inv := make([]complex128, n)
	cachedPlanA(n, p.ar).IFFT(inv, full)
	dst := make([]float64, n)
	for i := range dst {
		dst[i] = real(inv[i])
	}
	return dst
}

func oddSpectrum(n, bins int) []complex128 {
	s := make([]complex128, bins)
	for k := range s {
		s[k] = complex(math.Sin(0.7*float64(k+1))+0.25, math.Cos(1.3*float64(k*k%97)))
	}
	return s
}

// The pooled odd inverse is bit-identical to the allocating one, including after
// a call that left the pooled buffer full of another spectrum (a short input must
// see zeros, not the previous call's bins), under the default and Exact plans.
func TestIRFFTOddPooledBufferBitIdentical(t *testing.T) {
	for _, o := range []Options{{}, {Exact: true}, {FMA: true}} {
		for _, n := range []int{3, 5, 9, 15, 25, 125, 1001, 1125} {
			p := NewRealPlanWith(n, o)
			half := n/2 + 1
			// full, short, full again, shorter, empty: each after a dirtier predecessor
			for _, bins := range []int{half, half / 2, half, 1, half - 1, 0, half} {
				src := oddSpectrum(n, bins)
				want := irfftOddRef(p, src)
				got := p.IRFFT(make([]float64, n), src)
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("opts %+v n=%d bins=%d: sample %d = %v, want %v", o, n, bins, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// n == 1 has no pooled buffer; it must still invert (the value passes through).
func TestIRFFTLengthOne(t *testing.T) {
	p := NewRealPlan(1)
	if got := p.IRFFT(make([]float64, 1), []complex128{complex(2.5, 0)}); got[0] != 2.5 {
		t.Fatalf("IRFFT of one bin = %v, want 2.5", got[0])
	}
}

// A steady-state odd inverse allocates nothing: it used to make two length-n
// slices per call.
func TestIRFFTOddSteadyStateAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("sync.Pool drops entries at random under -race; pool reuse cannot be asserted")
	}
	p := NewRealPlan(1125)
	src, dst := oddSpectrum(1125, 563), make([]float64, 1125)
	p.IRFFT(dst, src)
	// A sync.Pool may be emptied by a GC (and drops entries at random under
	// the race detector), which costs the next call one buffer: the old code
	// allocated on every call, so one clean run in several is the witness.
	best := math.MaxFloat64
	for try := 0; try < 8 && best != 0; try++ {
		best = math.Min(best, testing.AllocsPerRun(50, func() { p.IRFFT(dst, src) }))
	}
	if best != 0 {
		t.Fatalf("odd IRFFT allocates %v times per call, want 0", best)
	}
}
