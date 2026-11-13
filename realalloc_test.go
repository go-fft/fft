package fft

import (
	"math"
	"math/cmplx"
	"sync"
	"testing"
)

func realAllocSignal(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(0.31*float64(i)) + 0.5*math.Cos(0.017*float64(i*i%1013)) + 0.1
	}
	return x
}

func promote(x []float64) []complex128 {
	c := make([]complex128, len(x))
	for i, v := range x {
		c[i] = complex(v, 0)
	}
	return c
}

func firstBitDiff(a, b []complex128) int {
	if len(a) != len(b) {
		return -1
	}
	for i := range a {
		if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) ||
			math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
			return i
		}
	}
	return len(a)
}

// FFTRealWith with Options.Exact is promote-then-FFT, bit for bit, on every
// engine (iterative pow2, Stockham, Rader, Bluestein, trivial); FFTReal is the
// same for an odd or trivial length, and for an even one the real transform
// plus the conjugate mirror: exactly Hermitian, and within a few ulp of the
// promoted transform's spectrum.
func TestFFTRealBitIdenticalToPromotedFFT(t *testing.T) {
	lens := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 17, 31, 64, 97, 100, 127, 211, 256, 509, 1000, 1009, 1024, 1125, 2003, 4096, 4099, 8192}
	for n := 0; n <= 130; n++ {
		lens = append(lens, n)
	}
	for _, n := range lens {
		x := realAllocSignal(n)
		want := FFT(promote(x))
		if i := firstBitDiff(FFTRealWith(x, Options{Exact: true}), want); i != len(want) {
			t.Fatalf("n=%d: FFTRealWith(Exact) differs from FFT(promoted) at bin %d", n, i)
		}
		got := FFTReal(x)
		if n%2 != 0 || n < 2 {
			if i := firstBitDiff(got, want); i != len(want) {
				t.Fatalf("n=%d: FFTReal differs from FFT(promoted) at bin %d", n, i)
			}
			continue
		}
		var norm, diff float64
		for k := range want {
			norm = math.Max(norm, cmplx.Abs(want[k]))
			diff = math.Max(diff, cmplx.Abs(got[k]-want[k]))
			if k > 0 && got[n-k] != complexConj(got[k]) {
				t.Fatalf("n=%d: FFTReal is not Hermitian at bin %d", n, k)
			}
		}
		if diff > 1e-13*norm*math.Log2(float64(n)+2) {
			t.Fatalf("n=%d: FFTReal differs from FFT(promoted) by %g (norm %g)", n, diff, norm)
		}
	}
}

// FFTRealWith: n and norm as numpy.fft.fft on a real signal, Exact and FMA.
func TestFFTRealWith(t *testing.T) {
	x := realAllocSignal(100)
	for _, o := range []Options{{}, {N: 64}, {N: 130}, {N: 101}, {Norm: NormOrtho}, {N: 64, Norm: NormForward}, {FMA: true}, {Exact: true, N: 96}} {
		want := FFTWith(promote(x), o)
		got := FFTRealWith(x, o)
		if len(got) != len(want) {
			t.Fatalf("%+v: length %d, want %d", o, len(got), len(want))
		}
		var norm, diff float64
		for k := range want {
			norm = math.Max(norm, cmplx.Abs(want[k]))
			diff = math.Max(diff, cmplx.Abs(got[k]-want[k]))
		}
		if diff > 1e-13*norm {
			t.Fatalf("%+v: differs by %g (norm %g)", o, diff, norm)
		}
	}
	if got := FFTRealWith(nil, Options{}); len(got) != 0 {
		t.Fatalf("empty: %v", got)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("Exact with FMA did not panic")
		}
	}()
	FFTRealWith(x, Options{Exact: true, FMA: true})
}

// The odd path of RealPlan.RFFT keeps the lower bins of the full complex
// transform of the promoted signal, bit for bit.
func TestRealPlanOddBitIdenticalToFullTransform(t *testing.T) {
	for _, n := range []int{3, 5, 7, 9, 15, 17, 97, 125, 211, 509, 1009, 1125, 2003} {
		x := realAllocSignal(n)
		p := NewRealPlan(n)
		got := p.RFFT(make([]complex128, n/2+1), x)
		want := FFT(promote(x))[:n/2+1]
		if i := firstBitDiff(got, want); i != len(want) {
			t.Fatalf("n=%d: odd RealPlan.RFFT differs at bin %d", n, i)
		}
	}
}

func TestRealPlanOddConcurrent(t *testing.T) {
	const n = 1009
	x := realAllocSignal(n)
	p := NewRealPlan(n)
	want := p.RFFT(make([]complex128, n/2+1), x)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dst := make([]complex128, n/2+1)
			for i := 0; i < 50; i++ {
				if firstBitDiff(p.RFFT(dst, x), want) != len(want) {
					t.Error("concurrent odd RealPlan.RFFT differs")
					return
				}
			}
		}()
	}
	wg.Wait()
}

// Exact allocation counts (testing.AllocsPerRun), after a warm-up call that
// builds the cached plan and primes the plan's scratch pool.
func TestRealAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("under -race sync.Pool drops items on purpose, so allocation counts are not meaningful")
	}
	for _, c := range []struct {
		name string
		n    int
		run  func(x []float64, p *RealPlan, dst []complex128)
		want float64
	}{
		{"RealPlan.RFFT/1009", 1009, func(x []float64, p *RealPlan, dst []complex128) { p.RFFT(dst, x) }, 0},
		{"RealPlan.RFFT/1000", 1000, func(x []float64, p *RealPlan, dst []complex128) { p.RFFT(dst, x) }, 0},
		{"RealPlan.RFFT/1125", 1125, func(x []float64, p *RealPlan, dst []complex128) { p.RFFT(dst, x) }, 0},
		{"RFFT/1009", 1009, func(x []float64, _ *RealPlan, _ []complex128) { RFFT(x) }, 1},
		{"RFFT/1024", 1024, func(x []float64, _ *RealPlan, _ []complex128) { RFFT(x) }, 1},
		{"FFTReal/1024", 1024, func(x []float64, _ *RealPlan, _ []complex128) { FFTReal(x) }, 1},
		{"FFTReal/1009", 1009, func(x []float64, _ *RealPlan, _ []complex128) { FFTReal(x) }, 1},
	} {
		x := realAllocSignal(c.n)
		p := NewRealPlan(c.n)
		dst := make([]complex128, c.n/2+1)
		c.run(x, p, dst)
		if got := testing.AllocsPerRun(50, func() { c.run(x, p, dst) }); got != c.want {
			t.Errorf("%s: %v allocations per call, want %v", c.name, got, c.want)
		}
	}
}
