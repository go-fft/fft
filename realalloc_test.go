package fft

import (
	"math"
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

// FFTReal is documented as promote-then-FFT; it stays bit-identical to that
// on every engine (iterative pow2, Stockham, Rader, Bluestein, trivial).
func TestFFTRealBitIdenticalToPromotedFFT(t *testing.T) {
	lens := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 17, 31, 64, 97, 100, 127, 211, 256, 509, 1000, 1009, 1024, 1125, 2003, 4096, 4099, 8192}
	for n := 0; n <= 130; n++ {
		lens = append(lens, n)
	}
	for _, n := range lens {
		x := realAllocSignal(n)
		got, want := FFTReal(x), FFT(promote(x))
		if i := firstBitDiff(got, want); i != len(want) {
			t.Fatalf("n=%d: FFTReal differs from FFT(promoted) at bin %d", n, i)
		}
	}
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
