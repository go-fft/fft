package fft

import (
	"math"
	"testing"
)

// fusedSBound is the c2c bound in single precision: c·eps32·log2 n, eps32 = 2^-23.
func fusedSBound(n int) float64 { return accBound["c2c"] * eps32 * math.Log2(float64(n)) }

func toC64Slice(x []complex128) []complex64 {
	out := make([]complex64, len(x))
	for i, v := range x {
		out[i] = complex(float32(real(v)), float32(imag(v)))
	}
	return out
}

// TestFusedSForcedRoute runs the Plan32 of every kernel length with the route
// forced on, on every platform: through the kernel where the CPU runs it (an AMD
// Zen 4 or an Intel Cascade Lake with AVX-512, or any AVX-512 host once its gate is
// on), and through the plan's passes where it refuses (no AVX-512, or another
// architecture), within the bound either way, forward and inverse.
func TestFusedSForcedRoute(t *testing.T) {
	isolatePlans(t)
	saved := map[int]bool{}
	for n := range fusedSEnabled {
		saved[n] = fusedSEnabled[n]
		fusedSEnabled[n] = true
	}
	defer func() {
		for n, v := range saved {
			fusedSEnabled[n] = v
		}
	}()
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192} {
		p := NewPlan32(n)
		if p.fs != n {
			t.Fatalf("n=%d: the plan does not take the forced route", n)
		}
		x := toC64Slice(accComplexF64(n, 21))
		for _, inv := range []bool{false, true} {
			got := make([]complex64, n)
			if inv {
				p.IFFTNorm(got, x, NormForward)
			} else {
				p.FFT(got, x)
			}
			if e := errC(got, ddDFT(cddFromC(x), inv)); e > fusedSBound(n) {
				t.Errorf("n=%d inverse=%v: %.3g above the bound %.3g", n, inv, e, fusedSBound(n))
			}
		}
	}
	// a length without a kernel never takes the route
	if p := NewPlan32(100); p.fs != 0 {
		t.Error("a plan of 100 points takes a fused kernel")
	}
}

// TestFusedComplex128RoutesPrepare: the complex128 routes of 256, 1024 and 2048
// points answer yes (and align their table, once) only when their flag is on and
// the length is theirs, on every platform.
func TestFusedComplex128RoutesPrepare(t *testing.T) {
	isolatePlans(t)
	defer func(a, b, c bool) { fused256Enabled, fused1024Enabled, fused2048Enabled = a, b, c }(fused256Enabled, fused1024Enabled, fused2048Enabled)
	fused256Enabled, fused1024Enabled, fused2048Enabled = true, true, true
	for n, route := range map[int]func(int) bool{256: fused256Route, 1024: fused1024Route, 2048: fused2048Route} {
		if !route(n) {
			t.Errorf("the route of %d points is off with its flag on", n)
		}
		if route(n + 1) {
			t.Errorf("the route of %d points answers yes for %d", n, n+1)
		}
	}
	fused256Enabled, fused1024Enabled, fused2048Enabled = false, false, false
	for n, route := range map[int]func(int) bool{256: fused256Route, 1024: fused1024Route, 2048: fused2048Route} {
		if route(n) {
			t.Errorf("the route of %d points is on with its flag off", n)
		}
	}
}
