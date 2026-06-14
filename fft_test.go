package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

const tol = 1e-9

// naiveDFT is a reference O(N²) forward DFT used to cross-check the FFT.
func naiveDFT(x []complex128) []complex128 {
	n := len(x)
	out := make([]complex128, n)
	for k := 0; k < n; k++ {
		var sum complex128
		for j := 0; j < n; j++ {
			ang := -2 * math.Pi * float64(k) * float64(j) / float64(n)
			sum += x[j] * cmplx.Rect(1, ang)
		}
		out[k] = sum
	}
	return out
}

// closeVec reports whether two complex slices match within tol.
func closeVec(t *testing.T, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if d := cmplx.Abs(got[i] - want[i]); d > tol {
			t.Fatalf("index %d: got %v, want %v (|diff|=%g)", i, got[i], want[i], d)
		}
	}
}

func TestFFTAgainstNaiveDFT(t *testing.T) {
	// Mix of power-of-two and non-power-of-two lengths, exercising both the
	// radix-2 and Bluestein paths.
	for _, n := range []int{2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 31, 64} {
		x := make([]complex128, n)
		for i := range x {
			// Deterministic but non-trivial signal.
			x[i] = complex(math.Sin(float64(i)*0.7)+float64(i%3), math.Cos(float64(i)*0.3))
		}
		closeVec(t, FFT(x), naiveDFT(x))
	}
}

func TestRoundTrip(t *testing.T) {
	for _, n := range []int{1, 2, 3, 5, 8, 11, 16, 25, 32, 63, 100} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(i)*0.5-1, math.Sin(float64(i)))
		}
		got := IFFT(FFT(x))
		closeVec(t, got, x)
	}
}

func TestImpulseGivesFlatSpectrum(t *testing.T) {
	// A unit impulse at index 0 transforms to an all-ones spectrum.
	for _, n := range []int{4, 6, 8, 13} {
		x := make([]complex128, n)
		x[0] = 1
		X := FFT(x)
		want := make([]complex128, n)
		for i := range want {
			want[i] = 1
		}
		closeVec(t, X, want)
	}
}

func TestConstantGivesDCSpike(t *testing.T) {
	// A constant signal transforms to a single spike at bin 0 of value N.
	for _, n := range []int{4, 7, 8} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = 1
		}
		X := FFT(x)
		want := make([]complex128, n)
		want[0] = complex(float64(n), 0)
		closeVec(t, X, want)
	}
}

func TestSinusoidSpike(t *testing.T) {
	// A complex exponential at frequency bin f produces a single spike at f.
	n := 16
	f := 3
	x := make([]complex128, n)
	for i := range x {
		ang := 2 * math.Pi * float64(f) * float64(i) / float64(n)
		x[i] = cmplx.Rect(1, ang)
	}
	X := FFT(x)
	for k := range X {
		mag := cmplx.Abs(X[k])
		if k == f {
			if math.Abs(mag-float64(n)) > tol {
				t.Fatalf("bin %d: magnitude %g, want %d", k, mag, n)
			}
		} else if mag > tol {
			t.Fatalf("bin %d: magnitude %g, want ~0", k, mag)
		}
	}
}

func TestLinearity(t *testing.T) {
	// FFT(a*x + b*y) == a*FFT(x) + b*FFT(y), checked on a Bluestein length.
	n := 10
	a, b := complex(2, -1), complex(0.5, 3)
	x := make([]complex128, n)
	y := make([]complex128, n)
	for i := range x {
		x[i] = complex(float64(i), -float64(i)*0.2)
		y[i] = complex(math.Cos(float64(i)), float64(i%4))
	}
	combined := make([]complex128, n)
	for i := range combined {
		combined[i] = a*x[i] + b*y[i]
	}
	got := FFT(combined)
	Fx, Fy := FFT(x), FFT(y)
	want := make([]complex128, n)
	for i := range want {
		want[i] = a*Fx[i] + b*Fy[i]
	}
	closeVec(t, got, want)
}

func TestFFTReal(t *testing.T) {
	for _, n := range []int{1, 5, 8, 12} {
		r := make([]float64, n)
		c := make([]complex128, n)
		for i := range r {
			r[i] = math.Sin(float64(i)) + float64(i)
			c[i] = complex(r[i], 0)
		}
		closeVec(t, FFTReal(r), FFT(c))
	}
}

func TestEdgeCases(t *testing.T) {
	// Empty input: empty, non-nil slices, no normalization division by zero.
	if got := FFT([]complex128{}); got == nil || len(got) != 0 {
		t.Fatalf("FFT empty: got %v", got)
	}
	if got := IFFT([]complex128{}); got == nil || len(got) != 0 {
		t.Fatalf("IFFT empty: got %v", got)
	}
	if got := FFTReal([]float64{}); got == nil || len(got) != 0 {
		t.Fatalf("FFTReal empty: got %v", got)
	}

	// Length 1: identity (a copy).
	one := []complex128{complex(7, -3)}
	if got := FFT(one); len(got) != 1 || got[0] != one[0] {
		t.Fatalf("FFT len-1: got %v", got)
	}
	if got := IFFT(one); len(got) != 1 || got[0] != one[0] {
		t.Fatalf("IFFT len-1: got %v", got)
	}
}

func TestNoMutation(t *testing.T) {
	// Neither FFT nor IFFT may modify the caller's input slice.
	for _, n := range []int{8, 10} {
		x := make([]complex128, n)
		orig := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64(i), float64(-i))
			orig[i] = x[i]
		}
		FFT(x)
		closeVec(t, x, orig)
		IFFT(x)
		closeVec(t, x, orig)
	}
}
