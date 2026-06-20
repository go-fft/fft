package fft

import (
	"math"
	"testing"
)

func closeReal(t *testing.T, got, want []float64, ctx string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", ctx, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(got[i] - want[i]); d > tol {
			t.Fatalf("%s index %d: got %.12g, want %.12g (|d|=%g)", ctx, i, got[i], want[i], d)
		}
	}
}

// TestWindowEdgeCases covers n<=0 (empty, non-nil) and n==1 ([1]) for every
// window, exercising the shared guards in cosineWindow and Bartlett.
func TestWindowEdgeCases(t *testing.T) {
	wins := map[string]func(int) []float64{
		"Hann": Hann, "Hamming": Hamming, "Blackman": Blackman,
		"BlackmanHarris": BlackmanHarris, "Bartlett": Bartlett,
	}
	for name, fn := range wins {
		if got := fn(0); got == nil || len(got) != 0 {
			t.Fatalf("%s(0): got %v", name, got)
		}
		if got := fn(-3); got == nil || len(got) != 0 {
			t.Fatalf("%s(-3): got %v", name, got)
		}
		if got := fn(1); len(got) != 1 || got[0] != 1 {
			t.Fatalf("%s(1): got %v", name, got)
		}
	}
}

// TestHannExact asserts the exact numpy.hanning values for a small length.
func TestHannExact(t *testing.T) {
	// numpy.hanning(5) = [0, 0.5, 1, 0.5, 0].
	want := []float64{0, 0.5, 1, 0.5, 0}
	closeReal(t, Hann(5), want, "Hann(5)")
}

// TestHammingExact asserts the exact numpy.hamming values.
func TestHammingExact(t *testing.T) {
	// numpy.hamming(5): endpoints 0.08, center 1.0; symmetric.
	// w[k] = 0.54 - 0.46*cos(2*pi*k/4).
	want := make([]float64, 5)
	for k := 0; k < 5; k++ {
		want[k] = 0.54 - 0.46*math.Cos(2*math.Pi*float64(k)/4)
	}
	closeReal(t, Hamming(5), want, "Hamming(5)")
	// Spot endpoints exactly.
	if math.Abs(want[0]-0.08) > tol || math.Abs(want[2]-1.0) > tol {
		t.Fatalf("hamming reference wrong: %v", want)
	}
}

func TestBlackmanExact(t *testing.T) {
	want := make([]float64, 7)
	for k := 0; k < 7; k++ {
		c := 2 * math.Pi * float64(k) / 6
		want[k] = 0.42 - 0.5*math.Cos(c) + 0.08*math.Cos(2*c)
	}
	closeReal(t, Blackman(7), want, "Blackman(7)")
}

func TestBlackmanHarrisExact(t *testing.T) {
	a := []float64{0.35875, 0.48829, 0.14128, 0.01168}
	n := 8
	want := make([]float64, n)
	for k := 0; k < n; k++ {
		c := 2 * math.Pi * float64(k) / float64(n-1)
		want[k] = a[0] - a[1]*math.Cos(c) + a[2]*math.Cos(2*c) - a[3]*math.Cos(3*c)
	}
	closeReal(t, BlackmanHarris(n), want, "BlackmanHarris(8)")
}

func TestBartlettExact(t *testing.T) {
	// numpy.bartlett(5) = [0, 0.5, 1, 0.5, 0].
	closeReal(t, Bartlett(5), []float64{0, 0.5, 1, 0.5, 0}, "Bartlett(5)")
	// numpy.bartlett(4) = [0, 2/3, 2/3, 0].
	closeReal(t, Bartlett(4), []float64{0, 2.0 / 3.0, 2.0 / 3.0, 0}, "Bartlett(4)")
}

// TestWindowSymmetry confirms every window is symmetric about its center.
func TestWindowSymmetry(t *testing.T) {
	for _, fn := range []func(int) []float64{Hann, Hamming, Blackman, BlackmanHarris, Bartlett} {
		w := fn(9)
		for i := range w {
			if math.Abs(w[i]-w[len(w)-1-i]) > tol {
				t.Fatalf("window not symmetric at %d: %v", i, w)
			}
		}
	}
}
