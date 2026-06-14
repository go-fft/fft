package kernels

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestIsPowerOfTwo(t *testing.T) {
	cases := map[int]bool{
		-4: false, -1: false, 0: false, 1: true, 2: true,
		3: false, 4: true, 5: false, 8: true, 12: false, 16: true,
	}
	for n, want := range cases {
		if got := IsPowerOfTwo(n); got != want {
			t.Fatalf("IsPowerOfTwo(%d) = %v, want %v", n, got, want)
		}
	}
}

func TestBitReverse(t *testing.T) {
	// For n=8 the bit-reversal permutation is well known.
	x := []complex128{0, 1, 2, 3, 4, 5, 6, 7}
	BitReverse(x)
	want := []complex128{0, 4, 2, 6, 1, 5, 3, 7}
	for i := range x {
		if x[i] != want[i] {
			t.Fatalf("index %d: got %v, want %v", i, x[i], want[i])
		}
	}
}

// TestRadix2RoundTrip exercises both the forward and inverse branches directly.
func TestRadix2RoundTrip(t *testing.T) {
	n := 8
	x := make([]complex128, n)
	orig := make([]complex128, n)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)), float64(i))
		orig[i] = x[i]
	}
	BitReverse(x)
	Radix2(x, false) // forward
	BitReverse(x)
	Radix2(x, true) // inverse (unnormalized)
	for i := range x {
		x[i] /= complex(float64(n), 0)
		if cmplx.Abs(x[i]-orig[i]) > 1e-12 {
			t.Fatalf("index %d: got %v, want %v", i, x[i], orig[i])
		}
	}
}
