package kernels

import (
	"math"
	"testing"
)

// TestCMulSignedZeroNEON holds the NEON complex multiply to CMulScalar on
// signed zeros, which the shared random and structured inputs never produce:
// (-0)·1 is -0, and a rounded product formed as 0 + x·y (the kernel's form
// before Go 1.27 had VFMUL) turns it into +0.
func TestCMulSignedZeroNEON(t *testing.T) {
	neg := math.Copysign(0, -1)
	vals := []float64{0, neg, 1, -1}
	var a, b []complex128
	for _, ar := range vals {
		for _, ai := range vals {
			for _, br := range vals {
				for _, bi := range vals {
					a = append(a, complex(ar, ai))
					b = append(b, complex(br, bi))
				}
			}
		}
	}
	want := append([]complex128(nil), a...)
	CMulScalar(want, b)
	got := append([]complex128(nil), a...)
	cmulSIMD(got, b)
	for i := range want {
		if !bitEqual(got[i], want[i]) {
			t.Fatalf("index %d: %v·%v = %v, CMulScalar %v", i, a[i], b[i], got[i], want[i])
		}
	}
}
