package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestSkLast20EVEXBitIdentical holds the EVEX-256 radix-20 final pass to the
// AVX2 one bit for bit, for even and odd l1 (the 128-bit tail) and both
// directions, on random data with signed zeros.
func TestSkLast20EVEXBitIdentical(t *testing.T) {
	if !detectedDispatch.Zen4 {
		t.Skip("skLast20EVEX needs AVX-512VL (Zen 4 here)")
	}
	rng := rand.New(rand.NewSource(20))
	for _, inverse := range []bool{false, true} {
		k := &skFwd[0][0]
		if inverse {
			k = &skInv[0][0]
		}
		for _, l1 := range []int{1, 2, 3, 4, 5, 8, 12, 96, 97} {
			n := 20 * l1
			in := make([]complex128, n)
			for i := range in {
				in[i] = complex(rng.NormFloat64(), rng.NormFloat64())
			}
			in[0] = complex(math.Copysign(0, -1), 0)
			a, b := make([]complex128, n), make([]complex128, n)
			skLast20AVX2(&in[0], &a[0], k, l1)
			skLast20EVEX(&in[0], &b[0], k, l1)
			for i := range a {
				if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) ||
					math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
					t.Fatalf("inverse=%v l1=%d: point %d differs: %v vs %v", inverse, l1, i, a[i], b[i])
				}
			}
		}
	}
}
