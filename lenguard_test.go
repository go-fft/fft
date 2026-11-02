package fft

import (
	"math"
	"testing"
)

// TestCheckAllocBoundary: the guard fires exactly where n*elemBytes stops
// fitting in an int, for every element size the package allocates. The
// black-box tests of every entry point are in internal/guardtest, which also
// runs on 32-bit targets (math.MaxInt is 2^31-1 there).
func TestCheckAllocBoundary(t *testing.T) {
	for _, b := range []int{4, 8, 16, 32, 64} {
		limit := math.MaxInt / b
		checkAlloc("boundary", limit, b) // must not panic
		checkAlloc("boundary", 0, b)
		checkAlloc("boundary", -1, b)
		mustPanicWith(t, "limit+1", "fft: boundary: length ", func() { checkAlloc("boundary", limit+1, b) })
		mustPanicWith(t, "MaxInt", "fft: boundary: length ", func() { checkAlloc("boundary", math.MaxInt, b) })
		if limit*b < 0 {
			t.Fatalf("elemBytes %d: the bound %d still overflows (%d)", b, limit, limit*b)
		}
	}
}

// TestBluesteinConvolutionBound: a non-smooth length above math.MaxInt/64 is
// refused before its 2n-1 convolution length is computed (the buffers are up to
// 2.8n complex128), where the 16-byte bound of NewPlan alone would let it
// through to bestConvLen.
func TestBluesteinConvolutionBound(t *testing.T) {
	mustPanicWith(t, "Bluestein", "fft: NewPlan: length ", func() { newBluesteinPlanOpt(math.MaxInt/64+1, false) })
}
