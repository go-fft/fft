package kernels

import "unsafe"

// The twiddle tables of the fused kernels (fused256Tw, fused1024Tw, generated) are
// read with 64-byte zmm loads, but Go's linker aligns a data symbol to 32 bytes at
// most, so a table can sit at 32 mod 64 and then every load straddles two cache
// lines (62 per pass-1 iteration of the 1024 kernel). The kernels read these
// copies, made at init in a heap block offset to a multiple of 64 (the Go heap
// does not move objects); the forward table is followed by the inverse one, as the
// kernels expect.
var (
	fused256TwA  = alignedTables(fused256Tw[0][:], fused256Tw[1][:])
	fused1024TwA = alignedTables(fused1024Tw[0][:], fused1024Tw[1][:])

	// The first element of each copy, for the kernels' calls (a pointer variable
	// keeps Fused256 and Fused1024 within the inlining budget).
	fused256TwP  = &fused256TwA[0]
	fused1024TwP = &fused1024TwA[0]
)

// alignedTables returns a, then b, copied into one slice whose first element is
// 64-byte aligned.
func alignedTables(a, b []float64) []float64 {
	n := len(a) + len(b)
	buf := make([]float64, n+8)
	off := int((64-uintptr(unsafe.Pointer(&buf[0]))%64)%64) / 8
	t := buf[off : off+n : off+n]
	copy(t, a)
	copy(t[len(a):], b)
	return t
}
