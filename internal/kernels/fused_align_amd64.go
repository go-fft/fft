package kernels

import "unsafe"

// The twiddle tables of the fused kernels (generated: fused256TwStore,
// fused1024TwStore, fused2048TwStore) are read with 64-byte zmm loads, but Go's
// linker aligns a data symbol to 32 bytes at most, so a table can sit at 32 mod 64
// and then every load straddles two cache lines (62 per pass-1 iteration of the
// 1024 kernel). Each generated array carries 4 float64s of padding on each side
// and its data from index 4: at init, alignedTable moves the data, in place, to
// the first 64-byte boundary (index 4 where the array is at 32 mod 64, so nothing
// moves; index 0 where it is at 0 mod 64, a copy of the table down by four
// elements). No heap is allocated and no second copy is kept. The forward table
// is followed by the inverse one, as the kernels expect.
var (
	fused256TwA  = alignedTable(fused256TwStore[:])
	fused1024TwA = alignedTable(fused1024TwStore[:])
	fused2048TwA = alignedTable(fused2048TwStore[:])

	// The first element of each table, for the kernels' calls (a pointer variable
	// keeps Fused256 and Fused1024 within the inlining budget).
	fused256TwP  = &fused256TwA[0]
	fused1024TwP = &fused1024TwA[0]
	fused2048TwP = &fused2048TwA[0]
)

// alignedTable returns the table held in store (4 elements of padding, the data,
// 4 of padding) as a slice whose first element is 64-byte aligned, having moved
// the data there within store.
func alignedTable(store []float64) []float64 {
	n := len(store) - 8
	off := int((64-uintptr(unsafe.Pointer(&store[0]))%64)%64) / 8 // 0 to 7: the first aligned element
	if off != 4 {
		copy(store[off:off+n], store[4:4+n]) // memmove: overlapping, within store
	}
	return store[off : off+n : off+n]
}

// alignedTable32 is alignedTable for the float32 tables: 8 elements (32 bytes) of
// padding on each side and the data from index 8, moved in place to the first
// 64-byte boundary, index 0 to 15 (8 where the array is at 32 mod 64: nothing
// moves).
func alignedTable32(store []float32) []float32 {
	n := len(store) - 16
	off := int((64-uintptr(unsafe.Pointer(&store[0]))%64)%64) / 4
	if off != 8 {
		copy(store[off:off+n], store[8:8+n]) // memmove: overlapping, within store
	}
	return store[off : off+n : off+n]
}

// halfTable32 is one direction's half of a table (forward first).
func halfTable32(t []float32, inverse bool) []float32 {
	if inverse {
		return t[len(t)/2:]
	}
	return t[:len(t)/2]
}
