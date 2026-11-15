package kernels

import (
	"sync"
	"unsafe"
)

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
//
// The alignment is lazy: Fused<n>Prepare (a sync.Once per table) does it, and it is
// called by the plan construction that chooses the kernel (and by Fused<n>Table), so
// that no package initializer refers to a table: a program that never builds a fused
// plan of that length does not link its table (a complex128-only program is 527 KB
// smaller). The kernels' calls read the aligned pointer, a plain variable that the
// Once has written before any plan that uses it exists; the hot path reads no Once.
var (
	fused256TwA, fused1024TwA, fused2048TwA []float64

	// The first element of each table, for the kernels' calls (a pointer variable
	// keeps Fused256 and Fused1024 within the inlining budget). Nil until prepared.
	fused256TwP, fused1024TwP, fused2048TwP *float64

	fused256Once, fused1024Once, fused2048Once sync.Once
)

// Fused256Prepare aligns the 256-point kernel's table (once). Fused256 and
// Fused256Ref must not be called before it (a plan of 256 points that takes the
// kernel calls it when it is built).
func Fused256Prepare() {
	fused256Once.Do(func() {
		fused256TwA = alignedTable(fused256TwStore[:])
		fused256TwP = &fused256TwA[0]
	})
}

// Fused1024Prepare is Fused256Prepare for the 1024-point kernel.
func Fused1024Prepare() {
	fused1024Once.Do(func() {
		fused1024TwA = alignedTable(fused1024TwStore[:])
		fused1024TwP = &fused1024TwA[0]
	})
}

// Fused2048Prepare is Fused256Prepare for the 2048-point kernel.
func Fused2048Prepare() {
	fused2048Once.Do(func() {
		fused2048TwA = alignedTable(fused2048TwStore[:])
		fused2048TwP = &fused2048TwA[0]
	})
}

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
