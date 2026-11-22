package fft

import (
	"syscall"
	"testing"
	"unsafe"
)

// guardedF32 returns n float32 that end exactly where an inaccessible page begins
// (end true) or start exactly where one ends (end false): a kernel that reads or
// writes one element outside the 64 faults, and the fault fails the test binary.
func guardedF32(t *testing.T, n int, end bool) []float32 {
	t.Helper()
	pg := syscall.Getpagesize()
	b, err := syscall.Mmap(-1, 0, 3*pg, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(b) })
	for _, off := range []int{0, 2 * pg} {
		if err := syscall.Mprotect(b[off:off+pg], syscall.PROT_NONE); err != nil {
			t.Skipf("mprotect: %v", err)
		}
	}
	mid := b[pg : 2*pg]
	start := 0
	if end {
		start = pg - 4*n
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&mid[start])), n)
}

// TestDCT32KernelGuardPages: the pair and quad kernels, scaled and not, touch exactly
// the 64 elements of each stream, at either edge of a mapping, in and out of place.
func TestDCT32KernelGuardPages(t *testing.T) {
	const n = 64
	p := NewDCTPlan32(n, 4)
	if !p.quad {
		t.Skip("no float32 kernel on this CPU")
	}
	for _, end := range []bool{false, true} {
		for _, norm := range []Norm{NormBackward, NormOrtho} {
			x := f32Streams(n, 11)
			var src, dst, want [4][]float32
			for i := range src {
				src[i] = guardedF32(t, n, end)
				dst[i] = guardedF32(t, n, end)
				copy(src[i], x[i])
				want[i] = make([]float32, n)
			}
			p.DCTQuad(want, x, norm)
			p.DCTQuad(dst, src, norm)
			for i := range dst {
				for j := 0; j < n; j++ {
					if dst[i][j] != want[i][j] {
						t.Fatalf("quad guarded (end %v, %v) stream %d element %d: %v, want %v", end, norm, i, j, dst[i][j], want[i][j])
					}
				}
			}
			p.DCTPair(dst[0], dst[1], src[0], src[1], norm)
			for i := 0; i < 2; i++ {
				for j := 0; j < n; j++ {
					if e := dst[i][j] - want[i][j]; e > 1e-4 || e < -1e-4 {
						t.Fatalf("pair guarded (end %v, %v) stream %d element %d: %v, want %v", end, norm, i, j, dst[i][j], want[i][j])
					}
				}
			}
			// in place
			p.DCTQuad(src, src, norm)
			p.DCTPair(src[0], src[1], src[0], src[1], norm)
		}
	}
}
