package kernels

import "unsafe"

// Fused DCT-II and DCT-III of 32 and 64 points as straight-line scalar programs
// in registers (go-fft/fft#162). dct23_amd64.s is written by internal/gendct23.
//
// The kernels are declared exported and without a Go wrapper: a body would be
// statements that only an AVX-512 CPU runs, and the coverage gate must hold on
// every CPU. They write dst while they read src, so an overlapping call goes
// through Run2x32 and its siblings, which copy src first. Callers check Dct23On.

// Dct2x32 writes the unnormalised DCT-II of the 32 points at src into dst.
//
//go:noescape
func Dct2x32(dst, src *float64)

// Dct3x32 writes the unnormalised DCT-III of the 32 points at src, src[0] taken
// times a0, into dst.
//
//go:noescape
func Dct3x32(dst, src *float64, a0 float64)

// Dct2x64 is Dct2x32 for 64 points.
//
//go:noescape
func Dct2x64(dst, src *float64)

// Dct3x64 is Dct3x32 for 64 points.
//
//go:noescape
func Dct3x64(dst, src *float64, a0 float64)

// Dct23On is whether the kernels run here: they use the 32 xmm registers of
// AVX-512 (the probe of UseStockhamAVX512) on an AMD Zen 4, the CPU they were
// measured on. Callers read it at every call.
var Dct23On = UseStockhamAVX512 && AMDZen4

// overlaps reports whether the n float64s at a and b share an address: the
// distance d = a - b, taken as unsigned, satisfies -8n < d < 8n exactly when
// d + 8n - 1 < 16n - 1 (one comparison, so that the Run helpers stay inlinable).
func overlaps(a, b unsafe.Pointer, n uintptr) bool {
	return uintptr(a)-uintptr(b)+8*n-1 < 16*n-1
}

// The Run helpers hold the Go around a kernel (the copy of src when dst overlaps
// it) with the kernel as a parameter, so that a test can put a pure-Go kernel in
// its place on a CPU without AVX-512. They are small enough to inline, and the
// caller passes a constant function (Dct2x32 itself): the call is then direct,
// dst and src do not escape, and the copy stays on the stack. A package-level
// function variable instead would make both escape (measured: one 256-byte
// allocation per in-place call). TestDCT23NoAllocs holds that.

// Run2x32 runs the DCT-II kernel k of 32 points; dst may alias src.
func Run2x32(dst, src *[32]float64, k func(dst, src *float64)) {
	if overlaps(unsafe.Pointer(dst), unsafe.Pointer(src), 32) {
		c := *src
		src = &c
	}
	k(&dst[0], &src[0])
}

// Run3x32 runs the DCT-III kernel k of 32 points, src[0] taken times a0.
func Run3x32(dst, src *[32]float64, a0 float64, k func(dst, src *float64, a0 float64)) {
	if overlaps(unsafe.Pointer(dst), unsafe.Pointer(src), 32) {
		c := *src
		src = &c
	}
	k(&dst[0], &src[0], a0)
}

// Run2x64 is Run2x32 for 64 points.
func Run2x64(dst, src *[64]float64, k func(dst, src *float64)) {
	if overlaps(unsafe.Pointer(dst), unsafe.Pointer(src), 64) {
		c := *src
		src = &c
	}
	k(&dst[0], &src[0])
}

// Run3x64 is Run3x32 for 64 points.
func Run3x64(dst, src *[64]float64, a0 float64, k func(dst, src *float64, a0 float64)) {
	if overlaps(unsafe.Pointer(dst), unsafe.Pointer(src), 64) {
		c := *src
		src = &c
	}
	k(&dst[0], &src[0], a0)
}
