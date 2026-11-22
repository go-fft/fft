package kernels

import "unsafe"

// Fused DCT-II and DCT-III of 32 and 64 points as straight-line scalar programs
// in registers (go-fft/fft#162). dct23_amd64.s is written by internal/gendct23.

//go:noescape
func dct2x32(dst, src *float64)

//go:noescape
func dct3x32(dst, src *float64, a0 float64)

//go:noescape
func dct2x64(dst, src *float64)

//go:noescape
func dct3x64(dst, src *float64, a0 float64)

// The kernels' entry points, behind variables so that the Go around them (the
// overlap copy of Dct2x32 and its siblings, and the routing in the fft package)
// runs, and is tested, on a CPU without AVX-512 with a pure-Go kernel in place of
// the assembly. Only tests replace them; the CPU gate is Dct23On.
var (
	Dct2x32Fn func(dst, src *float64)            = dct2x32
	Dct3x32Fn func(dst, src *float64, a float64) = dct3x32
	Dct2x64Fn func(dst, src *float64)            = dct2x64
	Dct3x64Fn func(dst, src *float64, a float64) = dct3x64
)

// Dct23On is whether the kernels run here: they use the 32 xmm registers of
// AVX-512 (the probe of UseStockhamAVX512) on an AMD Zen 4, the CPU they were
// measured on. Callers read it at every call.
var Dct23On = UseStockhamAVX512 && AMDZen4

// overlaps reports whether the n float64s at a and b share an address.
func overlaps(a, b *float64, n uintptr) bool {
	pa, pb := uintptr(unsafe.Pointer(a)), uintptr(unsafe.Pointer(b))
	return pa < pb+8*n && pb < pa+8*n
}

// detach returns src, or a copy of its n values when dst overlaps it, so the
// kernel reads all of src before it writes dst.
func detach(dst, src *float64, n int) *float64 {
	if overlaps(dst, src, uintptr(n)) {
		c := make([]float64, n)
		copy(c, unsafe.Slice(src, n))
		return &c[0]
	}
	return src
}

// Dct2x32 writes the unnormalised DCT-II of src into dst. dst may alias src
// (a copy of src is taken then). Callers check Dct23On.
func Dct2x32(dst, src *[32]float64) {
	Dct2x32Fn(&dst[0], detach(&dst[0], &src[0], 32))
}

// Dct3x32 writes the unnormalised DCT-III of src, src[0] taken times a0, into dst.
func Dct3x32(dst, src *[32]float64, a0 float64) {
	Dct3x32Fn(&dst[0], detach(&dst[0], &src[0], 32), a0)
}

// Dct2x64 is Dct2x32 for 64 points.
func Dct2x64(dst, src *[64]float64) {
	Dct2x64Fn(&dst[0], detach(&dst[0], &src[0], 64))
}

// Dct3x64 is Dct3x32 for 64 points.
func Dct3x64(dst, src *[64]float64, a0 float64) {
	Dct3x64Fn(&dst[0], detach(&dst[0], &src[0], 64), a0)
}
