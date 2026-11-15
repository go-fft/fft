package kernels

import "unsafe"

// Complex128 256-point FFT as two fused 16x16 AVX-512 passes (go-fft/fft#124).
// fused256_amd64.s and fused256_tw_amd64.go (the twiddle table, correctly
// rounded) are written by internal/genfused256.

//go:noescape
func fused256Fwd(dst, src, tw *complex128)

//go:noescape
func fused256Inv(dst, src, tw *complex128)

// Fused256Table is the twiddle table of one direction, for the tests that
// perturb it to show what the accuracy bound can see.
func Fused256Table(inverse bool) []float64 {
	if inverse {
		return fused256Tw[1][:]
	}
	return fused256Tw[0][:]
}

// fused256Impl is the assembly, per direction. A variable only so that the
// tests can put a Go stand-in in its place on a CPU without AVX-512 (the
// coverage gate runs on such hosts, and the wrappers' statements must run).
var fused256Impl = [2]func(dst, src, tw *complex128){fused256Fwd, fused256Inv}

// SetFused256Impl replaces the kernels (forward, inverse) and returns a
// function that puts the assembly back. For tests only.
func SetFused256Impl(f [2]func(dst, src, tw *complex128)) (restore func()) {
	old := fused256Impl
	fused256Impl = f
	return func() { fused256Impl = old }
}

// Fused256On is whether the fused kernel runs here: AVX-512 (F and DQ, the
// probe of UseStockhamAVX512) on an AMD Zen 4, the only CPU it was measured
// on (go-fft/fft#124). Intel's AVX-512 route is unmeasured and keeps its plans.
var Fused256On = UseStockhamAVX512 && AMDZen4

// Fused256 transforms the first 256 points of src into dst (which may alias
// src); inverse is unnormalized. The caller checks Fused256On.
func Fused256(dst, src []complex128, inverse bool) {
	_ = dst[255]
	_ = src[255]
	d := 0
	if inverse {
		d = 1
	}
	fused256Impl[d](&dst[0], &src[0], (*complex128)(unsafe.Pointer(&fused256Tw[d][0])))
}
