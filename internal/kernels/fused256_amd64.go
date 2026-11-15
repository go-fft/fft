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
	tw := (*complex128)(unsafe.Pointer(&fused256Tw[d][0]))
	if inverse {
		fused256Inv(&dst[0], &src[0], tw)
		return
	}
	fused256Fwd(&dst[0], &src[0], tw)
}
