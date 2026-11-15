package kernels

import "unsafe"

// Complex128 256-point FFT as two fused 16x16 AVX-512 passes (go-fft/fft#124).
// fused256_amd64.s and fused256_tw_amd64.go (the twiddle table, correctly
// rounded) are written by internal/genfused256.

// fused256 transforms 256 points (both directions in one entry) and reports
// whether it did: its first instructions read Fused256On and UseStockhamAVX512,
// at every call, and return false before any AVX-512 instruction when either is
// off. The caller then runs its own passes (ZnPlan.Run's pattern).
//
//go:noescape
func fused256(dst, src, tw *complex128, inv bool) (ok bool)

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
// The kernel reads it, and UseStockhamAVX512, at every call.
var Fused256On = UseStockhamAVX512 && AMDZen4

// Fused256 transforms the first 256 points of src into dst (which may alias
// src; inverse is unnormalized) and reports whether the kernel ran: false, with
// dst untouched, when Fused256On or UseStockhamAVX512 is off.
func Fused256(dst, src []complex128, inverse bool) bool {
	_ = dst[255]
	_ = src[255]
	d := 0
	if inverse {
		d = 1
	}
	return fused256(&dst[0], &src[0], (*complex128)(unsafe.Pointer(&fused256Tw[d][0])), inverse)
}
