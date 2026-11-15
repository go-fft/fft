package kernels

import (
	"math"
	"unsafe"
)

func ptr(p *float64) unsafe.Pointer { return unsafe.Pointer(p) }

// Prototype (go-fft/fft#124): complex128 256-point FFT as two fused 16x16
// AVX-512 passes; fused256_amd64.s is written by internal/genfused256.

//go:noescape
func fused256Fwd(dst, src, tw *complex128)

//go:noescape
func fused256Inv(dst, src, tw *complex128)

// fused256Tw[inv] is the table of w256^(k1*b), k1 0..15, b 0..15, in groups
// of four b: per (k1,group) 64 bytes of the real part duplicated, then 64 of
// the imaginary part duplicated, signed for the direction (forward e^-i).
var fused256Tw = func() (t [2][]float64) {
	for d := 0; d < 2; d++ {
		t[d] = make([]float64, 64*16)
		for k1 := 0; k1 < 16; k1++ {
			for g := 0; g < 4; g++ {
				base := (k1*4 + g) * 16
				for l := 0; l < 4; l++ {
					b := 4*g + l
					s, c := math.Sincos(2 * math.Pi * float64(k1*b) / 256)
					if d == 0 {
						s = -s
					}
					t[d][base+2*l], t[d][base+2*l+1] = c, c
					t[d][base+8+2*l], t[d][base+8+2*l+1] = s, s
				}
			}
		}
	}
	return
}()

// Fused256Table is the twiddle table of one direction, for the tests that
// perturb it to show what the accuracy bound can see.
func Fused256Table(inverse bool) []float64 {
	if inverse {
		return fused256Tw[1]
	}
	return fused256Tw[0]
}

// Fused256On is whether the fused kernel runs here: AVX-512 (F and DQ, the
// probe of UseStockhamAVX512) on an AMD Zen 4, the only CPU it was measured
// on (go-fft/fft#124). Intel's AVX-512 route is unmeasured and keeps its plans.
var Fused256On = UseStockhamAVX512 && AMDZen4

// Fused256 transforms the first 256 points of src into dst (which may alias
// src); inverse is unnormalized. The caller checks UseStockhamAVX512.
func Fused256(dst, src []complex128, inverse bool) {
	_ = dst[255]
	_ = src[255]
	if inverse {
		fused256Inv(&dst[0], &src[0], (*complex128)(ptr(&fused256Tw[1][0])))
		return
	}
	fused256Fwd(&dst[0], &src[0], (*complex128)(ptr(&fused256Tw[0][0])))
}
