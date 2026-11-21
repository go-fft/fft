package fft

import "github.com/go-fft/fft/internal/kernels"

// dct23Run2 runs the generated DCT-II kernel of n points (32 or 64) and reports
// whether it did; it is false where the kernels are off or serve no such length.
func dct23Run2(dst, src []float64, n int) bool {
	if !kernels.Dct23On {
		return false
	}
	switch n {
	case 32:
		kernels.Dct2x32((*[32]float64)(dst), (*[32]float64)(src))
		return true
	case 64:
		kernels.Dct2x64((*[64]float64)(dst), (*[64]float64)(src))
		return true
	}
	return false
}

// dct23Run3 is dct23Run2 for the DCT-III, src[0] taken times a0.
func dct23Run3(dst, src []float64, n int, a0 float64) bool {
	if !kernels.Dct23On {
		return false
	}
	switch n {
	case 32:
		kernels.Dct3x32((*[32]float64)(dst), (*[32]float64)(src), a0)
		return true
	case 64:
		kernels.Dct3x64((*[64]float64)(dst), (*[64]float64)(src), a0)
		return true
	}
	return false
}
