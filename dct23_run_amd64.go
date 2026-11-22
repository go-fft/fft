package fft

import "github.com/go-fft/fft/internal/kernels"

// dct23Run2 runs the generated DCT-II kernel of n points (32 or 64) and reports
// whether it did; it is false where the kernels are off or serve no such length.
// It is one statement that every CPU runs: the assembly functions are passed as
// they are, and the routes below hold the rest, which a test runs with pure-Go
// kernels on a CPU without AVX-512.
func dct23Run2(dst, src []float64, n int) bool {
	on := kernels.Dct23On
	return dct23Route2x32(dst, src, on && n == 32, kernels.Dct2x32) || dct23Route2x64(dst, src, on && n == 64, kernels.Dct2x64)
}

// dct23Run3 is dct23Run2 for the DCT-III, src[0] taken times a0.
func dct23Run3(dst, src []float64, n int, a0 float64) bool {
	on := kernels.Dct23On
	return dct23Route3x32(dst, src, a0, on && n == 32, kernels.Dct3x32) || dct23Route3x64(dst, src, a0, on && n == 64, kernels.Dct3x64)
}

// The routes run their kernel when ok. They are small enough to inline into
// dct23Run2 and dct23Run3, where the kernel is a constant: the call is then
// direct, so dst and src do not escape and the copy of an aliased src stays on the
// stack (TestDCT23NoAllocs).

func dct23Route2x32(dst, src []float64, ok bool, k func(dst, src *float64)) bool {
	if ok {
		kernels.Run2x32((*[32]float64)(dst), (*[32]float64)(src), k)
	}
	return ok
}

func dct23Route2x64(dst, src []float64, ok bool, k func(dst, src *float64)) bool {
	if ok {
		kernels.Run2x64((*[64]float64)(dst), (*[64]float64)(src), k)
	}
	return ok
}

func dct23Route3x32(dst, src []float64, a0 float64, ok bool, k func(dst, src *float64, a0 float64)) bool {
	if ok {
		kernels.Run3x32((*[32]float64)(dst), (*[32]float64)(src), a0, k)
	}
	return ok
}

func dct23Route3x64(dst, src []float64, a0 float64, ok bool, k func(dst, src *float64, a0 float64)) bool {
	if ok {
		kernels.Run3x64((*[64]float64)(dst), (*[64]float64)(src), a0, k)
	}
	return ok
}
