package fft

import "github.com/go-fft/fft/internal/kernels"

// fused256Run transforms 256 points with the fused kernel and reports whether
// it did (skPlan.transform falls back to its passes on false: the kernel
// refuses on a CPU, or while a test has the AVX-512 flag off, where it cannot
// run). The other platforms' twin reports false.
func fused256Run(dst, src []complex128, inverse bool) bool {
	return kernels.Fused256(dst, src, inverse)
}
