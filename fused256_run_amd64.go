package fft

import "github.com/go-fft/fft/internal/kernels"

// fused256Run transforms 256 points with the fused kernel and reports that it
// did (skPlan.transform's condition; the other platforms' twin reports false).
func fused256Run(dst, src []complex128, inverse bool) bool {
	kernels.Fused256(dst, src, inverse)
	return true
}
