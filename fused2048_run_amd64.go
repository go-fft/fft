package fft

import "github.com/go-fft/fft/internal/kernels"

// fused2048Run transforms 2048 points with the fused kernel and reports whether
// it did (the caller falls back to its passes on false). The slices convert to
// array pointers, which panics on a short one. The other platforms' twin reports
// false.
func fused2048Run(dst, src []complex128, inverse bool) bool {
	return kernels.Fused2048((*[2048]complex128)(dst), (*[2048]complex128)(src), inverse)
}
