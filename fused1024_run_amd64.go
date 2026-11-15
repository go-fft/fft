package fft

import "github.com/go-fft/fft/internal/kernels"

// fused1024Run transforms 1024 points with the fused kernel and reports whether
// it did (the caller falls back to its passes on false). The slices convert to
// array pointers, which panics on a short one. The other platforms' twin reports
// false.
func fused1024Run(dst, src []complex128, inverse bool) bool {
	return kernels.Fused1024((*[1024]complex128)(dst), (*[1024]complex128)(src), inverse)
}
