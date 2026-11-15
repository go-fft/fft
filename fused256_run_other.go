//go:build !amd64

package fft

// fused256Run is amd64's fused kernel; here it never runs.
func fused256Run(dst, src []complex128, inverse bool) bool { return false }
