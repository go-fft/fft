//go:build !amd64

package fft

// fused1024Run is amd64's fused kernel; here it never runs.
func fused1024Run(dst, src []complex128, inverse bool) bool { return false }
