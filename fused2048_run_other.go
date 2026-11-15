//go:build !amd64

package fft

// fused2048Run is amd64's fused kernel; here it never runs.
func fused2048Run(dst, src []complex128, inverse bool) bool { return false }
