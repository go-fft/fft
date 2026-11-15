//go:build !amd64

package fft

// fusedSRun is amd64's fused kernel; here it never runs.
func fusedSRun(n int, dst, src []complex64, inverse bool) bool { return false }
