//go:build !amd64

package fft

// fmaSplitDefault: there are no FMA kernels here, so a split pass keeps its
// kernel (arm64's NEON ones) rather than its Go fallback.
func fmaSplitDefault() bool { return false }
