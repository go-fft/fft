//go:build !amd64

package fft

// splitTableDefault is nil: the split factorizations are amd64's. arm64 keeps
// its data split too (Round 21), with skFactorize's factorizations.
func splitTableDefault() map[int][]int { return nil }
