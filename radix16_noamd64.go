//go:build !amd64

package fft

// radix16TableDefault is nil: only amd64 has a radix-16 pass kernel, and the
// Go radix-16 pass is its reference, not a faster pass.
func radix16TableDefault() map[int][]int { return nil }
