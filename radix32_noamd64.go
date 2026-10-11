//go:build !amd64

package fft

// r32TableDefault is nil: only amd64 has a radix-32 pass kernel, and the Go
// radix-32 pass is its reference, not a faster pass.
func r32TableDefault() map[int][]int { return nil }
