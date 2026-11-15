//go:build !amd64

package kernels

// Fused1024On is false off amd64: there is no fused 1024-point kernel.
const Fused1024On = false

// Fused1024 never runs here: it reports false.
func Fused1024(dst, src *[1024]complex128, inverse bool) bool { return false }

// Fused1024Table is nil off amd64.
func Fused1024Table(inverse bool) []float64 { return nil }
