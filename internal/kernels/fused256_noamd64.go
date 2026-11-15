//go:build !amd64

package kernels

// Fused256On is false off amd64: there is no fused kernel.
const Fused256On = false

// Fused256 never runs here (Fused256On is false).
func Fused256(dst, src []complex128, inverse bool) {}

// Fused256Table is nil off amd64.
func Fused256Table(inverse bool) []float64 { return nil }
