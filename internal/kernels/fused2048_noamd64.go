//go:build !amd64

package kernels

// Fused2048On is false off amd64: there is no fused 2048-point kernel.
const Fused2048On = false

// Fused2048 never runs here: it reports false.
func Fused2048(dst, src *[2048]complex128, inverse bool) bool { return false }

// Fused2048Table is nil off amd64.
func Fused2048Table(inverse bool) []float64 { return nil }
