//go:build !amd64

package kernels

// Fused256On is false off amd64: there is no fused kernel.
const Fused256On = false

// Fused256 never runs here: it reports false.
func Fused256(dst, src *[256]complex128, inverse bool) bool { return false }

// Fused256Ref is the reference schedule; here it never runs.
func Fused256Ref(dst, src *[256]complex128, inverse bool) bool { return false }

// Fused256Table is nil off amd64.
func Fused256Table(inverse bool) []float64 { return nil }
