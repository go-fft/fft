//go:build !amd64

package kernels

// Dct23On is false off amd64: there is no fused kernel.
const Dct23On = false

// Dct2x32 never runs here.
func Dct2x32(dst, src *[32]float64) {}

// Dct3x32 never runs here.
func Dct3x32(dst, src *[32]float64, a0 float64) {}

// Dct2x64 never runs here.
func Dct2x64(dst, src *[64]float64) {}

// Dct3x64 never runs here.
func Dct3x64(dst, src *[64]float64, a0 float64) {}
