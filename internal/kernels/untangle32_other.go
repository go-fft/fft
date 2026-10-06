//go:build !amd64 && !arm64

package kernels

// UseUntangle32 is false: there are no float32 untangle kernels here.
var UseUntangle32 = false

// Untangle32 returns 0: the fft package untangles every float32 bin in Go.
func Untangle32(dst, z, tw []complex64, m int) int { return 0 }

// Retangle32 returns 0: the fft package rebuilds every float32 bin in Go.
func Retangle32(z, x, tw []complex64, m int, h float32) int { return 0 }
