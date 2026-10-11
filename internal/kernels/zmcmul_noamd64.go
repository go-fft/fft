//go:build !amd64

package kernels

// zmCMulBulk writes nothing here: CMulTo runs its scalar loop.
func zmCMulBulk(dst, a, b []complex128) int { return 0 }
