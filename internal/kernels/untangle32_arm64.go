package kernels

// UseUntangle32 reports whether Untangle32 and Retangle32 take the NEON
// kernels: always, NEON being part of the arm64 baseline; a variable so the
// tests can compare them with the Go loops.
var UseUntangle32 = true

func f32rUntangle(dst, z, tw *complex64, m, quads int) {
	f32rUntangleNEON(dst, z, tw, m, quads)
}

func f32rRetangle(z, x, tw *complex64, h float32, m, quads int) {
	f32rRetangleNEON(z, x, tw, &h, m, quads)
}

//go:noescape
func f32rUntangleNEON(dst, z, tw *complex64, m, quads int)

//go:noescape
func f32rRetangleNEON(z, x, tw *complex64, h *float32, m, quads int)
