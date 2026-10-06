package kernels

// UseUntangle32 reports whether Untangle32 and Retangle32 take the AVX2
// kernels; a variable so the tests can compare them with the Go loops.
var UseUntangle32 = useAVX2

func f32rUntangle(dst, z, tw *complex64, m, quads int) {
	f32rUntangleAVX2(dst, z, tw, &sk32Fwd[0][0], m, quads)
}

func f32rRetangle(z, x, tw *complex64, h float32, m, quads int) {
	hv := [8]float32{h, h, h, h, h, h, h, h}
	f32rRetangleAVX2(z, x, tw, &sk32Fwd[0][0], &hv[0], m, quads)
}

//go:noescape
func f32rUntangleAVX2(dst, z, tw *complex64, k *float32, m, quads int)

//go:noescape
func f32rRetangleAVX2(z, x, tw *complex64, k, h *float32, m, quads int)
