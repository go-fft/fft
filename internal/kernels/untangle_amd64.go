package kernels

// UntangleSteps selects the untangle and retangle kernels by the number of
// steps they interleave (1: untangleAVX2 and retangleAVX2; 2, 3: Round 30's).
// Exploration only.
var UntangleSteps = 1

//go:noescape
func smallUntangle2AVX2(dst, z, tw *complex128, k *float64, m, pairs int)

//go:noescape
func smallRetangle2AVX2(z, x, tw *complex128, k *float64, h *[4]float64, m, pairs int)

//go:noescape
func smallUntangle3AVX2(dst, z, tw *complex128, k *float64, m, pairs int)

//go:noescape
func smallRetangle3AVX2(z, x, tw *complex128, k *float64, h *[4]float64, m, pairs int)
