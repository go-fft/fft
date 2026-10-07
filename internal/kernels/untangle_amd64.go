package kernels

// The float64 untangle and retangle kernels (untangle_amd64.s, Round 30):
// genSmallUntangleFile in asmgen/amd64/gen.go. Untangle and Retangle
// (stockham_amd64.go) call them.

//go:noescape
func smallUntangleAVX2(dst, z, tw *complex128, k *float64, m, pairs int)

//go:noescape
func smallRetangleAVX2(z, x, tw *complex128, k *float64, h *[4]float64, m, pairs int)
