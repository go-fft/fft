package kernels

// Prototype (Round 43, not for merge): BalProto swaps the kernels of the
// default 1024-point plan on AMD (8·8·16: skSplit8ISFwd, skSplit8SIFwd,
// skLast16AVX2) for copies with one addition in every v moved to the FMA
// pipes (balproto_amd64.s); v = 0 restores the originals.

//go:noescape
func skSplit8ISFwdBal2(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SIFwdBal2(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skLast16AVX2Bal2(cc, ch *complex128, k *float64, l1 int)

//go:noescape
func skSplit8ISFwdBal3(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SIFwdBal3(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skLast16AVX2Bal3(cc, ch *complex128, k *float64, l1 int)

//go:noescape
func skSplit8ISFwdBal4(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SIFwdBal4(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skLast16AVX2Bal4(cc, ch *complex128, k *float64, l1 int)

// BalProto selects the kernel set: 0 originals, 2, 3 or 4 the copies.
func BalProto(v int) {
	switch v {
	case 2:
		skSplitFwd[8][splitOut], skSplitFwd[8][splitIn], skLast256[16] = skSplit8ISFwdBal2, skSplit8SIFwdBal2, skLast16AVX2Bal2
	case 3:
		skSplitFwd[8][splitOut], skSplitFwd[8][splitIn], skLast256[16] = skSplit8ISFwdBal3, skSplit8SIFwdBal3, skLast16AVX2Bal3
	case 4:
		skSplitFwd[8][splitOut], skSplitFwd[8][splitIn], skLast256[16] = skSplit8ISFwdBal4, skSplit8SIFwdBal4, skLast16AVX2Bal4
	default:
		skSplitFwd[8][splitOut], skSplitFwd[8][splitIn], skLast256[16] = skSplit8ISFwd, skSplit8SIFwd, skLast16AVX2
	}
}
