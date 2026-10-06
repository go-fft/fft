package fft

// The split layout's factorizations (Round 23). Where the pass kernels keep a
// power of two's data block-split between passes (kernels.StockhamSplitModes,
// on amd64 with AVX2 and without AVX-512), a few factorizations run faster
// than the ones skFactorize picks for the interleaved passes: splitTable
// holds them. It is per-architecture (route_amd64.go; nil elsewhere).
var splitTable = splitTableDefault()

// splitPow2Factors is the split layout's factorization of 2^e (e >= 4): a
// final radix-4 pass, radix-8 passes for as many of the other bits as leave
// an even remainder, and radix-4 passes for that remainder, first. Every
// pass but the last then runs split, and the final pass, which reads
// interleaved data, is the cheapest one per point. Timed on Zen 3 against
// every ordering of radix-4 and radix-8 passes closed by a radix-4, -8 or
// -16 pass (Round 23), it was the best or within 3% of it from 65536 points
// up; below, where a pass's data stays in L2, other orders won by up to 7%.
func splitPow2Factors(e int) []int {
	rest := e - 2
	n8 := rest / 3
	if (rest-3*n8)%2 != 0 {
		n8--
	}
	var f []int
	for range (rest - 3*n8) / 2 {
		f = append(f, 4)
	}
	for range n8 {
		f = append(f, 8)
	}
	return append(f, 4)
}
