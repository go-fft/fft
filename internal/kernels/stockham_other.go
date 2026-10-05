//go:build !amd64 && !arm64

package kernels

// StockhamTwiddles returns nils: no pass kernel reads them.
func StockhamTwiddles(r, ido, l1 int, root []complex128) (fwd, conj []complex128) { return nil, nil }

// StockhamPass reports false: the fft package runs its scalar passes.
func StockhamPass(r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	return false
}

// Untangle returns 0: the fft package untangles every bin in Go.
func Untangle(dst, z, tw []complex128, m int) int { return 0 }

// Retangle returns 0: the fft package rebuilds every bin in Go.
func Retangle(z, x, tw []complex128, m int, h float64) int { return 0 }

// StockhamBatchPass reports false: the fft package runs its Go batched pass.
func StockhamBatchPass(r, ido, l1 int, cc, ch, tw []complex128, w, sIn, sOut int, inverse bool) bool {
	return false
}

// StockhamBatchKernels reports false: no batched pass kernels here.
func StockhamBatchKernels() bool { return false }

// StockhamSplitModes returns all zeros: the split layout between passes is
// arm64's (stockham_arm64.go).
func StockhamSplitModes(r, ido, l1 []int) []uint8 { return make([]uint8, len(r)) }

// StockhamPassLayout is StockhamPass: there is no split layout here, and the
// mode is always 0.
func StockhamPassLayout(mode uint8, r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	return StockhamPass(r, ido, l1, cc, ch, tw, inverse, wide)
}

// StockhamLastRun reports false: the fft package runs its Go final pass.
func StockhamLastRun(r int, cc, ch []complex128, os, runs, run, gap int, inverse, wide bool) bool {
	return false
}

// StockhamStrided reports false: the fft package runs whole passes instead.
func StockhamStrided(r int, cc, ch, tw []complex128, cnt, nb, sin, bin, sout, bout, btw int, first0, firstRest, inverse, wide bool) bool {
	return false
}
