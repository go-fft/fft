//go:build !amd64

package kernels

// UseFMA is false: the FMA kernels are amd64's (fma_amd64.go). On arm64 the
// default kernels already fuse what the Go compiler fuses.
var UseFMA = false

// StockhamPassFMA reports false: the fft package runs its Go pass.
func StockhamPassFMA(r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	return false
}

// StockhamPassLayoutFMA reports false: the fft package runs its Go pass.
func StockhamPassLayoutFMA(mode uint8, r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	return false
}

// StockhamBatchPassFMA reports false: the fft package runs its Go pass.
func StockhamBatchPassFMA(r, ido, l1 int, cc, ch, tw []complex128, w, sIn, sOut int, inverse bool) bool {
	return false
}

// StockhamStridedFMA reports false: the fft package runs its Go pass.
func StockhamStridedFMA(r int, cc, ch, tw []complex128, cnt, nb, sin, bin, sout, bout, btw int, first0, firstRest, inverse, wide bool) bool {
	return false
}

// ZnPlanForWideFMA returns nil: there is no frame kernel here.
func ZnPlanForWideFMA(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128, wide bool, fma []bool) *ZnPlan {
	return nil
}

// StockhamLastRunFMA reports false: the fft package runs its Go pass.
func StockhamLastRunFMA(r int, cc, ch []complex128, os, runs, run, gap int, inverse, wide bool) bool {
	return false
}
