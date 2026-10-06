//go:build !amd64 && !arm64

package kernels

// UseStockhamBatch32 is false: there are no float32 batched kernels here.
var UseStockhamBatch32 = false

// StockhamBatchPass32 reports false: the fft package runs its Go batched pass.
func StockhamBatchPass32(r, ido, l1 int, cc, ch, tw []complex64, w, sIn, sOut int, inverse bool) bool {
	return false
}

// StockhamBatchKernels32 reports false: no float32 batched pass kernels here.
func StockhamBatchKernels32() bool { return false }
