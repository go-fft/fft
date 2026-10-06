//go:build amd64 || arm64

package kernels

// StockhamBatchPass32 runs one float32 Stockham pass of radix r over a batch
// of w transforms laid out side by side (Round 25), and reports true, or
// reports false (and does nothing) when no kernel can: no AVX2 on amd64, or a
// radix without a kernel. Point p of the batch is the w values
// cc[p·sIn : p·sIn+w] on input and ch[p·sOut : p·sOut+w] on output (sIn,
// sOut >= w), so a pass reads or writes a strip of an N-D array in place. tw
// is the fft package's batched twiddle table for the direction: for
// i = 1 .. ido-1, the r-1 twiddles of point i (unused when ido == 1). Every
// value of the batch gets the arithmetic of the 1-D float32 kernels
// (StockhamPass32), which is the Go pass's. It panics on a layout its bound
// checks cannot vouch for, before any kernel runs.
func StockhamBatchPass32(r, ido, l1 int, cc, ch, tw []complex64, w, sIn, sOut int, inverse bool) bool {
	if w < 1 || sIn < w || sOut < w || ido < 1 || l1 < 1 {
		// The kernels walk p·sIn+j for j < w; a stride below w, or an empty
		// batch, is a layout the length check below cannot vouch for.
		panic("kernels: StockhamBatchPass32: need w >= 1, sIn and sOut >= w, ido and l1 >= 1")
	}
	if !UseStockhamBatch32 || !stockham32Kernel(r, ido, l1) {
		return false
	}
	n := r * ido * l1
	_, _ = cc[(n-1)*sIn+w-1], ch[(n-1)*sOut+w-1] // the kernels trust these lengths
	tp := (*complex64)(nil)
	if ido > 1 {
		_ = tw[(ido-1)*(r-1)-1]
		tp = &tw[0]
	}
	f32rBatch(r, inverse, &cc[0], &ch[0], tp, ido, l1, w/4, w&3, 8*ido*sIn, 8*l1*ido*sOut, 8*(sIn-w), 8*(sOut-w))
	return true
}
