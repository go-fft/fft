package kernels

// UseStockhamBatch32 reports whether StockhamBatchPass32 takes the AVX2
// kernels (genF32rBatchAVX2); a variable so the tests can compare them with
// the Go batched pass.
var UseStockhamBatch32 = useAVX2

// StockhamBatchKernels32 reports whether StockhamBatchPass32 has kernels on
// this machine (AVX2), which is when the fft package runs a non-contiguous
// float32 axis as batched passes instead of gathering its lines.
func StockhamBatchKernels32() bool { return useAVX2 }

type sk32BatchFn func(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)

var sk32Batch = [9]sk32BatchFn{2: sk32Batch2AVX2, 3: sk32Batch3AVX2, 4: sk32Batch4AVX2, 5: sk32Batch5AVX2, 8: sk32Batch8AVX2}

func f32rBatch(r int, inverse bool, cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int) {
	k := &sk32Fwd[0][0]
	if inverse {
		k = &sk32Inv[0][0]
	}
	sk32Batch[r](cc, ch, tw, k, ido, l1, quads, rem, jin, jout, adjin, adjout)
}

//go:noescape
func sk32Batch2AVX2(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch3AVX2(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch4AVX2(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch5AVX2(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch8AVX2(cc, ch, tw *complex64, k *float32, ido, l1, quads, rem, jin, jout, adjin, adjout int)
