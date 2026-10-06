package kernels

// UseStockhamBatch32 reports whether StockhamBatchPass32 takes the NEON
// kernels (genF32rBatchNEON); a variable so the tests can compare them with
// the Go batched pass.
var UseStockhamBatch32 = true

// StockhamBatchKernels32 reports true: the float32 batched pass kernels are
// NEON, part of the arm64 baseline, so the fft package runs a non-contiguous
// float32 axis as batched passes instead of gathering its lines.
func StockhamBatchKernels32() bool { return true }

type sk32BatchFn func(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

var (
	sk32BatchNEON    = [9]sk32BatchFn{2: sk32Batch2NEON, 3: sk32Batch3NEON, 4: sk32Batch4NEON, 5: sk32Batch5NEON, 8: sk32Batch8NEON}
	sk32BatchNEONInv = [9]sk32BatchFn{2: sk32Batch2NEONInv, 3: sk32Batch3NEONInv, 4: sk32Batch4NEONInv, 5: sk32Batch5NEONInv, 8: sk32Batch8NEONInv}
)

func f32rBatch(r int, inverse bool, cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int) {
	fn := sk32BatchNEON[r]
	if inverse {
		fn = sk32BatchNEONInv[r]
	}
	fn(cc, ch, tw, ido, l1, quads, rem, jin, jout, adjin, adjout)
}

//go:noescape
func sk32Batch2NEON(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch3NEON(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch4NEON(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch5NEON(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch8NEON(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch2NEONInv(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch3NEONInv(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch4NEONInv(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch5NEONInv(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)

//go:noescape
func sk32Batch8NEONInv(cc, ch, tw *complex64, ido, l1, quads, rem, jin, jout, adjin, adjout int)
