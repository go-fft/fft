package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// FuzzPublicAPIScalar is FuzzPublicAPI with every amd64 assembly route turned
// off, so the pure-Go reference passes are fuzzed on a CPU that has AVX2.
func FuzzPublicAPIScalar(f *testing.F) {
	fuzzPublicAPISeeds(f)
	isolatePlans(f) // the plans cached under the scalar route must not outlive it
	saved := [...]bool{
		kernels.UseStockhamAVX2, kernels.UseStockhamAVX512, kernels.UseStockhamBatchAVX2, kernels.UseStockham32,
		kernels.UseStockhamBatch32, kernels.UseUntangle32, kernels.UseUntangleAVX2, kernels.UseUntangleAVX512,
		kernels.UseStockhamSplit, kernels.UseStockhamSplit512, kernels.UseFMA,
	}
	f.Cleanup(func() {
		kernels.UseStockhamAVX2, kernels.UseStockhamAVX512, kernels.UseStockhamBatchAVX2, kernels.UseStockham32 = saved[0], saved[1], saved[2], saved[3]
		kernels.UseStockhamBatch32, kernels.UseUntangle32, kernels.UseUntangleAVX2, kernels.UseUntangleAVX512 = saved[4], saved[5], saved[6], saved[7]
		kernels.UseStockhamSplit, kernels.UseStockhamSplit512, kernels.UseFMA = saved[8], saved[9], saved[10]
	})
	kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = false, false
	kernels.UseStockhamBatchAVX2, kernels.UseStockham32 = false, false
	kernels.UseStockhamBatch32, kernels.UseUntangle32 = false, false
	kernels.UseUntangleAVX2, kernels.UseUntangleAVX512 = false, false
	kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = false, false
	kernels.UseFMA = false
	f.Fuzz(fuzzPublicAPI)
}
