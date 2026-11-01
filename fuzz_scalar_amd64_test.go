package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// FuzzPublicAPIScalar is FuzzPublicAPI with every amd64 assembly route turned
// off, so the pure-Go reference passes are fuzzed on a CPU that has AVX2.
func FuzzPublicAPIScalar(f *testing.F) {
	fuzzPublicAPISeeds(f)
	kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = false, false
	kernels.UseStockhamBatchAVX2, kernels.UseStockham32 = false, false
	kernels.UseStockhamBatch32, kernels.UseUntangle32 = false, false
	kernels.UseUntangleAVX2, kernels.UseUntangleAVX512 = false, false
	kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = false, false
	kernels.UseFMA = false
	f.Fuzz(fuzzPublicAPI)
}
