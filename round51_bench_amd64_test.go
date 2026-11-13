package fft

import (
	"os"
	"strconv"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// Round 51: GOFFT_FORCE512=1 puts a Zen 4 back on the AVX-512 route that
// v0.28.11 took (split table uncapped, no radix-16 table), so the 512-bit
// kernels can be sampled and timed on it. A test-only switch: the library
// itself has none. It applies to BenchmarkRound51 only, and is restored when
// the benchmark ends (it used to run in init, for every test of the binary).
func BenchmarkRound51(b *testing.B) {
	isolatePlans(b)
	if mode := os.Getenv("GOFFT_FORCE512"); mode != "" {
		zm, zs := zen3RouteMax, zen3RouteSkip
		sp, sp512 := kernels.UseStockhamSplit, kernels.UseStockhamSplit512
		st, r16 := splitTable, radix16Table
		b.Cleanup(func() {
			zen3RouteMax, zen3RouteSkip = zm, zs
			kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = sp, sp512
			splitTable, radix16Table = st, r16
		})
		zen3RouteMax, zen3RouteSkip = 0, nil
		if mode == "split" {
			// Cascade Lake's route: the 512-bit split layout.
			kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = false, true
			splitTable = intelSplitTableAMD64(false, true, true)
			radix16Table = radix16TableAMD64(true, true, true)
		} else {
			kernels.UseStockhamSplit = false // v0.28.11's splitDefault on an AVX-512 CPU
			splitTable = intelSplitTableAMD64(kernels.UseStockhamSplit, kernels.UseStockhamSplit512, kernels.IntelCPU)
			radix16Table = radix16TableAMD64(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512, kernels.IntelCPU)
		}
	}
	for _, n := range []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 1000, 1080, 1296, 1920} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
		}
		p := NewPlan(n)
		dst := make([]complex128, n)
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				p.FFT(dst, x)
			}
		})
	}
}
