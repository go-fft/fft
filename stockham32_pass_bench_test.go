//go:build amd64 || arm64

package fft

import (
	"fmt"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// BenchmarkSK32Pass times every float32 pass a kernel can run, one pass at a
// time, on its Go code ("go") and its kernel ("simd"), for the factorization
// Plan32 uses: the per-pass measurement of Round 22. Names are
// n/r=R/ido=I/l1=L/{go,simd}.
func BenchmarkSK32Pass(b *testing.B) {
	if !kernels.UseStockham32 {
		b.Skip("no float32 pass kernels on this machine")
	}
	defer func(v bool) { kernels.UseStockham32 = v }(kernels.UseStockham32)
	for _, n := range []int{64, 256, 1024, 4096, 16384, 65536, 1 << 18, 1 << 20, 512, 2048, 8192, 32768, 1000, 1080, 1296, 1920, 20160} {
		p := newSKPlan32(n)
		dst := make([]complex64, n)
		scr := make([]complex64, n)
		for i, v := range cmplxSignal(n) {
			dst[i] = complex64(v)
		}
		for k := range p.stages {
			st := &p.stages[k]
			kernels.UseStockham32 = true
			if !kernels.StockhamPass32(st.r, st.ido, st.l1, dst, scr, st.twK, false) {
				continue
			}
			for _, simd := range []bool{false, true} {
				name := "go"
				if simd {
					name = "simd"
				}
				b.Run(fmt.Sprintf("%d/r=%d/ido=%d/l1=%d/%s", n, st.r, st.ido, st.l1, name), func(b *testing.B) {
					kernels.UseStockham32 = simd
					b.SetBytes(int64(8 * n))
					for b.Loop() {
						st.pass(scr, dst, false)
					}
				})
			}
		}
	}
}
