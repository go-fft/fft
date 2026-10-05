package fft

import (
	"fmt"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// BenchmarkSKPassNEON times every pass a NEON kernel can run, one pass at a
// time, on its Go code ("go") and its kernel ("neon"), with the two buffers
// placed as the transform places them (scratch off dst's L1 sets). It is the
// per-pass measurement the arm64 routing rule was decided on; names are
// n/r=R/ido=I/l1=L/{go,neon}.
func BenchmarkSKPassNEON(b *testing.B) {
	defer func(v bool) { kernels.UseStockhamNEON = v }(kernels.UseStockhamNEON)
	for _, n := range []int{64, 256, 1024, 4096, 16384, 65536, 1 << 18, 1 << 20, 2048, 8192, 32768} {
		e := 0
		for m := n; m > 1; m >>= 1 {
			e++
		}
		p := newSKPlanFactors(n, pow2Radices(e, n <= pow2OneRadix8Max))
		dst := make([]complex128, n)
		buf := make([]complex128, n+setSpan)
		scr := offTheSets(buf, dst, n)
		copy(dst, cmplxSignal(n))
		for k := range p.stages {
			st := &p.stages[k]
			if !kernels.StockhamPass(st.r, st.ido, st.l1, dst, scr, st.twX, false, false) {
				continue
			}
			for _, neon := range []bool{false, true} {
				name := "go"
				if neon {
					name = "neon"
				}
				b.Run(fmt.Sprintf("%d/r=%d/ido=%d/l1=%d/%s", n, st.r, st.ido, st.l1, name), func(b *testing.B) {
					kernels.UseStockhamNEON = neon
					b.SetBytes(int64(16 * n))
					for b.Loop() {
						st.pass(scr, dst, false)
					}
				})
			}
		}
	}
}
