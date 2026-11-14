package fft

import (
	"strconv"
	"testing"
)

// BenchmarkMDCTFoldPath times the MDCT and the IMDCT with the fused Go loops and
// with the fold/unfold kernels, at every size, to place the window.
func BenchmarkMDCTFoldPath(b *testing.B) {
	isolatePlans(b)
	sf, sfx, su, sux := mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax
	defer func() { mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = sf, sfx, su, sux }()
	for _, n := range []int{64, 124, 128, 132, 256, 512, 2048, 4096, 6144, 8192, 16380, 16384, 16388} {
		p := NewMDCTPlan(n)
		x, c, y := make([]float64, n), make([]float64, n/2), make([]float64, n)
		x32, c32, y32 := make([]float32, n), make([]float32, n/2), make([]float32, n)
		for i := range x {
			x[i] = float64(i%37 - 18)
			x32[i] = float32(x[i])
		}
		for _, path := range []struct {
			name   string
			kernel bool
		}{{"loop", false}, {"kernel", true}} {
			name := path.name + "/" + strconv.Itoa(n)
			b.Run("MDCT/"+name, func(b *testing.B) {
				forcePath(path.kernel)
				for i := 0; i < b.N; i++ {
					p.MDCT(c, x, 1)
				}
			})
			b.Run("IMDCT/"+name, func(b *testing.B) {
				forcePath(path.kernel)
				for i := 0; i < b.N; i++ {
					p.IMDCT(y, c, 1)
				}
			})
			b.Run("MDCTMixed/"+name, func(b *testing.B) {
				forcePath(path.kernel)
				for i := 0; i < b.N; i++ {
					p.MDCTMixed(c32, x32, 1)
				}
			})
			b.Run("IMDCTMixed/"+name, func(b *testing.B) {
				forcePath(path.kernel)
				for i := 0; i < b.N; i++ {
					p.IMDCTMixed(y32, c32, 1)
				}
			})
		}
	}
}
