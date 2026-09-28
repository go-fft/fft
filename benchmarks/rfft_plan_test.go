package benchmarks

import (
	"math"
	"strconv"
	"testing"

	gofft "github.com/go-fft/fft"
)

// BenchmarkRealPlanRFFT measures the reusable, allocation-free spectrogram path.
func BenchmarkRealPlanRFFT(b *testing.B) {
	for _, n := range []int{64, 256, 512, 1024, 2048, 4096, 16384} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			p := gofft.NewRealPlan(n)
			x := make([]float64, n)
			for i := range x {
				x[i] = math.Sin(float64(i)*0.7) + float64(i%4)
			}
			dst := make([]complex128, n/2+1)
			p.RFFT(dst, x)
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				p.RFFT(dst, x)
			}
		})
	}
}
