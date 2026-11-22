package fft

import (
	"strconv"
	"testing"
)

// BenchmarkEvex256 times a whole Plan.FFT at the composite lengths whose
// radix-12/20 passes spill on 16 ymm registers (go-fft/fft#162), and at 1000,
// the control (no framed kernel).
func BenchmarkEvex256(b *testing.B) {
	isolatePlans(b)
	for _, n := range []int{1000, 1080, 1440, 1920, 2160} {
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
