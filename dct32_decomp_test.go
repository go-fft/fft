package fft

import (
	"testing"

	"github.com/go-simd/floats"
)

// BenchmarkDCT32Parts times each piece of the fused 32-point DCT-II/III route on
// its own (go-fft/fft#162): the whole call, the 16-point complex FFT through
// Plan.execute, each floats pass, the two scalar loops and the scratch round trip.
func BenchmarkDCT32Parts(b *testing.B) {
	const n = 32
	const h = 16
	p := NewDCTPlan(n, 2)
	d := p.fast
	x := benchReal(n)
	dst := make([]float64, n)
	z := make([]complex128, h)
	y := make([]complex128, h)
	u := make([]complex128, h+1)
	for i := range z {
		z[i] = complex(x[i], x[n-1-i])
	}
	b.Run("whole/dct2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.DCT(dst, x, NormBackward)
		}
	})
	b.Run("whole/dct3", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			p.IDCT(dst, x, NormBackward)
		}
	})
	b.Run("route/dct2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			d.forward(dst, x)
		}
	})
	b.Run("route/dct3", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			d.inverse(dst, x, 1)
		}
	})
	b.Run("fft16/fwd", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			d.half.execute(y, z, false)
		}
	})
	b.Run("fft16/inv", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			d.half.execute(y, z, true)
		}
	})
	b.Run("mirrorcombinesplit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			floats.MirrorCombineSplit(dst[:n], asFloats(y), d.c1f, d.c2f)
		}
	})
	b.Run("mirrorcombine", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			floats.MirrorCombine(asFloats(z), asFloats(u[:h]), d.onesf, d.mtf)
		}
	})
	b.Run("cmulziprev", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			floats.CMulZipRev(asFloats(u[1:h+1]), x[1:h+1], x[h:n], d.cwf[2:2*(h+1)])
		}
	})
	b.Run("reorder/fwd", func(b *testing.B) {
		q := h / 2
		for i := 0; i < b.N; i++ {
			for j := 0; j < q; j++ {
				z[j] = complex(x[4*j], x[4*j+2])
				z[q+j] = complex(x[2*n-1-4*(q+j)], x[2*n-3-4*(q+j)])
			}
		}
	})
	b.Run("scatter/inv", func(b *testing.B) {
		q := h / 2
		for i := 0; i < b.N; i++ {
			for j := 0; j < q; j++ {
				dst[4*j] = real(y[j])
				dst[4*j+2] = imag(y[j])
				dst[2*n-1-4*(q+j)] = real(y[q+j])
				dst[2*n-3-4*(q+j)] = imag(y[q+j])
			}
		}
	})
	b.Run("pool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			s := d.pool.get()
			d.pool.put(s)
		}
	})
}
