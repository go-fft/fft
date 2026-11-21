package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
	"github.com/go-simd/floats"
)

//go:generate go run internal/gendct23/main.go internal/kernels

// The fused DCT-II and DCT-III of the lengths 32, 64 and 128 (go-fft/fft#162):
// Makhoul's algorithm with the steps of the generic route (reorder, n-point real
// FFT = n/2-point complex FFT plus untangle, twiddle, scale) folded into three: the
// reorder is written straight into the half-length FFT's input, one
// floats.MirrorCombine pass untangles and twiddles at once, and the call touches
// neither the real plan's wrappers nor the generic scratch.

// dct23Len reports whether n is a length with the fused DCT-II/III.
func dct23Len(n int) bool { return n == 32 || n == 64 || n == 128 }

// dct23 holds the tables of one length, k = 0 .. n/2: c1[k] = exp(-iπk/2n) and
// c2[k] = c1[k]·exp(-2πik/n) for the forward pass; ct[k] = exp(+2πik/n) and
// cw[k] = conj(c1[k]) for the inverse, with ones and mt = -ct the weights of its
// combiner.
type dct23 struct {
	n               int
	half            *Plan
	c1, c2, ct, cw  []complex128
	ones, mt        []complex128
	c1f, c2f, onesf []float64
	cwf             []float64
	mtf             []float64
	pool            scratchCache[dct23Scratch]
}

// dct23Scratch is one call's working memory.
type dct23Scratch struct{ u, z, y [65]complex128 }

func newDCT23(n int) *dct23 {
	h := n / 2
	d := &dct23{n: n, half: NewPlan(h)}
	d.pool.init(func() *dct23Scratch { return new(dct23Scratch) })
	mk := func() []complex128 { return make([]complex128, h+1) }
	d.c1, d.c2, d.ct, d.cw, d.ones, d.mt = mk(), mk(), mk(), mk(), mk(), mk()
	for k := 0; k <= h; k++ {
		s, c := math.Sincos(-math.Pi * float64(k) / float64(2*n))
		d.c1[k] = complex(c, s)
		d.cw[k] = complex(c, -s)
		s, c = math.Sincos(-math.Pi * float64(5*k) / float64(2*n))
		d.c2[k] = complex(c, s)
		s, c = math.Sincos(2 * math.Pi * float64(k) / float64(n))
		d.ct[k] = complex(c, s)
		d.ones[k] = 1
		d.mt[k] = -d.ct[k]
	}
	d.c1f, d.c2f, d.onesf, d.mtf = asFloats(d.c1), asFloats(d.c2), asFloats(d.ones), asFloats(d.mt)
	d.cwf = asFloats(d.cw)
	return d
}

// forward writes the unnormalised DCT-II of src[:n] into dst[:n]; dst may alias
// src (src is read in full before dst is written). Same sums as makhoul.dct2:
// with V the real FFT of the reordered input, y[k] = 2·Re(w_k·V[k]) and
// y[n-k] = -2·Im(w_k·V[k]), and V[k] = (P + t_k·Q)/2 for P = Z[k] + conj Z[h-k],
// Q = -i(Z[k] - conj Z[h-k]), so 2·w_k·V[k] = c1[k]·P + c2[k]·Q, which is
// floats.MirrorCombineSplit.
func (d *dct23) forward(dst, src []float64) {
	n := d.n
	if kernels.Dct23On {
		switch n {
		case 32:
			kernels.Dct2x32((*[32]float64)(dst), (*[32]float64)(src))
			return
		case 64:
			kernels.Dct2x64((*[64]float64)(dst), (*[64]float64)(src))
			return
		}
	}
	h := n / 2
	s := d.pool.get()
	z, y := s.z[:h], s.y[:h]
	q := h / 2
	for j := 0; j < q; j++ {
		z[j] = complex(src[4*j], src[4*j+2])
		z[q+j] = complex(src[2*n-1-4*(q+j)], src[2*n-3-4*(q+j)])
	}
	d.half.execute(y, z, false)
	y0 := y[0]
	x0 := 2 * (real(y0) + imag(y0))
	xh := 2 * real(d.c1[h]) * (real(y0) - imag(y0))
	floats.MirrorCombineSplit(dst[:n], asFloats(y), d.c1f, d.c2f)
	dst[0], dst[h] = x0, xh
	d.pool.put(s)
}

// inverse writes the unnormalised DCT-III of src[:n] into dst[:n], src[0] taken
// times a0; dst may alias src. It is makhoul.dct3 with the same pass folding:
// U[k] = conj(w_k)·(X[k] - i·X[n-k]) is the Hermitian half spectrum, the packed
// spectrum is Z[k] = P + i·exp(+2πik/n)·D for P = U[k] + conj U[h-k] and
// D = U[k] - conj U[h-k] (floats.MirrorCombine with the weights 1 and -exp(+2πik/n),
// k = 1 .. h-1; k = 0 here), and the h-point inverse FFT gives v[2j] + i·v[2j+1].
func (d *dct23) inverse(dst, src []float64, a0 float64) {
	n := d.n
	if kernels.Dct23On {
		switch n {
		case 32:
			kernels.Dct3x32((*[32]float64)(dst), (*[32]float64)(src), a0)
			return
		case 64:
			kernels.Dct3x64((*[64]float64)(dst), (*[64]float64)(src), a0)
			return
		}
	}
	h := n / 2
	s := d.pool.get()
	u, z, y := s.u[:h+1], s.z[:h], s.y[:h]
	u[0] = complex(a0*src[0], 0)
	// u[k] = conj(w_k)·(src[k] - i·src[n-k]), k = 1 .. h: the real parts are src[1:h+1],
	// the imaginary ones src[h:n] read from its end.
	floats.CMulZipRev(asFloats(u[1:h+1]), src[1:h+1], src[h:n], d.cwf[2:2*(h+1)])
	a, b := u[0], u[h]
	z[0] = complex(real(a)+real(b)-(imag(a)+imag(b)), imag(a)-imag(b)+(real(a)-real(b)))
	floats.MirrorCombine(asFloats(z), asFloats(u[:h]), d.onesf, d.mtf)
	d.half.execute(y, z, true)
	q := h / 2
	for j := 0; j < q; j++ {
		dst[4*j] = real(y[j])
		dst[4*j+2] = imag(y[j])
		dst[2*n-1-4*(q+j)] = real(y[q+j])
		dst[2*n-3-4*(q+j)] = imag(y[q+j])
	}
	d.pool.put(s)
}
