package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
)

// Bluestein chirp-z plan.
//
// For a length N with a large prime factor (one that the mixed-radix engine
// cannot reduce cheaply), the DFT is computed as a length-M power-of-two
// convolution, M being the smallest power of two >= 2N-1. The chirp sequence,
// the convolution kernel's pre-transformed spectrum, and the output chirp are
// all precomputed once into the plan, so a transform costs two radix-2 FFTs and
// a pointwise product with no per-call trig.
type bluesteinPlan struct {
	n  int
	m  int          // convolution length, a power of two
	wF []complex128 // forward chirp  w[j]  = exp(-πi·j²/N), j = 0 .. n-1
	wI []complex128 // inverse chirp  conj  = exp(+πi·j²/N)
	bF []complex128 // FFT of the forward kernel b (built from conj(wF)), length m
	bI []complex128 // FFT of the inverse kernel (built from conj(wI)), length m
}

// newBluesteinPlan precomputes the chirps and the kernel spectra for both
// directions of a length-n Bluestein transform.
func newBluesteinPlan(n int) *bluesteinPlan {
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	p := &bluesteinPlan{n: n, m: m}

	p.wF = make([]complex128, n)
	p.wI = make([]complex128, n)
	for j := 0; j < n; j++ {
		jj := (j * j) % (2 * n) // keep the angle accurate for large N
		ang := math.Pi * float64(jj) / float64(n)
		// Forward chirp uses sign -1, inverse uses +1.
		p.wF[j] = complex(math.Cos(-ang), math.Sin(-ang))
		p.wI[j] = complex(math.Cos(ang), math.Sin(ang))
	}

	p.bF = buildKernelSpectrum(p.wF, n, m)
	p.bI = buildKernelSpectrum(p.wI, n, m)
	return p
}

// buildKernelSpectrum forms the mirrored conj(chirp) kernel b of length m and
// returns its forward radix-2 FFT, precomputed so transforms skip it.
func buildKernelSpectrum(w []complex128, n, m int) []complex128 {
	b := make([]complex128, m)
	b[0] = complexConj(w[0])
	for j := 1; j < n; j++ {
		c := complexConj(w[j])
		b[j] = c
		b[m-j] = c
	}
	kernels.BitReverse(b)
	kernels.Radix2(b, false)
	return b
}

// transform writes the unnormalized length-n DFT of src into dst via the
// precomputed chirp-z convolution. dst may alias src.
func (p *bluesteinPlan) transform(dst, src []complex128, inverse bool) {
	n, m := p.n, p.m
	w, bSpec := p.wF, p.bF
	if inverse {
		w, bSpec = p.wI, p.bI
	}

	// a[j] = src[j]·w[j], zero-padded to m.
	a := make([]complex128, m)
	for j := 0; j < n; j++ {
		a[j] = src[j] * w[j]
	}

	// Convolution: a = IFFT(FFT(a) · bSpec).
	kernels.BitReverse(a)
	kernels.Radix2(a, false)
	kernels.CMul(a, bSpec)
	kernels.BitReverse(a)
	kernels.Radix2(a, true)
	invM := complex(1/float64(m), 0)
	for i := 0; i < m; i++ {
		a[i] *= invM
	}

	// dst[k] = w[k]·conv[k].
	for k := 0; k < n; k++ {
		dst[k] = a[k] * w[k]
	}
}
