package fft

import (
	"math"
	"sync"

	"github.com/go-fft/fft/internal/kernels"
)

// Bluestein chirp-z plan.
//
// For a length N with a large prime factor (one that the mixed-radix engine
// cannot reduce cheaply), the DFT is computed as a length-M linear convolution,
// M being the cheapest 7-smooth length >= 2N-1 (bestConvLen), run on the
// Stockham engine through a cached plan. pocketfft's fftblue does the same with
// its good_size; this used to pad to the next power of two and run the
// standalone radix-2 kernel, which for N=641 meant a 2048-point transform
// instead of a 1296-point one. The chirp sequence and the convolution kernel's
// spectrum (pre-scaled by 1/M, so the inverse needs no normalization pass) are
// precomputed once into the plan, so a transform costs two length-M FFTs and a
// pointwise product with no per-call trig.
type bluesteinPlan struct {
	n  int
	m  int          // convolution length, 7-smooth, >= 2n-1
	wF []complex128 // forward chirp  w[j]  = exp(-πi·j²/N), j = 0 .. n-1
	wI []complex128 // inverse chirp  conj  = exp(+πi·j²/N)
	bF []complex128 // FFT of the forward kernel b (built from conj(wF)) / m
	bI []complex128 // FFT of the inverse kernel (built from conj(wI)) / m

	// scratch lends each concurrent transform its length-m convolution buffer,
	// so a steady-state transform allocates nothing.
	scratch sync.Pool
}

// newBluesteinPlan precomputes the chirps and the kernel spectra for both
// directions of a length-n Bluestein transform.
func newBluesteinPlan(n int) *bluesteinPlan {
	m := bestConvLen(2*n - 1)
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
	p.scratch.New = func() any { b := make([]complex128, m); return &b }
	return p
}

// buildKernelSpectrum forms the mirrored conj(chirp) kernel b of length m and
// returns its forward FFT scaled by 1/m, precomputed so transforms skip both.
func buildKernelSpectrum(w []complex128, n, m int) []complex128 {
	b := make([]complex128, m)
	b[0] = complexConj(w[0])
	for j := 1; j < n; j++ {
		c := complexConj(w[j])
		b[j] = c
		b[m-j] = c
	}
	cachedPlan(m).FFT(b, b)
	inv := complex(1/float64(m), 0)
	for i := range b {
		b[i] *= inv
	}
	return b
}

// transform writes the unnormalized length-n DFT of src into dst via the
// precomputed chirp-z convolution. dst may alias src.
func (p *bluesteinPlan) transform(dst, src []complex128, inverse bool) {
	n := p.n
	w, bSpec := p.wF, p.bF
	if inverse {
		w, bSpec = p.wI, p.bI
	}
	bp := p.scratch.Get().(*[]complex128)
	defer p.scratch.Put(bp)
	a := *bp
	for j := 0; j < n; j++ {
		a[j] = src[j] * w[j]
	}
	clear(a[n:]) // the pooled buffer is dirty; the pad must be zero

	// Convolution, in place: a = IFFT(FFT(a) · bSpec); bSpec carries the 1/m.
	plan := cachedPlan(p.m)
	plan.execute(a, a, false)
	kernels.CMul(a, bSpec)
	plan.execute(a, a, true)

	// dst[k] = w[k]·conv[k].
	for k := 0; k < n; k++ {
		dst[k] = a[k] * w[k]
	}
}
