// Command verify dumps go-fft transform outputs as JSON for the
// numerical-correctness check in ../verify_correctness.py, which compares them
// against numpy.fft within tolerance. A fast wrong answer does not count, so
// this runs before the timing harness in run.sh.
//
// Inputs are bit-identical to the h2hComplex/h2hReal generators used by the
// benchmark matrix, so the numbers verified are the numbers timed.
package main

import (
	"encoding/json"
	"os"

	fft "github.com/go-fft/fft"
)

func cmplx(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
	}
	return x
}

func realv(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*7+1)%13) * 0.1
	}
	return x
}

// halfSpectrum is the c2r inverse input, bit-identical to the FFTW c2r harness
// and verify_correctness.py's halfspec generator: an N/2+1-bin complex spectrum.
func halfSpectrum(n int) []complex128 {
	x := make([]complex128, n/2+1)
	for i := range x {
		x[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
	}
	return x
}

type pair struct{ Re, Im float64 }

func toPairs(z []complex128) []pair {
	p := make([]pair, len(z))
	for i, v := range z {
		p[i] = pair{real(v), imag(v)}
	}
	return p
}

func main() {
	sizes := []int{256, 1024, 4096, 1000, 1080, 1920, 1009, 1296, 10007}
	out := map[string]any{}

	cplx := map[string][]pair{}
	rfft := map[string][]pair{}
	irfft := map[string][]float64{}
	for _, n := range sizes {
		cplx[itoa(n)] = toPairs(fft.FFT(cmplx(n)))
	}
	for _, n := range []int{256, 1024, 4096, 1000, 1080, 1920} {
		rfft[itoa(n)] = toPairs(fft.RFFT(realv(n)))
		irfft[itoa(n)] = fft.IRFFT(halfSpectrum(n), n)
	}
	// 2-D
	fft2 := map[string][]pair{}
	for _, s := range [][2]int{{64, 64}, {128, 128}, {256, 256}} {
		fft2[itoa(s[0])+"x"+itoa(s[1])] = toPairs(fft.FFT2(cmplx(s[0]*s[1]), s))
	}
	out["complex"] = cplx
	out["real"] = rfft
	out["ireal"] = irfft
	out["fft2"] = fft2

	enc := json.NewEncoder(os.Stdout)
	_ = enc.Encode(out)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
