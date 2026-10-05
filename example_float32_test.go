package fft_test

import (
	"fmt"
	"math"

	"github.com/go-fft/fft"
)

// tidy turns the zeros of v positive for printing: whether an exactly
// cancelling sum comes out +0 or -0 depends on the CPU's fused multiply-add
// (loong64 and amd64 differ), and the sign of zero carries no information here.
func tidy(v []complex64) []complex64 {
	out := make([]complex64, len(v))
	for i, z := range v {
		out[i] = complex(real(z)+0, imag(z)+0)
	}
	return out
}

// FFT32 and IFFT32 are FFT and IFFT for complex64 data.
func ExampleFFT32() {
	x := []complex64{1, 2, 3, 4}
	X := fft.FFT32(x)
	fmt.Println(tidy(X))
	fmt.Println(tidy(fft.IFFT32(X)))
	// Output:
	// [(10+0i) (-2+2i) (-2+0i) (-2-2i)]
	// [(1+0i) (2+0i) (3+0i) (4+0i)]
}

// RFFT32 keeps the N/2+1 non-redundant bins of a float32 signal; IRFFT32
// needs the length back, since N and N+1 give as many bins.
func ExampleRFFT32() {
	x := []float32{1, 2, 3, 4}
	X := fft.RFFT32(x)
	fmt.Println(tidy(X))
	fmt.Println(fft.IRFFT32(X, len(x)))
	// Output:
	// [(10+0i) (-2+2i) (-2+0i)]
	// [1 2 3 4]
}

// A Plan32 is reused across transforms of one length, here with the unitary
// scaling (numpy's norm="ortho"): both directions divide by sqrt(N).
func ExamplePlan32_FFTNorm() {
	p := fft.NewPlan32(4)
	x := []complex64{1, 1, 1, 1}
	X := p.FFTNorm(make([]complex64, 4), x, fft.NormOrtho)
	fmt.Println(tidy(X))
	fmt.Println(tidy(p.IFFTNorm(make([]complex64, 4), X, fft.NormOrtho)))
	// Output:
	// [(2+0i) (0+0i) (0+0i) (0+0i)]
	// [(1+0i) (1+0i) (1+0i) (1+0i)]
}

// A RealPlan32 serves repeated real transforms of one length.
func ExampleRealPlan32() {
	p := fft.NewRealPlan32(8)
	x := []float32{0, 1, 0, -1, 0, 1, 0, -1} // two cycles of a sine
	X := p.RFFT(make([]complex64, 5), x)
	for k, v := range X {
		fmt.Printf("bin %d: magnitude %.3f\n", k, math.Hypot(float64(real(v)), float64(imag(v))))
	}
	// Output:
	// bin 0: magnitude 0.000
	// bin 1: magnitude 0.000
	// bin 2: magnitude 4.000
	// bin 3: magnitude 0.000
	// bin 4: magnitude 0.000
}
