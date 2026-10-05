package fft_test

import (
	"fmt"
	"math"

	"github.com/go-fft/fft"
)

// round4f32 rounds v to four decimals for printing, with positive zeros: a value
// that cancels exactly, or rounds to zero, may carry either sign depending on
// the CPU's fused multiply-add, and the sign carries no information here.
func round4f32(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Round(float64(x)*1e4)/1e4 + 0
	}
	return out
}

// FFTN32 and IFFTN32 are FFTN and IFFTN for complex64 data; FFT2_32 is the
// two-axis case (the underscore keeps "2" and "32" apart).
func ExampleFFTN32() {
	x := []complex64{1, 2, 3, 4} // a 2×2 matrix, row-major
	X := fft.FFTN32(x, []int{2, 2})
	fmt.Println(tidy(X))
	fmt.Println(tidy(fft.FFT2_32(x, [2]int{2, 2})))
	fmt.Println(tidy(fft.IFFTN32(X, []int{2, 2})))
	// Output:
	// [(10+0i) (-2+0i) (-4+0i) (0+0i)]
	// [(10+0i) (-2+0i) (-4+0i) (0+0i)]
	// [(1+0i) (2+0i) (3+0i) (4+0i)]
}

// RFFT2_32 keeps cols/2+1 bins per row of a float32 image; IRFFT2_32 takes
// the image's shape back. (A package-level example: go vet reads a name like
// ExampleRFFT2_32 as an example of a method "32" of RFFT2.)
func Example_rfft2_32() {
	img := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}
	S := fft.RFFT2_32(img, [2]int{2, 4}) // 2×3 bins
	fmt.Println(tidy(S))
	fmt.Println(fft.IRFFT2_32(S, [2]int{2, 4}))
	// Output:
	// [(36+0i) (-4+4i) (-4+0i) (-16+0i) (0+0i) (0+0i)]
	// [1 2 3 4 5 6 7 8]
}

// The Options of the float64 transforms apply unchanged: here numpy's n
// (zero-padding to 4 points) and norm="forward".
func ExampleFFT32With() {
	x := []complex64{4, 4}
	X := fft.FFT32With(x, fft.Options{N: 4, Norm: fft.NormForward})
	fmt.Println(tidy(X))
	// Output:
	// [(2+0i) (1-1i) (0+0i) (1+1i)]
}

// Batched real transforms: every row of a matrix of float32 signals, one
// plan, with Options.Axes as numpy's axis=-1.
func ExampleRFFTN32With() {
	rows := []float32{
		1, 1, 1, 1,
		1, 0, -1, 0,
	}
	S := fft.RFFTN32With(rows, []int{2, 4}, fft.Options{Axes: []int{-1}})
	fmt.Println(tidy(S))
	// Output:
	// [(4+0i) (0+0i) (0+0i) (0+0i) (2+0i) (0+0i)]
}

// DCT32 is scipy.fft.dct on float32 input; with NormOrtho, IDCT32 undoes it.
func ExampleDCT32() {
	x := []float32{1, 2, 3, 4}
	y := fft.DCT32(x, 2, fft.NormOrtho)
	fmt.Println(round4f32(y))
	fmt.Println(round4f32(fft.IDCT32(y, 2, fft.NormOrtho)))
	// Output:
	// [5 -2.2304 0 -0.1585]
	// [1 2 3 4]
}

// A PlanN32 serves repeated transforms of one shape, writing into the
// caller's slice.
func ExamplePlanN32() {
	p := fft.NewPlanN32(2, 3)
	x := []complex64{1, 1, 1, 1, 1, 1}
	X := p.FFTNorm(make([]complex64, p.Len()), x, fft.NormOrtho)
	fmt.Println(round4f32([]float32{real(X[0])}), tidy(X[1:]))
	// Output:
	// [2.4495] [(0+0i) (0+0i) (0+0i) (0+0i) (0+0i)]
}

// A RealPlan2_32 is the plan behind RFFT2_32 and IRFFT2_32.
func Example_realPlan2_32() {
	p := fft.NewRealPlan2_32(2, 2)
	S := p.RFFT(make([]complex64, p.SpectrumLen()), []float32{1, 2, 3, 4})
	fmt.Println(tidy(S))
	fmt.Println(p.IRFFT(make([]float32, 4), S))
	// Output:
	// [(10+0i) (-2+0i) (-4+0i) (0+0i)]
	// [1 2 3 4]
}
