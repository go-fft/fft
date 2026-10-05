package fft_test

import (
	"fmt"
	"math"

	"github.com/go-fft/fft"
)

// round4 rounds to 4 decimals and turns the rounding residue of an exact zero
// (which may print as -0.0000 on one architecture and 0.0000 on another) into 0.
func round4(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Round(v*1e4) / 1e4
		if out[i] == 0 {
			out[i] = 0
		}
	}
	return out
}

// The DCT-II with norm="ortho" is MATLAB's dct and the transform JPEG uses; a
// smooth signal packs its energy into the first coefficients.
func ExampleDCT() {
	x := []float64{1, 2, 3, 4}
	y := fft.DCT(x, 2, fft.NormOrtho)
	fmt.Println(round4(y))
	fmt.Println(round4(fft.IDCT(y, 2, fft.NormOrtho)))
	// Output:
	// [5 -2.2304 0 -0.1585]
	// [1 2 3 4]
}

// A plan computes repeated transforms of one length without allocating;
// dst may be src itself.
func ExampleNewDCTPlan() {
	p := fft.NewDCTPlan(4, 4)
	buf := []float64{1, 0, 0, 0}
	p.DCT(buf, buf, fft.NormBackward)
	fmt.Println(round4(buf))
	p.IDCT(buf, buf, fft.NormBackward)
	fmt.Println(round4(buf))
	// Output:
	// [1.9616 1.6629 1.1111 0.3902]
	// [1 0 0 0]
}

// The DST-I diagonalises the second-difference operator with zero boundary
// values: a sine mode is one coefficient.
func ExampleDST() {
	x := []float64{0.5, 0.8660254037844386, 1, 0.8660254037844386, 0.5} // sin(π(n+1)/6)
	fmt.Println(round4(fft.DST(x, 1, fft.NormBackward)))
	// Output:
	// [6 0 0 0 0]
}

// DCTN transforms every axis of a row-major array.
func ExampleDCTN() {
	img := []float64{
		1, 1, 1,
		1, 1, 1,
	}
	fmt.Println(round4(fft.DCTN(img, []int{2, 3}, 2, fft.NormOrtho)))
	// Output:
	// [2.4495 0 0 0 0 0]
}
