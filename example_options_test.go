package fft_test

import (
	"fmt"
	"math"

	"github.com/go-fft/fft"
)

// round keeps four decimals and turns -0 into 0, so the examples print stable
// output across architectures.
func round(v float64) float64 {
	r := math.Round(v*1e4) / 1e4
	if r == 0 {
		return 0
	}
	return r
}

func roundC(x []complex128) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex(round(real(v)), round(imag(v)))
	}
	return out
}

func roundR(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = round(v)
	}
	return out
}

func ExampleOptions() {
	x := []complex128{1, 2, 3, 4}
	// The zero Options is numpy's defaults: the same as FFT.
	fmt.Println(roundC(fft.FFTWith(x, fft.Options{})))
	// Name only what changes: pad to 8 points, unitary scaling.
	fmt.Println(len(fft.FFTWith(x, fft.Options{N: 8, Norm: fft.NormOrtho})))
	// Output:
	// [(10+0i) (-2+2i) (-2+0i) (-2-2i)]
	// 8
}

func ExampleFFTWith() {
	x := []complex128{1, 1, 1, 1, 1, 1}
	// Truncate to 4 points and scale by 1/sqrt(4) (numpy norm="ortho").
	fmt.Println(roundC(fft.FFTWith(x, fft.Options{N: 4, Norm: fft.NormOrtho})))
	// Output:
	// [(2+0i) (0+0i) (0+0i) (0+0i)]
}

func ExampleIFFTWith() {
	X := []complex128{4, 0, 0, 0}
	// Under NormForward the inverse is unscaled.
	fmt.Println(roundC(fft.IFFTWith(X, fft.Options{Norm: fft.NormForward})))
	// Output:
	// [(4+0i) (4+0i) (4+0i) (4+0i)]
}

func ExampleRFFTWith() {
	r := []float64{1, 2, 3}
	// Zero-pad to 4 points: 4/2+1 = 3 bins.
	fmt.Println(roundC(fft.RFFTWith(r, fft.Options{N: 4})))
	// Output:
	// [(6+0i) (-2-2i) (2+0i)]
}

func ExampleIRFFTWith() {
	spectrum := []complex128{6, -2 - 2i, 2}
	// N = 0 takes numpy's default length 2*(3-1) = 4.
	fmt.Println(roundR(fft.IRFFTWith(spectrum, fft.Options{})))
	// Output:
	// [1 2 3 0]
}

func ExampleHFFT() {
	// The first half of a Hermitian-symmetric signal y = [1, 2, 3, 2]; its DFT
	// is real.
	fmt.Println(roundR(fft.HFFT([]complex128{1, 2, 3}, 4)))
	// Output:
	// [8 -2 0 -2]
}

func ExampleHFFTWith() {
	// The same transform scaled as a forward transform under NormForward: 1/n.
	fmt.Println(roundR(fft.HFFTWith([]complex128{1, 2, 3}, fft.Options{N: 4, Norm: fft.NormForward})))
	// Output:
	// [2 -0.5 0 -0.5]
}

func ExampleIHFFT() {
	// IHFFT inverts HFFT: back to the half signal.
	fmt.Println(roundC(fft.IHFFT([]float64{8, -2, 0, -2})))
	// Output:
	// [(1+0i) (2+0i) (3+0i)]
}

func ExampleIHFFTWith() {
	x := []float64{8, -2, 0, -2}
	h := fft.IHFFTWith(x, fft.Options{Norm: fft.NormOrtho})
	// The pair round-trips under any Norm, given the same one both ways.
	fmt.Println(roundR(fft.HFFTWith(h, fft.Options{N: len(x), Norm: fft.NormOrtho})))
	// Output:
	// [8 -2 0 -2]
}

func ExampleFFTNWith() {
	// Three signals of four points, one per row: transforming the last axis
	// alone is numpy's fft(x, axis=-1), the batched 1-D transform.
	rows := []complex128{
		1, 0, 0, 0,
		1, 1, 1, 1,
		0, 1, 0, 0,
	}
	out := fft.FFTNWith(rows, []int{3, 4}, fft.Options{Axes: []int{-1}})
	for r := 0; r < 3; r++ {
		fmt.Println(roundC(out[r*4 : (r+1)*4]))
	}
	// Output:
	// [(1+0i) (1+0i) (1+0i) (1+0i)]
	// [(4+0i) (0+0i) (0+0i) (0+0i)]
	// [(1+0i) (0-1i) (-1+0i) (0+1i)]
}

func ExampleIFFTNWith() {
	X := []complex128{4, 0, 0, 0}
	// Inverse along axis 0 only, of a 2×2 array.
	fmt.Println(roundC(fft.IFFTNWith(X, []int{2, 2}, fft.Options{Axes: []int{0}})))
	// Output:
	// [(2+0i) (0+0i) (2+0i) (0+0i)]
}

func ExampleFFT2With() {
	x := []complex128{1, 1, 1, 1}
	fmt.Println(roundC(fft.FFT2With(x, [2]int{2, 2}, fft.Options{Norm: fft.NormOrtho})))
	// Output:
	// [(2+0i) (0+0i) (0+0i) (0+0i)]
}

func ExampleIFFT2With() {
	X := []complex128{2, 0, 0, 0}
	fmt.Println(roundC(fft.IFFT2With(X, [2]int{2, 2}, fft.Options{Norm: fft.NormOrtho})))
	// Output:
	// [(1+0i) (1+0i) (1+0i) (1+0i)]
}

func ExampleRFFTN() {
	// A 2×2×4 real array: the last axis keeps 4/2+1 = 3 bins.
	x := make([]float64, 16)
	for i := range x {
		x[i] = 1
	}
	s := fft.RFFTN(x, []int{2, 2, 4})
	fmt.Println(len(s), roundC(s[:3]))
	// Output:
	// 12 [(16+0i) (0+0i) (0+0i)]
}

func ExampleRFFTNWith() {
	x := []float64{
		1, 2,
		3, 4,
		5, 6,
	}
	// The real transform along axis 0 (the last axis listed): 3 rows keep
	// 3/2+1 = 2 bins, and the output is 2×2.
	fmt.Println(roundC(fft.RFFTNWith(x, []int{3, 2}, fft.Options{Axes: []int{0}})))
	// Output:
	// [(9+0i) (12+0i) (-3+1.7321i) (-3+1.7321i)]
}

func ExampleIRFFTN() {
	x := []float64{1, 2, 3, 4, 5, 6}
	// The shape is the real output's: its last length cannot be recovered
	// from the bins alone.
	back := fft.IRFFTN(fft.RFFTN(x, []int{2, 3}), []int{2, 3})
	fmt.Println(roundR(back))
	// Output:
	// [1 2 3 4 5 6]
}

func ExampleIRFFTNWith() {
	x := []float64{1, 2, 3, 4, 5, 6}
	o := fft.Options{Axes: []int{0}, Norm: fft.NormOrtho}
	fmt.Println(roundR(fft.IRFFTNWith(fft.RFFTNWith(x, []int{3, 2}, o), []int{3, 2}, o)))
	// Output:
	// [1 2 3 4 5 6]
}

func ExampleRFFT2With() {
	img := []float64{1, 1, 1, 1}
	fmt.Println(roundC(fft.RFFT2With(img, [2]int{2, 2}, fft.Options{Norm: fft.NormForward})))
	// Output:
	// [(1+0i) (0+0i) (0+0i) (0+0i)]
}

func ExampleIRFFT2With() {
	spec := []complex128{1, 0, 0, 0}
	fmt.Println(roundR(fft.IRFFT2With(spec, [2]int{2, 2}, fft.Options{Norm: fft.NormForward})))
	// Output:
	// [1 1 1 1]
}

func ExampleFFTShift() {
	// Bin frequencies in increasing order, the zero frequency in the middle.
	fmt.Println(fft.FFTShift(fft.FFTFreq(5, 1)))
	fmt.Println(fft.FFTShift([]int{0, 1, 2, 3, 4, 5}))
	// Output:
	// [-0.4 -0.2 0 0.2 0.4]
	// [3 4 5 0 1 2]
}

func ExampleIFFTShift() {
	shifted := fft.FFTShift([]int{0, 1, 2, 3, 4})
	fmt.Println(shifted, fft.IFFTShift(shifted))
	// Output:
	// [3 4 0 1 2] [0 1 2 3 4]
}

func ExampleFFTShiftN() {
	x := []int{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
	}
	fmt.Println(fft.FFTShiftN(x, []int{3, 3}, nil))
	fmt.Println(fft.FFTShiftN(x, []int{3, 3}, []int{-1})) // the rows only
	// Output:
	// [8 6 7 2 0 1 5 3 4]
	// [2 0 1 5 3 4 8 6 7]
}

func ExampleIFFTShiftN() {
	x := []int{0, 1, 2, 3, 4, 5}
	s := fft.FFTShiftN(x, []int{2, 3}, nil)
	fmt.Println(s, fft.IFFTShiftN(s, []int{2, 3}, nil))
	// Output:
	// [5 3 4 2 0 1] [0 1 2 3 4 5]
}

func ExampleNextFastLen() {
	fmt.Println(fft.NextFastLen(1009, false)) // 1009 is prime; the next 7-smooth length is 1024
	fmt.Println(fft.NextFastLen(1001, true))  // 1001 = 7·11·13 is not; 1008 = 2⁴·3²·7 is even and 7-smooth
	// Output:
	// 1024
	// 1008
}

func ExamplePlan_FFTNorm() {
	p := fft.NewPlan(4)
	dst := make([]complex128, 4)
	fmt.Println(roundC(p.FFTNorm(dst, []complex128{1, 1, 1, 1}, fft.NormForward)))
	// Output:
	// [(1+0i) (0+0i) (0+0i) (0+0i)]
}

func ExamplePlan_IFFTNorm() {
	p := fft.NewPlan(4)
	dst := make([]complex128, 4)
	fmt.Println(roundC(p.IFFTNorm(dst, []complex128{1, 0, 0, 0}, fft.NormForward)))
	// Output:
	// [(1+0i) (1+0i) (1+0i) (1+0i)]
}

func ExampleRealPlan_RFFTNorm() {
	p := fft.NewRealPlan(4)
	dst := make([]complex128, 3)
	fmt.Println(roundC(p.RFFTNorm(dst, []float64{1, 1, 1, 1}, fft.NormOrtho)))
	// Output:
	// [(2+0i) (0+0i) (0+0i)]
}

func ExampleRealPlan_IRFFTNorm() {
	p := fft.NewRealPlan(4)
	dst := make([]float64, 4)
	fmt.Println(roundR(p.IRFFTNorm(dst, []complex128{2, 0, 0}, fft.NormOrtho)))
	// Output:
	// [1 1 1 1]
}

func ExamplePlanN_FFTNorm() {
	p := fft.NewPlanN(2, 2)
	dst := make([]complex128, 4)
	fmt.Println(roundC(p.FFTNorm(dst, []complex128{1, 1, 1, 1}, fft.NormOrtho)))
	// Output:
	// [(2+0i) (0+0i) (0+0i) (0+0i)]
}

func ExamplePlanN_IFFTNorm() {
	p := fft.NewPlanN(2, 2)
	dst := make([]complex128, 4)
	fmt.Println(roundC(p.IFFTNorm(dst, []complex128{2, 0, 0, 0}, fft.NormOrtho)))
	// Output:
	// [(1+0i) (1+0i) (1+0i) (1+0i)]
}

func ExampleRealPlan2_RFFTNorm() {
	p := fft.NewRealPlan2(2, 2)
	dst := make([]complex128, p.SpectrumLen())
	fmt.Println(roundC(p.RFFTNorm(dst, []float64{1, 1, 1, 1}, fft.NormForward)))
	// Output:
	// [(1+0i) (0+0i) (0+0i) (0+0i)]
}

func ExampleRealPlan2_IRFFTNorm() {
	p := fft.NewRealPlan2(2, 2)
	dst := make([]float64, 4)
	fmt.Println(roundR(p.IRFFTNorm(dst, []complex128{1, 0, 0, 0}, fft.NormForward)))
	// Output:
	// [1 1 1 1]
}
