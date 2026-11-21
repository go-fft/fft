//go:build !amd64

package fft

// The generated DCT-II/III kernels are amd64's; here the table route always runs.

func dct23Run2(dst, src []float64, n int) bool { return false }

func dct23Run3(dst, src []float64, n int, a0 float64) bool { return false }
