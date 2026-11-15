package fft

import "github.com/go-simd/floats"

// The MDCT of N = 128 and 256 samples (the AAC short block and its half) is a fold
// into a DCT-IV of N/2 = 64 or 128 points, which has a fused kernel (dct4_64.go):
// the fold and the unfold are the go-simd/floats kernels, the DCT-IV the fused
// scalar kernel, all on a stack array, in place (the kernels read every input
// before the first store). The arithmetic is the plan's float64 whatever the data
// type is: a float32 result is float32 of the float64 one, as for the other paths.

// mdctForwardFused is mdctForward for a fused plan.
func mdctForwardFused[S, D floatT](p *MDCTPlan, dst []D, src []S, scale float64) []D {
	n := p.n
	m := n / 2
	var buf [128]float64
	u := buf[:m]
	switch s := any(src[:n]).(type) {
	case []float64:
		floats.MDCTFold(u, s)
	case []float32:
		floats.MDCTFoldFrom32(u, s)
	}
	dct4Single(m, u, u, scale/2) // the package's DCT-IV is twice the sum
	dst = dst[:m]
	switch d := any(dst).(type) {
	case []float64:
		copy(d, u)
	case []float32:
		for i, v := range u {
			d[i] = float32(v)
		}
	}
	return dst
}

// mdctInverseFused is mdctInverse for a fused plan.
func mdctInverseFused[S, D floatT](p *MDCTPlan, dst []D, src []S, scale float64) []D {
	n := p.n
	m := n / 2
	var buf [128]float64
	u := buf[:m]
	for i := range u {
		u[i] = float64(src[i])
	}
	dct4Single(m, u, u, scale/2)
	dst = dst[:n]
	switch d := any(dst).(type) {
	case []float64:
		floats.MDCTUnfold(d, u)
	case []float32:
		floats.MDCTUnfoldTo32(d, u)
	}
	return dst
}
