package fft

import "unsafe"

//go:generate go run internal/gendct4/main.go

// The fused DCT-IV of length 64 (go-fft/fft#132, go-avkit/decode#77: HE-AAC's QMF
// runs it six times per slot). A DCT-IV plan of length 64 on a platform that
// routes it (dct4Fused64Route) skips the pooled scratch, the 32-point Plan and
// the separate twiddle passes: dct4n64* in dct4_64_gen.go is one straight-line
// function whose 64 values live in registers and on the stack, with every
// twiddle a constant. Same algorithm as dct4Even (a pre-twiddle, a 32-point
// complex FFT, a post-twiddle), so the error is of the same order: the tests
// assert the bound against a double-double oracle.
//
// The pair forms run two independent transforms in one function, the two
// instruction streams interleaved.

// run64 is run for the fused plan: DCT-IV, n = 64, so the logical size is 128.
func (p *DCTPlan) run64(dst, src []float64, norm Norm, inverse bool) []float64 {
	if len(dst) < 64 || len(src) < 64 {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
	dct4Single64(dst, src, norm.scale(128, inverse))
	return dst[:64]
}

// dct4Single64 runs the scalar kernel, scaled by f unless f is 1.
func dct4Single64(dst, src []float64, f float64) {
	if f != 1 {
		dct4n64f64s(dst, src, f)
	} else {
		dct4n64f64(dst, src)
	}
}

// dct4Single64Mixed is dct4Single64 on float32 data.
func dct4Single64Mixed(dst, src []float32, f float64) {
	if f != 1 {
		dct4n64f32s(dst, src, f)
	} else {
		dct4n64f32(dst, src)
	}
}

// run64Mixed is runMixed for the fused plan.
func (p *DCTPlan) run64Mixed(dst, src []float32, norm Norm, inverse bool) []float32 {
	if len(dst) < 64 || len(src) < 64 {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
	dct4Single64Mixed(dst, src, norm.scale(128, inverse))
	return dst[:64]
}

// DCTPair writes the DCT of src0 into dst0 and that of src1 into dst1, like
// p.DCT(dst0, src0, norm) followed by p.DCT(dst1, src1, norm), and returns
// dst0[:Len()] and dst1[:Len()]. On a DCT-IV plan of length 64 on arm64 it runs
// both transforms in one NEON kernel, one in each lane of every vector register,
// a third faster than two calls (Apple M4 Max, go-fft/fft#132); elsewhere it is
// the two calls. The results are within the same error bound as DCT's, not
// necessarily bit for bit (the NEON kernel fuses multiply-adds differently).
// The slice rules are DCT's for each stream and are all checked before anything
// is written, so a panic leaves dst0 and dst1 untouched. dst0 and dst1 must not
// overlap each other or the other stream's source (if they do the two calls run
// in order, as written).
func (p *DCTPlan) DCTPair(dst0, dst1, src0, src1 []float64, norm Norm) (r0, r1 []float64) {
	return p.runPair(dst0, dst1, src0, src1, norm, false)
}

// IDCTPair is DCTPair for the inverse transform.
func (p *DCTPlan) IDCTPair(dst0, dst1, src0, src1 []float64, norm Norm) (r0, r1 []float64) {
	return p.runPair(dst0, dst1, src0, src1, norm, true)
}

// checkPair panics, before anything is written, unless the four slices hold
// the plan's length and norm is known.
func (p *DCTPlan) checkPair(l0, l1, l2, l3 int, norm Norm) {
	norm.scale(1, false) // panics on an unknown Norm
	if n := p.k.n; l0 < n || l1 < n || l2 < n || l3 < n {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
}

func (p *DCTPlan) runPair(dst0, dst1, src0, src1 []float64, norm Norm, inverse bool) (r0, r1 []float64) {
	p.checkPair(len(dst0), len(dst1), len(src0), len(src1), norm)
	if p.fused && !overlap(dst0[:64], src1[:64]) && !overlap(dst1[:64], src0[:64]) && !overlap(dst0[:64], dst1[:64]) {
		dct4Pair64(dst0, dst1, src0, src1, norm.scale(128, inverse))
		return dst0[:64], dst1[:64]
	}
	return p.run(dst0, src0, norm, inverse), p.run(dst1, src1, norm, inverse)
}

// DCTMixedPair is DCTMixed on two streams at once, as DCTPair is to DCT. The
// Mixed forms of a fused plan are exactly float32 of the float64 pair.
func (p *DCTPlan) DCTMixedPair(dst0, dst1, src0, src1 []float32, norm Norm) (r0, r1 []float32) {
	return p.runMixedPair(dst0, dst1, src0, src1, norm, false)
}

// IDCTMixedPair is IDCTMixed on two streams at once.
func (p *DCTPlan) IDCTMixedPair(dst0, dst1, src0, src1 []float32, norm Norm) (r0, r1 []float32) {
	return p.runMixedPair(dst0, dst1, src0, src1, norm, true)
}

func (p *DCTPlan) runMixedPair(dst0, dst1, src0, src1 []float32, norm Norm, inverse bool) (r0, r1 []float32) {
	if p.typ != 4 {
		panic("fft: DCTPlan.DCTMixed needs a DCT-IV plan")
	}
	p.checkPair(len(dst0), len(dst1), len(src0), len(src1), norm)
	if p.fused && !overlap(dst0[:64], src1[:64]) && !overlap(dst1[:64], src0[:64]) && !overlap(dst0[:64], dst1[:64]) {
		dct4Pair64Mixed(dst0, dst1, src0, src1, norm.scale(128, inverse))
		return dst0[:64], dst1[:64]
	}
	return p.runMixed(dst0, src0, norm, inverse), p.runMixed(dst1, src1, norm, inverse)
}

// overlap reports whether the memory of a and b intersects; both are non-empty.
func overlap[A, B float32 | float64](a []A, b []B) bool {
	pa, pb := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	sa, sb := uintptr(len(a))*unsafe.Sizeof(a[0]), uintptr(len(b))*unsafe.Sizeof(b[0])
	return pa < pb+sb && pb < pa+sa
}
