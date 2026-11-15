package fft

import "unsafe"

//go:generate go run internal/gendct4/main.go internal/gendct4/amd64.go

// The fused DCT-IV of length 64 and 128 (go-fft/fft#132, go-avkit/decode#77:
// HE-AAC's QMF runs the 64 six times per slot, and the AAC short-block MDCT of 256
// samples is a DCT-IV of 128). A DCT-IV plan of one of those lengths on a platform
// that routes it (dct4FusedRoute) skips the pooled scratch, the n/2-point Plan and
// the separate twiddle passes: dct4n64* and dct4n128* in dct4_64_gen.go and
// dct4_128_gen.go are each one straight-line function whose values live in registers
// and on the stack, with every twiddle a constant. Same algorithm as dct4Even (a
// pre-twiddle, an n/2-point complex FFT, a post-twiddle), so the error is of the
// same order: the tests assert the bound against a double-double oracle.
//
// The pair forms run two independent transforms in one NEON kernel.

// dct4FusedLen reports whether n is a length with a fused kernel.
func dct4FusedLen(n int) bool { return n == 64 || n == 128 }

// runFused is run for the fused plan: DCT-IV of length n, so the logical size is 2n.
func (p *DCTPlan) runFused(dst, src []float64, norm Norm, inverse bool) []float64 {
	n := p.k.n
	if len(dst) < n || len(src) < n {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
	dct4Single(n, dst, src, norm.scale(2*n, inverse))
	return dst[:n]
}

// dct4Single runs the scalar kernel of length n, scaled by f unless f is 1.
func dct4Single(n int, dst, src []float64, f float64) {
	switch {
	case n == 64 && f != 1:
		dct4n64f64s(dst, src, f)
	case n == 64:
		dct4n64f64(dst, src)
	case f != 1:
		dct4n128f64s(dst, src, f)
	default:
		dct4n128f64(dst, src)
	}
}

// dct4SingleMixed is dct4Single on float32 data.
func dct4SingleMixed(n int, dst, src []float32, f float64) {
	switch {
	case n == 64 && f != 1:
		dct4n64f32s(dst, src, f)
	case n == 64:
		dct4n64f32(dst, src)
	case f != 1:
		dct4n128f32s(dst, src, f)
	default:
		dct4n128f32(dst, src)
	}
}

// runFusedMixed is runMixed for the fused plan.
func (p *DCTPlan) runFusedMixed(dst, src []float32, norm Norm, inverse bool) []float32 {
	n := p.k.n
	if len(dst) < n || len(src) < n {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
	dct4SingleMixed(n, dst, src, norm.scale(2*n, inverse))
	return dst[:n]
}

// DCTPair writes the DCT of src0 into dst0 and that of src1 into dst1, like
// p.DCT(dst0, src0, norm) followed by p.DCT(dst1, src1, norm), and returns
// dst0[:Len()] and dst1[:Len()]. On a DCT-IV plan of length 64 or 128 on arm64 it
// runs both transforms in one NEON kernel, one in each lane of every vector
// register, a third faster than two calls at 64 (Apple M4 Max, go-fft/fft#132);
// on amd64 with AVX2 and FMA3 a plan of length 64 runs an AVX2 kernel, twice as
// fast as two calls (Ryzen 9 7950X); elsewhere, and for other lengths, it is the
// two calls. The results are within the same error bound as DCT's, not bit for
// bit: the NEON and AVX2 kernels fuse multiply-adds and order the arithmetic
// differently from a single call, and on amd64 most of the 64 outputs differ from
// two single calls in the last bits. The slice rules are DCT's for each stream
// and are all checked before anything is written, so a panic leaves dst0 and dst1
// untouched. dst0 and dst1 must not overlap each other or the other stream's
// source (if they do the two calls run in order, as written).
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
	if n := p.k.n; p.pair && !overlap(dst0[:n], src1[:n]) && !overlap(dst1[:n], src0[:n]) && !overlap(dst0[:n], dst1[:n]) {
		dct4Pair(n, dst0, dst1, src0, src1, norm.scale(2*n, inverse))
		return dst0[:n], dst1[:n]
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
	if n := p.k.n; p.pair && !overlap(dst0[:n], src1[:n]) && !overlap(dst1[:n], src0[:n]) && !overlap(dst0[:n], dst1[:n]) {
		dct4PairMixed(n, dst0, dst1, src0, src1, norm.scale(2*n, inverse))
		return dst0[:n], dst1[:n]
	}
	return p.runMixed(dst0, src0, norm, inverse), p.runMixed(dst1, src1, norm, inverse)
}

// overlap reports whether the memory of a and b intersects; both are non-empty.
func overlap[A, B float32 | float64](a []A, b []B) bool {
	pa, pb := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	sa, sb := uintptr(len(a))*unsafe.Sizeof(a[0]), uintptr(len(b))*unsafe.Sizeof(b[0])
	return pa < pb+sb && pb < pa+sa
}
