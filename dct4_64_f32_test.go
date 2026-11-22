package fft

import (
	"math"
	"testing"
)

// The float32 pair and quad forms of the DCT-IV(64) (dct4_64_f32.go) are held to the
// #121 contract in single precision: each result within eps32·log2(n) (bound32) of
// the double-double oracle, relative to the oracle's 2-norm, on every platform (off
// amd64 the forms are the single calls, which the same tests then cover).

var f32PairNorms = accNorms

func f32Streams(n int, seed uint64) (s [4][]float32) {
	for i := range s {
		s[i] = accRealF32(n, seed*4+uint64(i))
	}
	return s
}

// TestDCT32PairQuadOracle: DCTPair, IDCTPair, DCTQuad and IDCTQuad against the oracle
// and against the generic float32 plan, for every norm.
func TestDCT32PairQuadOracle(t *testing.T) {
	const n = 64
	b := bound32(n)
	p := NewDCTPlan32(n, 4)
	ref := *p
	ref.quad = false // the generic path, whatever the platform
	var worst float64
	for seed := uint64(1); seed <= 8; seed++ {
		x := f32Streams(n, seed)
		for _, norm := range f32PairNorms {
			for _, inverse := range []bool{false, true} {
				var want [4][]cdd
				for i := range want {
					want[i] = oR2R(cddFromR(x[i]), true, 4, norm, inverse)
				}
				var d [4][]float32
				for i := range d {
					d[i] = make([]float32, n)
				}
				pairForm, quadForm := p.DCTPair, p.DCTQuad
				if inverse {
					pairForm, quadForm = p.IDCTPair, p.IDCTQuad
				}
				var q [4][]float32
				for i := range q {
					q[i] = make([]float32, n)
				}
				quadForm(q, x, norm)
				r0, r1 := pairForm(d[0], d[1], x[0], x[1], norm)
				r2, r3 := pairForm(d[2], d[3], x[2], x[3], norm)
				for i, got := range [][]float32{r0, r1, r2, r3} {
					if e := errR(got, want[i]); e > b {
						t.Errorf("pair seed %d %v inverse=%v stream %d: error %.3g exceeds %.3g", seed, norm, inverse, i, e, b)
					} else {
						worst = math.Max(worst, e)
					}
				}
				for i := range q {
					if e := errR(q[i], want[i]); e > b {
						t.Errorf("quad seed %d %v inverse=%v stream %d: error %.3g exceeds %.3g", seed, norm, inverse, i, e, b)
					} else {
						worst = math.Max(worst, e)
					}
					g := make([]float32, n)
					if inverse {
						ref.IDCT(g, x[i], norm)
					} else {
						ref.DCT(g, x[i], norm)
					}
					if e := errR(g, want[i]); e > b {
						t.Errorf("generic plan seed %d stream %d: error %.3g exceeds %.3g", seed, i, e, b)
					}
				}
			}
		}
	}
	t.Logf("worst relative error %.3g = %.3f of the bound %.3g (quad route %v)", worst, worst/b, b, p.quad)
}

// TestDCT32PairQuadInPlace: dst[i] may alias src[i]; overlapping streams fall back
// to the calls in order, as written.
func TestDCT32PairQuadInPlace(t *testing.T) {
	const n = 64
	p := NewDCTPlan32(n, 4)
	x := f32Streams(n, 99)
	var want, got [4][]float32
	for i := range x {
		want[i] = make([]float32, n)
		got[i] = append([]float32(nil), x[i]...)
	}
	p.DCTQuad(want, x, NormOrtho)
	p.DCTQuad(got, got, NormOrtho)
	for i := range want {
		for j := range want[i] {
			if want[i][j] != got[i][j] {
				t.Fatalf("quad in place: stream %d element %d: %v, out of place %v", i, j, got[i][j], want[i][j])
			}
		}
	}
	a, b := append([]float32(nil), x[0]...), append([]float32(nil), x[1]...)
	p.DCTPair(a, b, a, b, NormBackward)
	wa, wb := make([]float32, n), make([]float32, n)
	p.DCTPair(wa, wb, x[0], x[1], NormBackward)
	for j := 0; j < n; j++ {
		if a[j] != wa[j] || b[j] != wb[j] {
			t.Fatalf("pair in place differs from out of place at %d", j)
		}
	}
	// dst0 overlapping src1: the two calls in order, so stream 1 reads what stream 0
	// wrote.
	buf := append(append([]float32(nil), x[0]...), x[1]...)
	s0, s1 := buf[:n], buf[:n] // both streams read the same data; dst0 aliases src1
	d0, d1 := buf[:n], make([]float32, n)
	p.DCTPair(d0, d1, s0, s1, NormBackward)
	first := make([]float32, n)
	p.DCT(first, x[0], NormBackward)
	second := make([]float32, n)
	p.DCT(second, first, NormBackward)
	for j := 0; j < n; j++ {
		if d1[j] != second[j] {
			t.Fatalf("overlapping pair is not the two calls in order, element %d", j)
		}
	}
}

// TestDCT32PairQuadValidatesBeforeWriting: a short slice, in any position, panics
// with every destination untouched.
func TestDCT32PairQuadValidatesBeforeWriting(t *testing.T) {
	const n = 64
	p := NewDCTPlan32(n, 4)
	x := f32Streams(n, 5)
	for bad := 0; bad < 8; bad++ {
		var dst, src [4][]float32
		for i := range dst {
			dst[i] = make([]float32, n)
			for j := range dst[i] {
				dst[i][j] = 7
			}
			src[i] = x[i]
		}
		if bad < 4 {
			dst[bad] = dst[bad][:n-1]
		} else {
			src[bad-4] = src[bad-4][:n-1]
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("quad with a short slice (%d) did not panic", bad)
				}
			}()
			p.DCTQuad(dst, src, NormBackward)
		}()
		if bad%4 < 2 { // the short slice is one of the pair's four
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("pair with a short slice (%d) did not panic", bad)
					}
				}()
				p.DCTPair(dst[0], dst[1], src[0], src[1], NormBackward)
			}()
		}
		for i := range dst {
			for j := range dst[i][:len(dst[i])] {
				if dst[i][j] != 7 {
					t.Fatalf("short slice %d: dst[%d][%d] was written", bad, i, j)
				}
			}
		}
	}
}

// TestDCT32PairQuadRoutes: only DCT-IV of length 64 routes to the kernel, and the
// other plans give the same results through the single calls.
func TestDCT32PairQuadOtherPlans(t *testing.T) {
	for _, c := range []struct{ n, typ int }{{32, 4}, {64, 2}, {128, 4}, {64, 3}} {
		p := NewDCTPlan32(c.n, c.typ)
		if p.quad {
			t.Errorf("DCTPlan32(%d, %d) routes to the DCT-IV(64) kernel", c.n, c.typ)
		}
		x := f32Streams(c.n, 3)
		var d, q [4][]float32
		for i := range d {
			d[i] = make([]float32, c.n)
			q[i] = make([]float32, c.n)
		}
		p.DCTQuad(q, x, NormBackward)
		for i := range d {
			p.DCT(d[i], x[i], NormBackward)
			for j := range d[i] {
				if d[i][j] != q[i][j] {
					t.Fatalf("n=%d typ=%d: quad differs from the single call", c.n, c.typ)
				}
			}
		}
	}
}
