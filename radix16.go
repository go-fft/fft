package fft

// Radix-16 Stockham passes (Round 19). A radix-16 pass does the work of two
// radix-4 passes in one trip through memory, and its 16-point butterfly needs
// fewer operations than the two passes it replaces: 15 twiddle products per 16
// points against 2·(3/4)·16 = 24 for two twiddled radix-4 passes. It is only
// in the factorization where the amd64 pass kernels run it (radix16Factors);
// the Go pass below is their reference, operation for operation, and their
// fallback.
//
// The butterfly is the 4×4 split (Cooley–Tukey with n = n2 + 4·n1, k = k1 +
// 4·k2, as in FFTW's n1_16 and t1_16 codelets, Frigo 1999):
//
//	Y[n2][k1] = W16^(n2·k1) · Σ_n1 x[n2+4·n1]·W4^(n1·k1)   (four bfly4, then the internal twiddles)
//	X[k1+4·k2] = Σ_n2 Y[n2][k1]·W4^(n2·k2)                  (four bfly4)
//
// with W16 = exp(s·2πi/16) and W4 = s·i (s = -1 forward, +1 inverse). The
// internal twiddles are W16^e for e = n2·k1 in {1, 2, 3, 2, 4, 6, 3, 6, 9}:
// e = 4 is the exact rotation rotS; every other one is radix16Rot, a product
// by the constant cos θ + s·i·sin θ (θ = 2π·e/16) in four packed operations.

// radix16Table maps the powers of two that take radix-16 passes to their
// factorization; it is per-architecture (route_amd64.go; nil elsewhere).
var radix16Table = radix16TableDefault()

// cachedPlanNo16 returns the cached plan for length n factored without
// radix 16 or radix 12 (skFactorizeOrder; radix 12 has no batched kernel
// either, Round 24). An N-D plan's axes other than the last (its columns, which run as
// batched passes over strips) take it: there a radix-16 pass reads or writes
// sixteen streams a whole column stride apart. A batched radix-16 kernel was
// built and timed (Round 19): n×n time with radix-16 columns ÷ without, on
// Haswell, was 0.89 at 128, 1.18 at 256, 0.99 at 1024 and 1.07 at 2048, and
// on Zen 3 every 2-D row from 128 to 1024 lost 9–15% with it; so the
// kernel was dropped. The lengths of intelStripOrder take it too (Round 28).
func cachedPlanNo16(n int) *Plan {
	planMu.Lock()
	defer planMu.Unlock()
	p, ok := planNo16Cache[n]
	if !ok {
		p = &Plan{n: n, sk: newSKPlanFactors(n, skFactorizeOrder(n, compOddFirst))}
		planNo16Cache[n] = p
	}
	return p
}

var planNo16Cache = map[int]*Plan{}

// cos and sin of π/8, the radix-16 butterfly's constants (with hsqt2).
const (
	r16C1 = 0.92387953251128675612818318939678828682241662586364 // cos(π/8)
	r16S1 = 0.38268343236508977172845998403039886676134456248563 // sin(π/8)
)

// radix16Rot returns a·(c + i·ss): (c·re − ss·im, c·im + ss·re), each product
// rounded, then the sum. ss is s·sin θ, which the caller forms exactly. The
// kernels compute the same values as c·a plus swap(a)·(−ss, ss), whose
// products are the negated and the same ones: x − y is x + (−y) in IEEE 754.
func radix16Rot(a complex128, c, ss float64) complex128 {
	return complex(c*real(a)-ss*imag(a), c*imag(a)+ss*real(a))
}

// bfly16 is the size-16 DFT of x into y (output order 0..15); s is the
// direction sign. The operation order is the kernels' (genStockham16 in
// asmgen/amd64/gen.go).
func bfly16(y, x *[16]complex128, s float64) {
	var v [4][4]complex128 // v[n2][k1]
	for n2 := 0; n2 < 4; n2++ {
		y0, y1, y2, y3 := bfly4(x[n2], x[n2+4], x[n2+8], x[n2+12], s)
		v[n2] = [4]complex128{y0, y1, y2, y3}
	}
	// The internal twiddles W16^(n2·k1), in the order the kernels apply them.
	v[1][1] = radix16Rot(v[1][1], r16C1, s*r16S1)
	v[1][2] = radix16Rot(v[1][2], hsqt2, s*hsqt2)
	v[1][3] = radix16Rot(v[1][3], r16S1, s*r16C1)
	v[2][1] = radix16Rot(v[2][1], hsqt2, s*hsqt2)
	v[2][2] = rotS(v[2][2], s)
	v[2][3] = radix16Rot(v[2][3], -hsqt2, s*hsqt2)
	v[3][1] = radix16Rot(v[3][1], r16S1, s*r16C1)
	v[3][2] = radix16Rot(v[3][2], -hsqt2, s*hsqt2)
	v[3][3] = radix16Rot(v[3][3], -r16C1, s*-r16S1)
	for k1 := 0; k1 < 4; k1++ {
		y[k1], y[k1+4], y[k1+8], y[k1+12] = bfly4(v[0][k1], v[1][k1], v[2][k1], v[3][k1], s)
	}
}

// pass16 is the radix-16 Stockham pass (see pass4 for the layout).
func pass16(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass16last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	var x, y [16]complex128
	for k := 0; k < l1; k++ {
		b := 16 * ido * k
		for i := 0; i < ido; i++ {
			for j := range x {
				x[j] = cc[b+j*ido+i]
			}
			bfly16(&y, &x, s)
			ch[ido*k+i] = y[0]
			for j := 1; j < 16; j++ {
				v := y[j]
				if i > 0 {
					v *= tw[(j-1)*m+i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}

// pass16last is pass16 with ido == 1: block k is cc[16·k : 16·k+16], and
// output j the unit-stride stream ch[j·l1 : (j+1)·l1].
func pass16last(l1 int, cc, ch []complex128, s float64) {
	var x, y [16]complex128
	for k := 0; k < l1; k++ {
		copy(x[:], cc[16*k:16*k+16])
		bfly16(&y, &x, s)
		for j := range y {
			ch[j*l1+k] = y[j]
		}
	}
}
