package fft

// Radix-9 Stockham passes (Round 35). On Cascade Lake complex 1296 =
// 2^4·3^4 ran 3·3·12·12 at 1.20× FFTW's time, which plans it 9·12·12
// (t1fv_9, t1fv_12, n1fv_12): its two radix-3 passes cost 0.8 ns per point
// each, the radix-12 ones 1.1 and 0.8, so two trips through memory went to
// three bits of the transform. The amd64 kernels (csPass9AVX2, csLast9AVX2)
// run where the composite rule takes radix 9; the Go pass below is their
// reference, operation for operation, and their fallback.
//
// The butterfly is the 3×3 split (Cooley–Tukey with n = n2 + 3·n1, k = k1 +
// 3·k2, as FFTW's n1_9 and t1_9 codelets):
//
//	Y[n2][k1] = W9^(n2·k1) · Σ_n1 x[n2+3·n1]·W3^(n1·k1)   (three bfly3, then the internal twiddles)
//	X[k1+3·k2] = Σ_n2 Y[n2][k1]·W3^(n2·k2)                  (three bfly3)
//
// with W9 = exp(s·2πi/9) (s = -1 forward, +1 inverse). The internal twiddles
// are W9^e for e = n2·k1 in {1, 2, 2, 4}, each radix16Rot's product by the
// constant cos θ + s·i·sin θ (θ = 2π·e/9).

// cos and sin of 2π·e/9 for e = 1, 2, 4, the radix-9 butterfly's constants.
const (
	r9C1 = 0.76604444311897803520239265055541667393583245708040 // cos(2π/9)
	r9S1 = 0.64278760968653932632264340990726343290755988420568 // sin(2π/9)
	r9C2 = 0.17364817766693034885171662676931479600037567718407 // cos(4π/9)
	r9S2 = 0.98480775301220805936674302458952301367064325171984 // sin(4π/9)
	r9C4 = -0.9396926207859083840541092773247314699362081342645 // cos(8π/9)
	r9S4 = 0.34202014332566873304409961468225958076308336751416 // sin(8π/9)
)

// bfly9 is the size-9 DFT of x into y (output order 0..8); s is the
// direction sign. The operation order is the kernels' (csBody9 in
// asmgen/amd64/gen.go).
func bfly9(y, x *[9]complex128, s float64) {
	var v [3][3]complex128 // v[n2][k1]
	for n2 := 0; n2 < 3; n2++ {
		v[n2][0], v[n2][1], v[n2][2] = bfly3(x[n2], x[n2+3], x[n2+6], s)
	}
	v[1][1] = radix16Rot(v[1][1], r9C1, s*r9S1)
	v[1][2] = radix16Rot(v[1][2], r9C2, s*r9S2)
	v[2][1] = radix16Rot(v[2][1], r9C2, s*r9S2)
	v[2][2] = radix16Rot(v[2][2], r9C4, s*r9S4)
	for k1 := 0; k1 < 3; k1++ {
		y[k1], y[k1+3], y[k1+6] = bfly3(v[0][k1], v[1][k1], v[2][k1], s)
	}
}

// pass9 is the radix-9 Stockham pass (see pass4 for the layout).
func pass9(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	s := dirSign(inverse)
	if ido == 1 {
		pass9last(l1, cc, ch, s)
		return
	}
	m := ido - 1
	var x, y [9]complex128
	for k := 0; k < l1; k++ {
		b := 9 * ido * k
		for i := 0; i < ido; i++ {
			for j := range x {
				x[j] = cc[b+j*ido+i]
			}
			bfly9(&y, &x, s)
			ch[ido*k+i] = y[0]
			for j := 1; j < 9; j++ {
				v := y[j]
				if i > 0 {
					v *= tw[(j-1)*m+i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}

// pass9last is pass9 with ido == 1: block k is cc[9·k : 9·k+9], and output
// j the unit-stride stream ch[j·l1 : (j+1)·l1].
func pass9last(l1 int, cc, ch []complex128, s float64) {
	var x, y [9]complex128
	for k := 0; k < l1; k++ {
		copy(x[:], cc[9*k:9*k+9])
		bfly9(&y, &x, s)
		for j := range y {
			ch[j*l1+k] = y[j]
		}
	}
}
