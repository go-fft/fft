package fft

import "github.com/go-fft/fft/internal/kernels"

// Radix-32 passes, in FMA plans only (Round 42). FFTW does complex 1,024 in
// two passes of 32-point codelets (t2fv_32, then n2fv_32) where the default
// plans take three or four passes of radix 8, 16 and 4; Round 38 built a
// radix-32 pass without fused products and it lost to split radix 8. The
// FMA plans' radix-32 pass is the split layout's (kernels: skSplit32…FMA,
// intelSplit512R32…FMA, skLast32AVX2FMA), with every product fused as
// FFTW's FMA codelets fuse them; no default plan has one, so the default
// plans keep their bits.

// bfly32FMA is the size-32 DFT of x into y (output order 0..31) in an FMA
// plan; s is the direction sign. It is the 4×8 split: four radix-8
// butterflies (bfly8FMA) over the inputs n2 + 4·n1, the internal twiddles
// W32^(n2·k1) = cos θ + s·i·sin θ with one rounding per part
// (radix16RotFMA), W32^8 = s·i as the exact rotS, then eight radix-4
// butterflies over V[0..3][k1], whose outputs are k1 + 8·k2. The kernels
// compute exactly these operations.
func bfly32FMA(y, x *[32]complex128, s float64) {
	var v [4][8]complex128 // v[n2][k1]
	for n2 := 0; n2 < 4; n2++ {
		var a [8]complex128
		for n1 := range a {
			a[n1] = x[n2+4*n1]
		}
		bfly8FMA(&v[n2], &a, s)
	}
	for n2 := 1; n2 < 4; n2++ {
		for k1 := 1; k1 < 8; k1++ {
			if e := n2 * k1; e == 8 {
				v[n2][k1] = rotS(v[n2][k1], s)
			} else {
				c, sn := kernels.R32Trig(e)
				v[n2][k1] = radix16RotFMA(v[n2][k1], c, s*sn)
			}
		}
	}
	for k1 := 0; k1 < 8; k1++ {
		y[k1], y[k1+8], y[k1+16], y[k1+24] = bfly4(v[0][k1], v[1][k1], v[2][k1], v[3][k1], s)
	}
}
