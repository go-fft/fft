package fft

// Radix-12 Stockham passes (Round 24): the prime-factor (Good–Thomas)
// algorithm for 12 = 3·4. With n = (4·n1 + 3·n2) mod 12 and k = (4·k1 +
// 9·k2) mod 12, n·k ≡ 4·n1·k1 + 3·n2·k2 (mod 12), so the 12-point DFT is four
// 3-point DFTs (over n1, one per n2) followed by three 4-point DFTs (over n2,
// one per k1), with no twiddle between them (Good 1958, Thomas 1963; Temperton,
// "Implementation of a self-sorting in-place prime factor FFT algorithm",
// J. Comput. Phys. 58, 1985). FFTW's t1_12 counts 118 additions and 68
// multiplications for twelve points, 4.32 per point per bit, against 6.31
// for t1_3 and 4.25 for t1_4 (FFTW 3.3.10 codelet headers). The Go pass below
// is the kernels' reference, operation for operation, and their fallback.

// comp12In is the input index of (n2, n1); comp12Out the output index of
// (k1, k2).
var (
	comp12In  = [4][3]int{{0, 4, 8}, {3, 7, 11}, {6, 10, 2}, {9, 1, 5}}
	comp12Out = [3][4]int{{0, 9, 6, 3}, {4, 1, 10, 7}, {8, 5, 2, 11}}
)

// bfly12 is the size-12 DFT of x, written into y (output order 0..11); s is
// the direction sign.
func bfly12(y, x *[12]complex128, s float64) {
	var a [4][3]complex128
	for n2 := range a {
		in := &comp12In[n2]
		a[n2][0], a[n2][1], a[n2][2] = bfly3(x[in[0]], x[in[1]], x[in[2]], s)
	}
	for k1 := range comp12Out {
		out := &comp12Out[k1]
		y[out[0]], y[out[1]], y[out[2]], y[out[3]] = bfly4(a[0][k1], a[1][k1], a[2][k1], a[3][k1], s)
	}
}

func pass12(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	s := dirSign(inverse)
	if ido == 1 {
		pass12last(l1, cc, ch, s)
		return
	}
	m := ido - 1
	var x, y [12]complex128
	for k := 0; k < l1; k++ {
		b := 12 * ido * k
		for i := 0; i < ido; i++ {
			for j := range x {
				x[j] = cc[b+j*ido+i]
			}
			bfly12(&y, &x, s)
			ch[ido*k+i] = y[0]
			for j := 1; j < 12; j++ {
				v := y[j]
				if i > 0 {
					v *= tw[(j-1)*m+i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}

func pass12last(l1 int, cc, ch []complex128, s float64) {
	var x, y [12]complex128
	for k := 0; k < l1; k++ {
		copy(x[:], cc[12*k:12*k+12])
		bfly12(&y, &x, s)
		for j := range y {
			ch[k+j*l1] = y[j]
		}
	}
}
