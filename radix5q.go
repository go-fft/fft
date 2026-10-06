package fft

// Radix-10, -15 and -20 Stockham passes (Round 26): the prime-factor (Good–
// Thomas) algorithm for r = 5·q, q = 2, 3, 4, the way radix12.go does 12 =
// 3·4. With n = (q·n1 + 5·n2) mod r and k = (a·k1 + b·k2) mod r, where a ≡ 1
// (mod 5), a ≡ 0 (mod q), b ≡ 0 (mod 5) and b ≡ 1 (mod q), n·k ≡ q·n1·k1 +
// 5·n2·k2 (mod r): the r-point DFT is q 5-point DFTs (over n1, one per n2)
// followed by five q-point DFTs (over n2, one per k1), with no twiddle between
// them (Good 1958, Thomas 1963; Temperton, J. Comput. Phys. 58, 1985). FFTW's
// twiddled codelets count 5.24 (t1_10), 5.53 (t1_15) and 4.56 (t1_20)
// operations per point per bit against 5.96, 6.34 and 5.39 for the two passes
// each replaces (FFTW 3.3.10 codelet headers, Round 24). The Go pass below is
// the kernels' reference, operation for operation, and their fallback.

// comp2Map is the index map of one radix r = 5·q: in[n2][n1] is the input
// index of (n1, n2), out[k1][k2] the output index of (k1, k2).
type comp2Map struct {
	q   int
	in  [4][5]int
	out [5][4]int
}

// comp2Maps holds the maps by radix.
var comp2Maps = [21]*comp2Map{10: comp2NewMap(2), 15: comp2NewMap(3), 20: comp2NewMap(4)}

// comp2NewMap builds the map of radix 5·q.
func comp2NewMap(q int) *comp2Map {
	r := 5 * q
	var a, b int
	for x := range r {
		if x%5 == 1 && x%q == 0 {
			a = x
		}
		if x%5 == 0 && x%q == 1 {
			b = x
		}
	}
	m := &comp2Map{q: q}
	for n2 := range q {
		for n1 := range 5 {
			m.in[n2][n1] = (q*n1 + 5*n2) % r
		}
	}
	for k1 := range 5 {
		for k2 := range q {
			m.out[k1][k2] = (a*k1 + b*k2) % r
		}
	}
	return m
}

// comp2Bfly is the size-5q DFT of x, written into y (output order 0..r-1); s
// is the direction sign.
func comp2Bfly(y, x *[20]complex128, m *comp2Map, s float64) {
	var a [4][5]complex128
	for n2 := range m.q {
		in := &m.in[n2]
		a[n2][0], a[n2][1], a[n2][2], a[n2][3], a[n2][4] = bfly5(x[in[0]], x[in[1]], x[in[2]], x[in[3]], x[in[4]], s)
	}
	for k1 := range 5 {
		o := &m.out[k1]
		switch m.q {
		case 2:
			y[o[0]], y[o[1]] = a[0][k1]+a[1][k1], a[0][k1]-a[1][k1]
		case 3:
			y[o[0]], y[o[1]], y[o[2]] = bfly3(a[0][k1], a[1][k1], a[2][k1], s)
		default:
			y[o[0]], y[o[1]], y[o[2]], y[o[3]] = bfly4(a[0][k1], a[1][k1], a[2][k1], a[3][k1], s)
		}
	}
}

// comp2Pass is the Stockham pass of radix r = 10, 15 or 20.
func comp2Pass(r, ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	s := dirSign(inverse)
	mp := comp2Maps[r]
	var x, y [20]complex128
	if ido == 1 {
		for k := 0; k < l1; k++ {
			copy(x[:r], cc[r*k:r*k+r])
			comp2Bfly(&y, &x, mp, s)
			for j := range r {
				ch[k+j*l1] = y[j]
			}
		}
		return
	}
	m := ido - 1
	for k := 0; k < l1; k++ {
		b := r * ido * k
		for i := 0; i < ido; i++ {
			for j := range r {
				x[j] = cc[b+j*ido+i]
			}
			comp2Bfly(&y, &x, mp, s)
			ch[ido*k+i] = y[0]
			for j := 1; j < r; j++ {
				v := y[j]
				if i > 0 {
					v *= tw[(j-1)*m+i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}
