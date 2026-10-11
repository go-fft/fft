package fft

// Radix-32 Stockham passes (Round 38). FFTW reaches 256 and 1024 points in
// two passes with 32- and 64-point codelets (t2fv_32, n2fv_8; Round 36's
// plans); go-fft's largest radix was 16. A radix-32 pass does the work of a
// radix-4 and a radix-8 pass in one trip through memory. Where it is in the
// factorization (r32Table, route_amd64.go) the amd64 AVX2 kernels run it
// (genR32 in asmgen/amd64/gen.go); the Go pass below is their reference,
// operation for operation, and their fallback.
//
// The butterfly is the 8×4 split (Cooley–Tukey with n = n2 + 4·n1, k = k1 +
// 8·k2):
//
//	V[n2][k1] = W32^(n2·k1) · Σ_n1 x[n2+4·n1]·W8^(n1·k1)   (four bfly8, then the internal twiddles)
//	X[k1+8·k2] = Σ_n2 V[n2][k1]·W4^(n2·k2)                  (eight bfly4)
//
// with W32 = exp(s·2πi/32), W8 and W4 likewise (s = -1 forward, +1
// inverse). The internal twiddles are W32^e for e = n2·k1, n2 in 1..3 and k1
// in 1..7: e = 8 is the exact rotation rotS, every other one radix16Rot, a
// product by the constant cos θ + s·i·sin θ (θ = 2π·e/32) in four packed
// operations.

// r32Table maps the lengths that take radix-32 passes to their
// factorization; it is per-architecture (route_amd64.go; nil elsewhere).
var r32Table = r32TableDefault()

// r32CosK is cos(k·π/16) for k = 0..8: every cos and sin of θ = 2π·e/32
// is one of them up to its sign (r32Trig).
var r32CosK = [9]float64{
	1,
	0.98078528040323044912618223613423903697393373089334, // cos(π/16)
	r16C1, // cos(π/8)
	0.83146961230254523707878837761790575673856081198725, // cos(3π/16)
	hsqt2, // cos(π/4)
	0.55557023301960222474283081394853287437493719075480, // cos(5π/16)
	r16S1, // cos(3π/8)
	0.19509032201612826784828486847702224092769161775195, // cos(7π/16)
	0,
}

// r32Cos returns cos(π·e/16) for 0 <= e < 32, exactly a table value or its
// negation.
func r32Cos(e int) float64 {
	switch {
	case e <= 8:
		return r32CosK[e]
	case e <= 16:
		return -r32CosK[16-e]
	case e <= 24:
		return -r32CosK[e-16]
	}
	return r32CosK[32-e]
}

// r32Trig returns cos θ and sin θ for θ = π·e/16, 0 <= e < 32.
func r32Trig(e int) (c, s float64) { return r32Cos(e), r32Cos((e + 24) % 32) }

// bfly32 is the size-32 DFT of x into y (output order 0..31); s is the
// direction sign. The operation order is the kernels' (r32Body in
// asmgen/amd64/gen.go).
func bfly32(y, x *[32]complex128, s float64) {
	var v [4][8]complex128 // v[n2][k1]
	for n2 := 0; n2 < 4; n2++ {
		bfly8(&v[n2], x[n2], x[n2+4], x[n2+8], x[n2+12], x[n2+16], x[n2+20], x[n2+24], x[n2+28], s)
		for k1 := 1; k1 < 8 && n2 > 0; k1++ {
			v[n2][k1] = r32Twiddle(v[n2][k1], n2*k1, s)
		}
	}
	for k1 := 0; k1 < 8; k1++ {
		y[k1], y[k1+8], y[k1+16], y[k1+24] = bfly4(v[0][k1], v[1][k1], v[2][k1], v[3][k1], s)
	}
}

// r32Twiddle returns a·W32^e (1 <= e <= 21): rotS for e = 8, radix16Rot
// with the constants of θ = π·e/16 for the others.
func r32Twiddle(a complex128, e int, s float64) complex128 {
	if e == 8 {
		return rotS(a, s)
	}
	c, sn := r32Trig(e)
	return radix16Rot(a, c, s*sn)
}

// pass32 is the radix-32 Stockham pass (see pass4 for the layout).
func pass32(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	s := dirSign(inverse)
	if ido == 1 {
		pass32last(l1, cc, ch, s)
		return
	}
	m := ido - 1
	var x, y [32]complex128
	for k := 0; k < l1; k++ {
		b := 32 * ido * k
		for i := 0; i < ido; i++ {
			for j := range x {
				x[j] = cc[b+j*ido+i]
			}
			bfly32(&y, &x, s)
			ch[ido*k+i] = y[0]
			for j := 1; j < 32; j++ {
				v := y[j]
				if i > 0 {
					v *= tw[(j-1)*m+i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}

// pass32last is pass32 with ido == 1: block k is cc[32·k : 32·k+32], and
// output j the unit-stride stream ch[j·l1 : (j+1)·l1].
func pass32last(l1 int, cc, ch []complex128, s float64) {
	var x, y [32]complex128
	for k := 0; k < l1; k++ {
		copy(x[:], cc[32*k:32*k+32])
		bfly32(&y, &x, s)
		for j := range y {
			ch[j*l1+k] = y[j]
		}
	}
}
