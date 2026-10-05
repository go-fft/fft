package fft

// Single-precision (complex64) Stockham passes.
//
// These are the scalar passes of stockham.go, the float64 engine, with every
// complex128 a complex64 and every float64 a float32: the same radices, the
// same pass layout (CC(i,m,k) = cc[i+ido·(m+r·k)], CH(i,k,m) =
// ch[i+ido·(k+l1·m)]), the same butterflies, so a correctness argument made
// for one holds for the other. They are a copy rather than one generic
// engine instantiated twice because the generic engine made the float64
// transforms slower (see the comment on Plan32); a change to a pass in
// stockham.go should be carried here.
//
// The radix constants (sin120, c51…, c71…, hsqt2) are shared untyped
// constants, so here they round once to float32.

// f32Mul is the complex product a·b computed in float32. Go's a*b on
// complex64 widens both operands to float64, multiplies there and rounds
// back (cmd/compile, ssagen: "Compute in Float64 to minimize cancellation
// error"): four conversions to and two from float64 per product, which made
// the first version of these passes 1.15–1.42× slower than the float64 ones.
// Written out, the product stays in float32 like fftwf_'s and
// pocketfft<float>'s, and costs what a complex128 product does.
func f32Mul(a, b complex64) complex64 {
	return complex(real(a)*real(b)-imag(a)*imag(b), real(a)*imag(b)+imag(a)*real(b))
}

// f32DirSign is -1 for the forward transform and +1 for the inverse.
func f32DirSign(inverse bool) float32 {
	if inverse {
		return 1
	}
	return -1
}

// f32RotS multiplies z by s·i: by -i forward, +i inverse.
func f32RotS(z complex64, s float32) complex64 {
	return complex(-s*imag(z), s*real(z))
}

// f32Stream returns the length-m run of buf starting at off.
func f32Stream(buf []complex64, off, m int) []complex64 { return buf[off:][:m] }

func f32Pass2(ido, l1 int, cc, ch, tw []complex64) {
	if ido == 1 {
		f32Pass2last(l1, cc, ch)
		return
	}
	m := ido - 1
	w := tw[:m]
	for k := 0; k < l1; k++ {
		a, b := cc[2*ido*k], cc[2*ido*k+ido]
		ch[ido*k] = a + b
		ch[ido*(k+l1)] = a - b
		x0, x1 := f32Stream(cc, 2*ido*k+1, m), f32Stream(cc, 2*ido*k+ido+1, m)
		o0, o1 := f32Stream(ch, ido*k+1, m), f32Stream(ch, ido*(k+l1)+1, m)
		for i := range w {
			a, b := x0[i], x1[i]
			o0[i] = a + b
			o1[i] = f32Mul(a-b, w[i])
		}
	}
}

// f32Bfly3 is the size-3 DFT; s is the direction sign.
func f32Bfly3(x0, x1, x2 complex64, s float32) (y0, y1, y2 complex64) {
	t1, t2 := x1+x2, x1-x2
	ca := complex(real(x0)-0.5*real(t1), imag(x0)-0.5*imag(t1))
	cb := f32RotS(complex(sin120*real(t2), sin120*imag(t2)), s)
	return x0 + t1, ca + cb, ca - cb
}

func f32Pass3(ido, l1 int, cc, ch, tw []complex64, inverse bool) {
	if ido == 1 {
		f32Pass3last(l1, cc, ch, f32DirSign(inverse))
		return
	}
	s := f32DirSign(inverse)
	m := ido - 1
	w1, w2 := tw[:m], tw[m:][:m]
	for k := 0; k < l1; k++ {
		b := 3 * ido * k
		y0, y1, y2 := f32Bfly3(cc[b], cc[b+ido], cc[b+2*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)] = y0, y1, y2
		x0, x1, x2 := f32Stream(cc, b+1, m), f32Stream(cc, b+ido+1, m), f32Stream(cc, b+2*ido+1, m)
		o0, o1, o2 := f32Stream(ch, ido*k+1, m), f32Stream(ch, ido*(k+l1)+1, m), f32Stream(ch, ido*(k+2*l1)+1, m)
		for i := range w1 {
			y0, y1, y2 := f32Bfly3(x0[i], x1[i], x2[i], s)
			o0[i] = y0
			o1[i] = f32Mul(y1, w1[i])
			o2[i] = f32Mul(y2, w2[i])
		}
	}
}

// f32Bfly4 is the size-4 DFT in output order 0,1,2,3.
func f32Bfly4(x0, x1, x2, x3 complex64, s float32) (y0, y1, y2, y3 complex64) {
	t2, t1 := x0+x2, x0-x2
	t3, t4 := x1+x3, x1-x3
	t4 = f32RotS(t4, s)
	return t2 + t3, t1 + t4, t2 - t3, t1 - t4
}

func f32Pass4(ido, l1 int, cc, ch, tw []complex64, inverse bool) {
	if ido == 1 {
		f32Pass4last(l1, cc, ch, f32DirSign(inverse))
		return
	}
	s := f32DirSign(inverse)
	m := ido - 1
	w1, w2, w3 := tw[:m], tw[m:][:m], tw[2*m:][:m]
	for k := 0; k < l1; k++ {
		b := 4 * ido * k
		y0, y1, y2, y3 := f32Bfly4(cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)], ch[ido*(k+3*l1)] = y0, y1, y2, y3
		x0, x1 := f32Stream(cc, b+1, m), f32Stream(cc, b+ido+1, m)
		x2, x3 := f32Stream(cc, b+2*ido+1, m), f32Stream(cc, b+3*ido+1, m)
		o0, o1 := f32Stream(ch, ido*k+1, m), f32Stream(ch, ido*(k+l1)+1, m)
		o2, o3 := f32Stream(ch, ido*(k+2*l1)+1, m), f32Stream(ch, ido*(k+3*l1)+1, m)
		for i := range w1 {
			y0, y1, y2, y3 := f32Bfly4(x0[i], x1[i], x2[i], x3[i], s)
			o0[i] = y0
			o1[i] = f32Mul(y1, w1[i])
			o2[i] = f32Mul(y2, w2[i])
			o3[i] = f32Mul(y3, w3[i])
		}
	}
}

// f32Bfly5 is the size-5 DFT in output order 0..4.
func f32Bfly5(a, x1, x2, x3, x4 complex64, s float32) (y0, y1, y2, y3, y4 complex64) {
	t1, t2 := x1+x4, x1-x4
	t3, t4 := x2+x3, x2-x3
	r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
	r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
	i1 := f32RotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
	i2 := f32RotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
	return a + t1 + t3, r1 + i1, r2 + i2, r2 - i2, r1 - i1
}

func f32Pass5(ido, l1 int, cc, ch, tw []complex64, inverse bool) {
	if ido == 1 {
		f32Pass5last(l1, cc, ch, f32DirSign(inverse))
		return
	}
	s := f32DirSign(inverse)
	m := ido - 1
	w1, w2, w3, w4 := tw[:m], tw[m:][:m], tw[2*m:][:m], tw[3*m:][:m]
	for k := 0; k < l1; k++ {
		b := 5 * ido * k
		y0, y1, y2, y3, y4 := f32Bfly5(cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)] = y0, y1, y2
		ch[ido*(k+3*l1)], ch[ido*(k+4*l1)] = y3, y4
		x0, x1, x2 := f32Stream(cc, b+1, m), f32Stream(cc, b+ido+1, m), f32Stream(cc, b+2*ido+1, m)
		x3, x4 := f32Stream(cc, b+3*ido+1, m), f32Stream(cc, b+4*ido+1, m)
		o0, o1, o2 := f32Stream(ch, ido*k+1, m), f32Stream(ch, ido*(k+l1)+1, m), f32Stream(ch, ido*(k+2*l1)+1, m)
		o3, o4 := f32Stream(ch, ido*(k+3*l1)+1, m), f32Stream(ch, ido*(k+4*l1)+1, m)
		for i := range w1 {
			// f32Bfly5, inlined by hand: it is over the inliner's budget.
			a := x0[i]
			t1, t2 := x1[i]+x4[i], x1[i]-x4[i]
			t3, t4 := x2[i]+x3[i], x2[i]-x3[i]
			r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
			r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
			i1 := f32RotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
			i2 := f32RotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
			o0[i] = a + t1 + t3
			o1[i] = f32Mul(r1+i1, w1[i])
			o2[i] = f32Mul(r2+i2, w2[i])
			o3[i] = f32Mul(r2-i2, w3[i])
			o4[i] = f32Mul(r1-i1, w4[i])
		}
	}
}

// f32Bfly7 is the size-7 DFT, written into y (output order 0..6).
func f32Bfly7(y *[7]complex64, a, b, c, d, e, f, g complex64, s float32) {
	s16, d16 := b+g, b-g
	s25, d25 := c+f, c-f
	s34, d34 := d+e, d-e
	y[0] = a + s16 + s25 + s34
	r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
	r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
	r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
	i1 := f32RotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
	i2 := f32RotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
	i3 := f32RotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
	y[1], y[6] = r1+i1, r1-i1
	y[2], y[5] = r2+i2, r2-i2
	y[3], y[4] = r3+i3, r3-i3
}

func f32Pass7(ido, l1 int, cc, ch, tw []complex64, inverse bool) {
	if ido == 1 {
		f32Pass7last(l1, cc, ch, f32DirSign(inverse))
		return
	}
	s := f32DirSign(inverse)
	m := ido - 1
	var w [7][]complex64
	for j := 1; j < 7; j++ {
		w[j] = tw[(j-1)*m:][:m]
	}
	var y [7]complex64
	for k := 0; k < l1; k++ {
		b := 7 * ido * k
		f32Bfly7(&y, cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], cc[b+5*ido], cc[b+6*ido], s)
		for j := 0; j < 7; j++ {
			ch[ido*(k+j*l1)] = y[j]
		}
		var x, o [7][]complex64
		for j := 0; j < 7; j++ {
			x[j] = f32Stream(cc, b+j*ido+1, m)
			o[j] = f32Stream(ch, ido*(k+j*l1)+1, m)
		}
		x0, x1, x2, x3, x4, x5, x6 := x[0], x[1][:m], x[2][:m], x[3][:m], x[4][:m], x[5][:m], x[6][:m]
		o0, o1, o2, o3, o4, o5, o6 := o[0][:m], o[1][:m], o[2][:m], o[3][:m], o[4][:m], o[5][:m], o[6][:m]
		w1, w2, w3, w4, w5, w6 := w[1][:m], w[2][:m], w[3][:m], w[4][:m], w[5][:m], w[6][:m]
		for i := range x0 {
			// f32Bfly7, inlined by hand: it is over the inliner's budget.
			a := x0[i]
			s16, d16 := x1[i]+x6[i], x1[i]-x6[i]
			s25, d25 := x2[i]+x5[i], x2[i]-x5[i]
			s34, d34 := x3[i]+x4[i], x3[i]-x4[i]
			r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
			r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
			r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
			i1 := f32RotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
			i2 := f32RotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
			i3 := f32RotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
			o0[i] = a + s16 + s25 + s34
			o1[i] = f32Mul(r1+i1, w1[i])
			o6[i] = f32Mul(r1-i1, w6[i])
			o2[i] = f32Mul(r2+i2, w2[i])
			o5[i] = f32Mul(r2-i2, w5[i])
			o3[i] = f32Mul(r3+i3, w3[i])
			o4[i] = f32Mul(r3-i3, w4[i])
		}
	}
}

// f32Passg is the general radix-r pass (r an odd prime without a specialized
// butterfly): an O(r²) size-r DFT per point, then the twiddle.
func f32Passg(r, ido, l1 int, cc, ch, tw, rt []complex64) {
	var bufArr [maxRadix]complex64
	x := bufArr[:r]
	for k := 0; k < l1; k++ {
		base := ido * r * k
		for i := 0; i < ido; i++ {
			for m := 0; m < r; m++ {
				x[m] = cc[base+m*ido+i]
			}
			for q := 0; q < r; q++ {
				var sum complex64
				idx := 0
				for m := 0; m < r; m++ {
					sum += f32Mul(x[m], rt[idx])
					idx += q
					if idx >= r {
						idx -= r
					}
				}
				if q > 0 && i > 0 {
					sum = f32Mul(sum, tw[(q-1)*(ido-1)+i-1])
				}
				ch[ido*(k+q*l1)+i] = sum
			}
		}
	}
}

// f32Bfly8 is pocketfft's radix-8 butterfly: a split 2·4 whose odd half is
// rotated by ∓45°/∓135° (pocketfft's ROTX45/ROTX135, direction folded into s)
// instead of multiplied by general roots. Outputs in order 0..7.
func f32Bfly8(y *[8]complex64, x0, x1, x2, x3, x4, x5, x6, x7 complex64, s float32) {
	a1, a5 := x1+x5, x1-x5
	a3, a7 := x3+x7, x3-x7
	a1, a3 = a1+a3, a1-a3
	a3 = f32RotS(a3, s)
	a7 = f32RotS(a7, s)
	a5, a7 = a5+a7, a5-a7
	// ×exp(s·iπ/4) and ×exp(s·3iπ/4).
	a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
	a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
	a0, a4 := x0+x4, x0-x4
	a2, a6 := x2+x6, x2-x6
	a0, a2 = a0+a2, a0-a2
	a6 = f32RotS(a6, s)
	a4, a6 = a4+a6, a4-a6
	y[0], y[4] = a0+a1, a0-a1
	y[2], y[6] = a2+a3, a2-a3
	y[1], y[5] = a4+a5, a4-a5
	y[3], y[7] = a6+a7, a6-a7
}

func f32Pass8(ido, l1 int, cc, ch, tw []complex64, inverse bool) {
	if ido == 1 {
		f32Pass8last(l1, cc, ch, f32DirSign(inverse))
		return
	}
	s := f32DirSign(inverse)
	m := ido - 1
	var w [8][]complex64
	for j := 1; j < 8; j++ {
		w[j] = tw[(j-1)*m:][:m]
	}
	var y [8]complex64
	for k := 0; k < l1; k++ {
		b := 8 * ido * k
		f32Bfly8(&y, cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], cc[b+5*ido], cc[b+6*ido], cc[b+7*ido], s)
		for j := 0; j < 8; j++ {
			ch[ido*(k+j*l1)] = y[j]
		}
		var x, o [8][]complex64
		for j := 0; j < 8; j++ {
			x[j] = f32Stream(cc, b+j*ido+1, m)
			o[j] = f32Stream(ch, ido*(k+j*l1)+1, m)
		}
		x0, x1, x2, x3, x4, x5, x6, x7 := x[0], x[1][:m], x[2][:m], x[3][:m], x[4][:m], x[5][:m], x[6][:m], x[7][:m]
		o0, o1, o2, o3, o4, o5, o6, o7 := o[0][:m], o[1][:m], o[2][:m], o[3][:m], o[4][:m], o[5][:m], o[6][:m], o[7][:m]
		w1, w2, w3, w4, w5, w6, w7 := w[1][:m], w[2][:m], w[3][:m], w[4][:m], w[5][:m], w[6][:m], w[7][:m]
		for i := range x0 {
			// f32Bfly8, inlined by hand: it is over the inliner's budget.
			a1, a5 := x1[i]+x5[i], x1[i]-x5[i]
			a3, a7 := x3[i]+x7[i], x3[i]-x7[i]
			a1, a3 = a1+a3, a1-a3
			a3 = f32RotS(a3, s)
			a7 = f32RotS(a7, s)
			a5, a7 = a5+a7, a5-a7
			a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
			a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
			a0, a4 := x0[i]+x4[i], x0[i]-x4[i]
			a2, a6 := x2[i]+x6[i], x2[i]-x6[i]
			a0, a2 = a0+a2, a0-a2
			a6 = f32RotS(a6, s)
			a4, a6 = a4+a6, a4-a6
			o0[i] = a0 + a1
			o4[i] = f32Mul(a0-a1, w4[i])
			o2[i] = f32Mul(a2+a3, w2[i])
			o6[i] = f32Mul(a2-a3, w6[i])
			o1[i] = f32Mul(a4+a5, w1[i])
			o5[i] = f32Mul(a4-a5, w5[i])
			o3[i] = f32Mul(a6+a7, w3[i])
			o7[i] = f32Mul(a6-a7, w7[i])
		}
	}
}

// The passXlast functions are the ido == 1 case — the final pass, where every
// point is a block of its own. There the per-block setup of the general loop
// runs once per point, so it is replaced by one flat loop over k: input block
// k is cc[r·k : r·k+r] and output j is the unit-stride f32Stream ch[j·l1 : (j+1)·l1].

func f32Pass2last(l1 int, cc, ch []complex64) {
	in := cc[:2*l1]
	o0, o1 := ch[:l1], ch[l1:][:l1]
	for k := range o0 {
		a, b := in[2*k], in[2*k+1]
		o0[k] = a + b
		o1[k] = a - b
	}
}

func f32Pass3last(l1 int, cc, ch []complex64, s float32) {
	in := cc[:3*l1]
	o0, o1, o2 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1]
	for k := range o0 {
		x := in[3*k : 3*k+3]
		o0[k], o1[k], o2[k] = f32Bfly3(x[0], x[1], x[2], s)
	}
}

func f32Pass4last(l1 int, cc, ch []complex64, s float32) {
	in := cc[:4*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	for k := range o0 {
		x := in[4*k : 4*k+4]
		o0[k], o1[k], o2[k], o3[k] = f32Bfly4(x[0], x[1], x[2], x[3], s)
	}
}

func f32Pass5last(l1 int, cc, ch []complex64, s float32) {
	in := cc[:5*l1]
	o0, o1, o2, o3, o4 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1], ch[4*l1:][:l1]
	for k := range o0 {
		x := in[5*k : 5*k+5]
		// f32Bfly5, inlined by hand: it is over the inliner's budget.
		a := x[0]
		t1, t2 := x[1]+x[4], x[1]-x[4]
		t3, t4 := x[2]+x[3], x[2]-x[3]
		r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
		r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
		i1 := f32RotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
		i2 := f32RotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
		o0[k] = a + t1 + t3
		o1[k] = r1 + i1
		o2[k] = r2 + i2
		o3[k] = r2 - i2
		o4[k] = r1 - i1
	}
}

func f32Pass8last(l1 int, cc, ch []complex64, s float32) {
	in := cc[:8*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	o4, o5, o6, o7 := ch[4*l1:][:l1], ch[5*l1:][:l1], ch[6*l1:][:l1], ch[7*l1:][:l1]
	for k := range o0 {
		x := in[8*k : 8*k+8]
		// f32Bfly8, inlined by hand: it is over the inliner's budget.
		a1, a5 := x[1]+x[5], x[1]-x[5]
		a3, a7 := x[3]+x[7], x[3]-x[7]
		a1, a3 = a1+a3, a1-a3
		a3 = f32RotS(a3, s)
		a7 = f32RotS(a7, s)
		a5, a7 = a5+a7, a5-a7
		a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
		a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
		a0, a4 := x[0]+x[4], x[0]-x[4]
		a2, a6 := x[2]+x[6], x[2]-x[6]
		a0, a2 = a0+a2, a0-a2
		a6 = f32RotS(a6, s)
		a4, a6 = a4+a6, a4-a6
		o0[k], o4[k] = a0+a1, a0-a1
		o2[k], o6[k] = a2+a3, a2-a3
		o1[k], o5[k] = a4+a5, a4-a5
		o3[k], o7[k] = a6+a7, a6-a7
	}
}

func f32Pass7last(l1 int, cc, ch []complex64, s float32) {
	in := cc[:7*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	o4, o5, o6 := ch[4*l1:][:l1], ch[5*l1:][:l1], ch[6*l1:][:l1]
	for k := range o0 {
		x := in[7*k : 7*k+7]
		// f32Bfly7, inlined by hand: it is over the inliner's budget.
		a := x[0]
		s16, d16 := x[1]+x[6], x[1]-x[6]
		s25, d25 := x[2]+x[5], x[2]-x[5]
		s34, d34 := x[3]+x[4], x[3]-x[4]
		r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
		r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
		r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
		i1 := f32RotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
		i2 := f32RotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
		i3 := f32RotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
		o0[k] = a + s16 + s25 + s34
		o1[k], o6[k] = r1+i1, r1-i1
		o2[k], o5[k] = r2+i2, r2-i2
		o3[k], o4[k] = r3+i3, r3-i3
	}
}
