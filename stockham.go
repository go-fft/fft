package fft

import "sync"

// Iterative mixed-radix (Stockham autosort) engine.
//
// The recursive engine in mixedradix.go decimates in time by recursion: every
// level gathers strided inputs and reads its twiddles at stride n/len, so the
// memory schedule degrades as the factorization deepens. This engine runs the
// same radices as a flat sequence of passes instead (the pocketfft schedule):
// pass s with radix r, l1 = product of the earlier radices and ido =
// n/(l1·r) reads cc laid out [l1][r][ido] and writes ch laid out [r][l1][ido],
// ping-ponging between dst and one pooled scratch buffer. Every inner loop runs
// over i in [0, ido) with unit stride, and each pass has its own contiguous
// twiddle block, so reads are sequential and there is no bit-reversal pass
// (the autosort layout lands the output in natural order).

// skStage is one Stockham pass: radix r applied with l1 earlier radices and
// ido remaining points. tw/twc hold W^(j·l1·i), j in [1,r), i in [1,ido), at
// index (j-1)·(ido-1)+(i-1), forward and conjugate.
type skStage struct {
	r, l1, ido int
	tw, twc    []complex128
	// rt/rtc hold the size-r roots W_r^k (forward and conjugate) for the
	// general radix pass; nil for the specialized radices.
	rt, rtc []complex128
}

// skPlan is the Stockham plan for one smooth length.
type skPlan struct {
	n       int
	stages  []skStage
	scratch sync.Pool
}

func newSKPlan(n int) *skPlan {
	root := twiddleTable(n)
	p := &skPlan{n: n}
	l1 := 1
	for _, r := range factorize(n) {
		ido := n / (l1 * r)
		st := skStage{r: r, l1: l1, ido: ido}
		if ido > 1 {
			st.tw = make([]complex128, (r-1)*(ido-1))
			st.twc = make([]complex128, (r-1)*(ido-1))
			for j := 1; j < r; j++ {
				for i := 1; i < ido; i++ {
					w := root[(j*l1*i)%n]
					st.tw[(j-1)*(ido-1)+i-1] = w
					st.twc[(j-1)*(ido-1)+i-1] = complex(real(w), -imag(w))
				}
			}
		}
		switch r {
		case 2, 3, 4, 5, 7:
		default:
			st.rt = make([]complex128, r)
			st.rtc = make([]complex128, r)
			for k := 0; k < r; k++ {
				w := root[k*(n/r)]
				st.rt[k] = w
				st.rtc[k] = complex(real(w), -imag(w))
			}
		}
		p.stages = append(p.stages, st)
		l1 *= r
	}
	p.scratch.New = func() any { b := make([]complex128, n); return &b }
	return p
}

// transform writes the unnormalized DFT of src into dst (conjugate roots when
// inverse). dst may alias src.
func (p *skPlan) transform(dst, src []complex128, inverse bool) {
	bp := p.scratch.Get().(*[]complex128)
	scr := *bp
	s := len(p.stages)
	// Pass s writes dst when (S-1-s) is even, so the last pass lands in dst.
	in := src
	if s%2 == 1 && &dst[0] == &src[0] {
		// Pass 0 would write dst while reading it: read a copy instead.
		copy(scr, src)
		in = scr
	}
	for k := range p.stages {
		out := scr
		if (s-1-k)%2 == 0 {
			out = dst
		}
		p.stages[k].pass(out, in, inverse)
		in = out
	}
	p.scratch.Put(bp)
}

func (st *skStage) pass(ch, cc []complex128, inverse bool) {
	tw := st.tw
	if inverse {
		tw = st.twc
	}
	switch st.r {
	case 2:
		pass2(st.ido, st.l1, cc, ch, tw)
	case 3:
		pass3(st.ido, st.l1, cc, ch, tw, inverse)
	case 4:
		pass4(st.ido, st.l1, cc, ch, tw, inverse)
	case 5:
		pass5(st.ido, st.l1, cc, ch, tw, inverse)
	case 7:
		pass7(st.ido, st.l1, cc, ch, tw, inverse)
	default:
		rt := st.rt
		if inverse {
			rt = st.rtc
		}
		passg(st.r, st.ido, st.l1, cc, ch, tw, rt)
	}
}

// In every pass: CC(i,m,k) = cc[i+ido·(m+r·k)], CH(i,k,m) = ch[i+ido·(k+l1·m)],
// WA(x,i) = tw[(i-1)+x·(ido-1)].

func pass2(ido, l1 int, cc, ch, tw []complex128) {
	for k := 0; k < l1; k++ {
		a := cc[ido*2*k : ido*2*k+ido]
		b := cc[ido*(2*k+1) : ido*(2*k+1)+ido]
		o0 := ch[ido*k : ido*k+ido]
		o1 := ch[ido*(k+l1) : ido*(k+l1)+ido]
		o0[0] = a[0] + b[0]
		o1[0] = a[0] - b[0]
		for i := 1; i < ido; i++ {
			o0[i] = a[i] + b[i]
			o1[i] = (a[i] - b[i]) * tw[i-1]
		}
	}
}

func pass3(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	const tw1r = -0.5
	tw1i := -sin120
	if inverse {
		tw1i = sin120
	}
	var w1, w2 []complex128
	if ido > 1 {
		w1, w2 = tw[:ido-1], tw[ido-1:2*(ido-1)]
	}
	for k := 0; k < l1; k++ {
		base := ido * 3 * k
		x0 := cc[base : base+ido]
		x1 := cc[base+ido : base+2*ido]
		x2 := cc[base+2*ido : base+3*ido]
		o0 := ch[ido*k : ido*k+ido]
		o1 := ch[ido*(k+l1) : ido*(k+l1)+ido]
		o2 := ch[ido*(k+2*l1) : ido*(k+2*l1)+ido]
		for i := 0; i < ido; i++ {
			t0 := x0[i]
			t1 := x1[i] + x2[i]
			t2 := x1[i] - x2[i]
			o0[i] = t0 + t1
			ca := complex(real(t0)+tw1r*real(t1), imag(t0)+tw1r*imag(t1))
			cb := complex(-tw1i*imag(t2), tw1i*real(t2))
			if i == 0 {
				o1[0] = ca + cb
				o2[0] = ca - cb
			} else {
				o1[i] = (ca + cb) * w1[i-1]
				o2[i] = (ca - cb) * w2[i-1]
			}
		}
	}
}

func pass4(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	var w1, w2, w3 []complex128
	if ido > 1 {
		w1, w2, w3 = tw[:ido-1], tw[ido-1:2*(ido-1)], tw[2*(ido-1):3*(ido-1)]
	}
	for k := 0; k < l1; k++ {
		base := ido * 4 * k
		x0 := cc[base : base+ido]
		x1 := cc[base+ido : base+2*ido]
		x2 := cc[base+2*ido : base+3*ido]
		x3 := cc[base+3*ido : base+4*ido]
		o0 := ch[ido*k : ido*k+ido]
		o1 := ch[ido*(k+l1) : ido*(k+l1)+ido]
		o2 := ch[ido*(k+2*l1) : ido*(k+2*l1)+ido]
		o3 := ch[ido*(k+3*l1) : ido*(k+3*l1)+ido]
		{
			t2, t1 := x0[0]+x2[0], x0[0]-x2[0]
			t3, t4 := x1[0]+x3[0], x1[0]-x3[0]
			t4 = rotNeg90(t4, inverse)
			o0[0] = t2 + t3
			o2[0] = t2 - t3
			o1[0] = t1 + t4
			o3[0] = t1 - t4
		}
		if inverse {
			for i := 1; i < ido; i++ {
				t2, t1 := x0[i]+x2[i], x0[i]-x2[i]
				t3, t4 := x1[i]+x3[i], x1[i]-x3[i]
				t4 = complex(-imag(t4), real(t4))
				o0[i] = t2 + t3
				o2[i] = (t2 - t3) * w2[i-1]
				o1[i] = (t1 + t4) * w1[i-1]
				o3[i] = (t1 - t4) * w3[i-1]
			}
		} else {
			for i := 1; i < ido; i++ {
				t2, t1 := x0[i]+x2[i], x0[i]-x2[i]
				t3, t4 := x1[i]+x3[i], x1[i]-x3[i]
				t4 = complex(imag(t4), -real(t4))
				o0[i] = t2 + t3
				o2[i] = (t2 - t3) * w2[i-1]
				o1[i] = (t1 + t4) * w1[i-1]
				o3[i] = (t1 - t4) * w3[i-1]
			}
		}
	}
}

func pass5(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	const (
		c1 = 0.30901699437494742410229341718281905886015458990288
		s1 = 0.95105651629515357211643933337938214340569863412575
		c2 = -0.80901699437494742410229341718281905886015458990289
		s2 = 0.58778525229247312916870595463907276859765243764314
	)
	sg := -1.0
	if inverse {
		sg = 1.0
	}
	ss1, ss2 := sg*s1, sg*s2
	var w [4][]complex128
	if ido > 1 {
		for j := 0; j < 4; j++ {
			w[j] = tw[j*(ido-1) : (j+1)*(ido-1)]
		}
	}
	for k := 0; k < l1; k++ {
		base := ido * 5 * k
		x0 := cc[base : base+ido]
		x1 := cc[base+ido : base+2*ido]
		x2 := cc[base+2*ido : base+3*ido]
		x3 := cc[base+3*ido : base+4*ido]
		x4 := cc[base+4*ido : base+5*ido]
		o0 := ch[ido*k : ido*k+ido]
		o1 := ch[ido*(k+l1) : ido*(k+l1)+ido]
		o2 := ch[ido*(k+2*l1) : ido*(k+2*l1)+ido]
		o3 := ch[ido*(k+3*l1) : ido*(k+3*l1)+ido]
		o4 := ch[ido*(k+4*l1) : ido*(k+4*l1)+ido]
		for i := 0; i < ido; i++ {
			a := x0[i]
			t1 := x1[i] + x4[i]
			t2 := x1[i] - x4[i]
			t3 := x2[i] + x3[i]
			t4 := x2[i] - x3[i]
			o0[i] = a + t1 + t3
			r1 := a + complex(c1*real(t1)+c2*real(t3), c1*imag(t1)+c2*imag(t3))
			r2 := a + complex(c2*real(t1)+c1*real(t3), c2*imag(t1)+c1*imag(t3))
			// i·(ss1·t2 + ss2·t4) and i·(ss2·t2 − ss1·t4)
			i1 := complex(-(ss1*imag(t2) + ss2*imag(t4)), ss1*real(t2)+ss2*real(t4))
			i2 := complex(-(ss2*imag(t2) - ss1*imag(t4)), ss2*real(t2)-ss1*real(t4))
			if i == 0 {
				o1[0] = r1 + i1
				o4[0] = r1 - i1
				o2[0] = r2 + i2
				o3[0] = r2 - i2
			} else {
				o1[i] = (r1 + i1) * w[0][i-1]
				o4[i] = (r1 - i1) * w[3][i-1]
				o2[i] = (r2 + i2) * w[1][i-1]
				o3[i] = (r2 - i2) * w[2][i-1]
			}
		}
	}
}

func pass7(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	const (
		c1 = 0.6234898018587335305250048840042398106322747308237
		s1 = 0.7818314824680298087084445266740577502323345187493
		c2 = -0.2225209339563144042889025644967948758319743752712
		s2 = 0.9749279121818236070181316829939312172327858006199
		c3 = -0.9009688679024191262361023195074450511659191621318
		s3 = 0.4338837391175581204757683328483587546099907277859
	)
	sg := -1.0
	if inverse {
		sg = 1.0
	}
	var w [6][]complex128
	if ido > 1 {
		for j := 0; j < 6; j++ {
			w[j] = tw[j*(ido-1) : (j+1)*(ido-1)]
		}
	}
	for k := 0; k < l1; k++ {
		base := ido * 7 * k
		for i := 0; i < ido; i++ {
			a := cc[base+i]
			b := cc[base+ido+i]
			c := cc[base+2*ido+i]
			d := cc[base+3*ido+i]
			e := cc[base+4*ido+i]
			f := cc[base+5*ido+i]
			g := cc[base+6*ido+i]
			s16, d16 := b+g, b-g
			s25, d25 := c+f, c-f
			s34, d34 := d+e, d-e
			var y [7]complex128
			y[0] = a + s16 + s25 + s34
			r1 := a + complex(c1*real(s16)+c2*real(s25)+c3*real(s34), c1*imag(s16)+c2*imag(s25)+c3*imag(s34))
			r2 := a + complex(c2*real(s16)+c3*real(s25)+c1*real(s34), c2*imag(s16)+c3*imag(s25)+c1*imag(s34))
			r3 := a + complex(c3*real(s16)+c1*real(s25)+c2*real(s34), c3*imag(s16)+c1*imag(s25)+c2*imag(s34))
			i1 := rotI(complex(sg*(s1*real(d16)+s2*real(d25)+s3*real(d34)), sg*(s1*imag(d16)+s2*imag(d25)+s3*imag(d34))))
			i2 := rotI(complex(sg*(s2*real(d16)-s3*real(d25)-s1*real(d34)), sg*(s2*imag(d16)-s3*imag(d25)-s1*imag(d34))))
			i3 := rotI(complex(sg*(s3*real(d16)-s1*real(d25)+s2*real(d34)), sg*(s3*imag(d16)-s1*imag(d25)+s2*imag(d34))))
			y[1], y[6] = r1+i1, r1-i1
			y[2], y[5] = r2+i2, r2-i2
			y[3], y[4] = r3+i3, r3-i3
			ch[ido*k+i] = y[0]
			for j := 1; j < 7; j++ {
				v := y[j]
				if i > 0 {
					v *= w[j-1][i-1]
				}
				ch[ido*(k+j*l1)+i] = v
			}
		}
	}
}

// passg is the general radix-r pass (r an odd prime without a specialized
// butterfly): an O(r²) size-r DFT per point, then the twiddle.
func passg(r, ido, l1 int, cc, ch, tw, rt []complex128) {
	var bufArr [maxRadix]complex128
	x := bufArr[:r]
	for k := 0; k < l1; k++ {
		base := ido * r * k
		for i := 0; i < ido; i++ {
			for m := 0; m < r; m++ {
				x[m] = cc[base+m*ido+i]
			}
			for q := 0; q < r; q++ {
				var sum complex128
				idx := 0
				for m := 0; m < r; m++ {
					sum += x[m] * rt[idx]
					idx += q
					if idx >= r {
						idx -= r
					}
				}
				if q > 0 && i > 0 {
					sum *= tw[(q-1)*(ido-1)+i-1]
				}
				ch[ido*(k+q*l1)+i] = sum
			}
		}
	}
}
