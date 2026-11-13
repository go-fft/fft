package fft

// floatT is the element type of the data a float64-arithmetic transform reads
// or writes: float64 for the plain methods, float32 for the Mixed ones, which
// convert while reading and round once while writing.
type floatT interface{ ~float32 | ~float64 }

// mdctForward is MDCTPlan.MDCT and MDCTPlan.MDCTMixed: the arithmetic is float64
// whatever S and D are, so MDCTMixed's result is float32 of MDCT's on the same
// values, exactly.
func mdctForward[S, D floatT](p *MDCTPlan, dst []D, src []S, scale float64) []D {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < m || len(src) < n {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for i := 0; i < n; i++ {
			z[i] = complex(float64(src[i]), 0) * p.pre[i]
		}
		p.fft.FFT(z, z)
		for k := 0; k < m; k++ {
			a, b := p.post[k], z[k]
			dst[k] = D(scale * (real(a)*real(b) - imag(a)*imag(b)))
		}
		p.pool.put(sp)
		return dst[:m]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	// The fold (a, b, c, d) → (−c_r − d, a − b_r) of the article is read in place:
	// the DCT-IV's pre-twiddle pairs the folded values u[2j] and u[m-1-2j]; for
	// j < j1 = ceil(h/2) the first lies in the first half of u (from c and d) and
	// the second in the other (from a and b), for j >= j1 the reverse.
	j1 := (h + 1) / 2
	src = src[:n]
	pre := k.pre[:h]
	for j := 0; j < j1; j++ {
		ua := -(float64(src[m+h-1-2*j]) + float64(src[m+h+2*j]))
		ub := float64(src[h-1-2*j]) - float64(src[h+2*j])
		c[j] = complex(ua, ub) * pre[j]
	}
	for j := j1; j < h; j++ {
		ua := float64(src[2*j-h]) - float64(src[3*h-1-2*j])
		ub := -(float64(src[h+2*j]) + float64(src[5*h-1-2*j]))
		c[j] = complex(ua, ub) * pre[j]
	}
	k.half.FFT(c, c)
	f := scale / 2 // the package's DCT-IV is twice the sum
	dst = dst[:m]
	post := k.post[:h]
	for i := 0; i < h; i++ {
		w := post[i] * c[i]
		dst[2*i] = D(real(w) * f)
		dst[m-1-2*i] = D(-imag(w) * f)
	}
	p.pool.put(sp)
	return dst
}

// mdctInverse is MDCTPlan.IMDCT and MDCTPlan.IMDCTMixed.
func mdctInverse[S, D floatT](p *MDCTPlan, dst []D, src []S, scale float64) []D {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < n || len(src) < m {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for k := 0; k < m; k++ {
			z[k] = complex(float64(src[k]), 0) * p.post[k]
		}
		clear(z[m:])
		p.fft.FFT(z, z)
		for i := 0; i < n; i++ {
			a, b := p.pre[i], z[i]
			dst[i] = D(scale * (real(a)*real(b) - imag(a)*imag(b)))
		}
		p.pool.put(sp)
		return dst[:n]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	src = src[:m]
	pre := k.pre[:h]
	for j := 0; j < h; j++ {
		c[j] = complex(float64(src[2*j]), float64(src[m-1-2*j])) * pre[j]
	}
	k.half.FFT(c, c)
	// The DCT-IV output u[k] is the unfolded (u, v) → (v, −v_r, −u_r, −u) of the
	// article, written where it belongs: u[k] with k < h goes to dst[m+h-1-k] and
	// dst[m+h+k] negated, u[h+j] to dst[j] and, negated, dst[m-1-j]. The outputs
	// k = 2i and k = m-1-2i of one post-twiddle fall in the first half for
	// i < j1 and the second half for the other, and the reverse.
	f := scale / 2
	dst = dst[:n]
	post := k.post[:h]
	j1 := (h + 1) / 2
	for i := 0; i < j1; i++ {
		w := post[i] * c[i]
		s := D(real(w) * f)  // u[2i], 2i < h
		t := D(-imag(w) * f) // u[m-1-2i], in the second half: index j = h-1-2i
		dst[m+h-1-2*i] = -s
		dst[m+h+2*i] = -s
		dst[h-1-2*i] = t
		dst[m-h+2*i] = -t
	}
	for i := j1; i < h; i++ {
		w := post[i] * c[i]
		t := D(real(w) * f)  // u[2i], 2i >= h: j = 2i-h
		s := D(-imag(w) * f) // u[m-1-2i], m-1-2i < h
		dst[2*i-h] = t
		dst[m-1-2*i+h] = -t
		dst[m+h-1-(m-1-2*i)] = -s
		dst[m+h+(m-1-2*i)] = -s
	}
	p.pool.put(sp)
	return dst
}

// MDCTMixed is MDCT on float32 data with the arithmetic of the float64 plan:
// src is converted to float64 while the fold and the pre-twiddle read it, the
// transform runs in float64, and each coefficient is rounded to float32 once, as
// the post-twiddle writes it, with no pass of its own and no allocation. The
// result is exactly float32 of what MDCT returns for the same values converted
// to float64, so its error against the exact transform is the float64 bound plus
// one rounding to float32 (relative RMS: c·ε64·log2 N + ε32/2, see
// docs/accuracy.md). MDCTPlan32 computes in float32 and is less accurate: use
// this when the error budget is a float32 rounding, not log2 N of them.
//
// scale is float64: it multiplies in float64 before the rounding.
func (p *MDCTPlan) MDCTMixed(dst, src []float32, scale float64) []float32 {
	return mdctForward(p, dst, src, scale)
}

// IMDCTMixed is IMDCT on float32 data with the arithmetic of the float64 plan;
// see MDCTMixed.
func (p *MDCTPlan) IMDCTMixed(dst, src []float32, scale float64) []float32 {
	return mdctInverse(p, dst, src, scale)
}
