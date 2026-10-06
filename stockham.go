package fft

import (
	"slices"
	"sync"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

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
	// twX/twXc are tw/twc extended to i = 0 ((r-1) blocks of ido entries, the
	// i = 0 entry being 1), the layout the SIMD pass kernels read; nil when
	// no pass kernel runs this pass on this architecture.
	twX, twXc []complex128
	// wide lets the pass use the AVX-512 kernels (see wide512).
	wide bool
	// split is the pass's data layout: on arm64 and on amd64 with AVX2, a
	// run of passes keeps its data block-split between them
	// (kernels.StockhamSplitModes); 0 is interleaved.
	split uint8
}

// skPlan is the Stockham plan for one smooth length.
type skPlan struct {
	n       int
	stages  []skStage
	scratch sync.Pool
	// cascT and cascB are the blocked schedule's shape (cascade.go): passes
	// over the whole array, then groups of cascB blocks; 0 runs breadth first.
	cascT, cascB int
	// cascPair runs passes 0 and 1 as one sweep (cascadePair).
	cascPair bool
}

func newSKPlan(n int) *skPlan { return newSKPlanFactors(n, skFactorize(n)) }

// newSKPlanFactors builds the plan for the given ordered radices, whose
// product must be n.
func newSKPlanFactors(n int, factors []int) *skPlan {
	root := twiddleTable(n)
	p := &skPlan{n: n}
	l1 := 1
	for _, r := range factors {
		ido := n / (l1 * r)
		st := skStage{r: r, l1: l1, ido: ido, wide: wide512(n)}
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
		st.twX, st.twXc = kernels.StockhamTwiddles(r, ido, l1, root)
		switch r {
		case 2, 3, 4, 5, 7, 8, 12, 16:
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
	rs, idos, l1s := make([]int, len(p.stages)), make([]int, len(p.stages)), make([]int, len(p.stages))
	for k, st := range p.stages {
		rs[k], idos[k], l1s[k] = st.r, st.ido, st.l1
	}
	for k, m := range kernels.StockhamSplitModes(rs, idos, l1s) {
		p.stages[k].split = m
	}
	size := n
	p.cascT, p.cascB = cascadeShape(n, p.stages, cascadeMin)
	p.cascPair = cascadePairs(p.cascT, p.stages)
	switch {
	case p.cascT > 0:
		size = cascadeScratch(n)
		// The groups and the final runs read and write interleaved data,
		// so the whole schedule does (arm64 keeps powers of two split).
		for k := range p.stages {
			p.stages[k].split = 0
		}
	case takesGap(n):
		size += setSpan // room to slide the window off dst's sets
	case len(p.stages)%2 == 1:
		size *= 2 // the in-place ping-pong (see run)
	}
	// A split pass reads its twiddles in the split layout's order.
	for k := range p.stages {
		if st := &p.stages[k]; st.split != 0 {
			st.twX, st.twXc = kernels.StockhamSplitTwiddles(st.r, st.ido, st.l1, root)
		}
	}
	p.scratch.New = func() any { b := make([]complex128, size); return &b }
	return p
}

// skFactorize orders the radices for the passes. The powers of two come first
// as radix-8 passes where that pays, then radix 4, with no lone radix-2 pass
// unless the length has a single factor 2; the odd primes follow ascending, so
// the expensive odd butterflies run on the short, late passes.
//
// pocketfft (cfftp::factorize) takes every 8 it can. Measured here, radix 8
// wins while the working set is small (512: 1.30×, 2048: 1.17× over radix 4)
// but loses on large pure powers of two (65536: 0.71–0.78×), where its eight
// input and eight output streams sit at power-of-two strides and alias in the
// cache. So a large power of two stays on radix 4; a length with an odd factor
// has no such aliasing and keeps radix 8 (20160: 1.07×).
//
// Where the amd64 radix-16 kernel runs, a few powers of two take radix-16
// passes instead (radix16Table, Round 19), and where the amd64 split layout
// runs, the powers of two from 256 points take splitTable's factorizations
// (Round 23).
//
// Some composites end with radix-12 or radix-16 passes (compFactorize, Round
// 24) where the amd64 kernels run them, and the
// complex128 order is compOddFirst, which may differ from the float32 plans'.
func skFactorize(n int) []int {
	if f := splitTable[n]; f != nil {
		return slices.Clone(f)
	}
	if f := radix16Table[n]; f != nil {
		return slices.Clone(f)
	}
	return compFactorize(n)
}

// compOddFirst selects skFactorize's radix order for complex128 plans; it is
// per-architecture (comp_route_*.go).
var compOddFirst = compOddFirstDefault()

// compTakes16 reports whether skFactorize gives n radix-16 or radix-12
// passes, which have no batched kernel for the strips of an N-D plan.
func compTakes16(n int) bool {
	return slices.ContainsFunc(skFactorize(n), func(r int) bool { return r == 12 || r == 16 })
}

// skFactorizeOrder is skFactorize with the order chosen: with oddFirst, the
// radices 3 and 5 come first, then the powers of two in reverse (radix 4
// before radix 8), so the final pass, which has no twiddles, is a radix-8
// pass where there is one; 7, 11 and 13, which have no SIMD pass, stay last.
func skFactorizeOrder(n int, oddFirst bool) []int {
	e, odd := 0, n
	for odd%2 == 0 {
		odd /= 2
		e++
	}
	var f []int
	if n <= r8MaxPow2 || odd > 1 {
		f = radix8Maximal(e)
	} else {
		f = pow2Radices(e, n <= pow2OneRadix8Max)
	}
	var o []int
	for _, prime := range []int{3, 5, 7, 11, 13} {
		for odd%prime == 0 {
			o = append(o, prime)
			odd /= prime
		}
	}
	if !oddFirst {
		return append(f, o...)
	}
	k := 0
	for k < len(o) && o[k] <= 5 {
		k++
	}
	slices.Reverse(f)
	return append(append(o[:k:k], f...), o[k:]...)
}

// oddRadicesFirst selects skFactorizeOrder's order; it is per-architecture
// (route_*.go).
var oddRadicesFirst = oddRadicesFirstDefault()

// radix8Maximal factors 2^e into as many radix-8 passes as it can, then radix
// 4, taking two radix-4 passes rather than one radix-2 pass when that fits.
func radix8Maximal(e int) []int {
	var f []int
	n8 := e / 3
	switch e % 3 {
	case 1:
		if e >= 4 {
			// 2^(3k+1) = 8^(k-1)·4·4: two radix-4 passes beat a radix-2 pass.
			n8--
			f = append(f, 4, 4)
		} else {
			f = append(f, 2)
		}
	case 2:
		f = append(f, 4)
	}
	for ; n8 > 0; n8-- {
		f = append([]int{8}, f...)
	}
	return f
}

// pow2Radices factors a pure power of two 2^e the way a large one is factored
// (radix 4), finishing an odd exponent either with one radix-2 pass or, when
// oneRadix8 is set, by opening with one radix-8 pass instead.
func pow2Radices(e int, oneRadix8 bool) []int {
	var f []int
	if oneRadix8 && e%2 == 1 && e >= 3 {
		f = append(f, 8)
		e -= 3
	}
	for ; e >= 2; e -= 2 {
		f = append(f, 4)
	}
	if e == 1 {
		f = append(f, 2)
	}
	return f
}

// r8MaxPow2 is the largest pure power of two factored with radix-8 passes;
// it is per-architecture (route_*.go).
var r8MaxPow2 = r8MaxPow2Default()

// wide512 reports whether a transform of length n may use the AVX-512 pass
// kernels: a power of two of at least 256 points. Measured on Cascade Lake
// (AVX2 time ÷ AVX-512 time, Stockham, 2026-10-01): composites 0.88–1.04
// (see kernels.StockhamPass), powers of two 8–128 0.94–1.03, 256–2048
// 1.19–1.34, 4096 1.49.
func wide512(n int) bool { return n&(n-1) == 0 && n >= 256 }

// transform writes the unnormalized DFT of src into dst (conjugate roots when
// inverse). dst may alias src.
//
// The common case, every pass count out of place and an even one in place,
// runs here rather than through run: the call cost about 7 ns (5%) at 64
// points on Cascade Lake, and inlining it measured 1.01–1.05× over the
// previous release at 64–256 points there (Round 17).
func (p *skPlan) transform(dst, src []complex128, inverse bool) {
	bp := p.scratch.Get().(*[]complex128)
	s := len(p.stages)
	if p.cascT > 0 {
		p.cascadeRun(dst, src, *bp, inverse)
		p.scratch.Put(bp)
		return
	}
	if s%2 == 1 && &dst[0] == &src[0] {
		p.run(dst, src, *bp, inverse)
		p.scratch.Put(bp)
		return
	}
	scr := (*bp)[:p.n]
	if takesGap(p.n) {
		scr = offTheSets(*bp, dst, p.n)
	}
	in := src
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

// run is transform with the scratch buffer given: one from p.scratch.
//
// Pass k writes dst when s-1-k is even, so the last pass lands in dst. With
// an odd pass count that makes pass 0 write dst, which it reads when the
// transform is in place. The scratch buffer then holds two n-point halves
// and the passes before the last alternate between them, except for the
// lengths that place their scratch off dst's sets (takesGap): those copy src
// to the scratch buffer and read it from there. In a 2-D transform every row
// is transformed in place, so the copy was one more pass over the matrix: at
// 128×128 on Zen 3, copying 16384 values alone took 6 µs against 96 µs for
// the whole transform (2026-10-05).
func (p *skPlan) run(dst, src, buf []complex128, inverse bool) {
	if p.cascT > 0 {
		p.cascadeRun(dst, src, buf, inverse)
		return
	}
	n, s := p.n, len(p.stages)
	gap := takesGap(n)
	scr := buf[:n]
	if gap {
		scr = offTheSets(buf, dst, n)
	}
	in, even := src, dst // even: where the passes with s-1-k even write
	if s%2 == 1 && &dst[0] == &src[0] {
		if gap {
			copy(scr, src)
			in = scr
		} else {
			even = buf[n : 2*n]
		}
	}
	for k := range p.stages {
		out := scr
		if k == s-1 {
			out = dst
		} else if (s-1-k)%2 == 0 {
			out = even
		}
		p.stages[k].pass(out, in, inverse)
		in = out
	}
}

// Every pass writes its r output streams n/r points apart, and the first pass
// reads its r inputs as far apart. For a power of two that distance is a
// multiple of 4 KB, so all of a pass's output streams fall in one L1 set, and
// its input streams in one set too. When dst and the scratch buffer also sit
// a multiple of 4 KB apart — the usual case, since Go places large
// allocations on page boundaries — the two groups land in the SAME set: 2r
// lines for an 8-way (Zen 3, Intel) or 4-way (Neoverse-N1) L1, which then
// evicts lines still in use.
//
// For a power of two, the scratch buffer is therefore allocated setSpan
// points longer and sliced so that it starts setGap bytes past dst, modulo
// 4 KB. Measured through the transform (2026-10-04, ns per point, dst and
// scratch 4 KB-aligned ÷ setGap apart): Zen 3 1.34 at 1024, 1.49 at 2048,
// 1.65 at 4096, 1.21 at 16384, 1.23 at 65536; Haswell 1.03–1.04 and 1.04 at
// 65536; Neoverse-N1 1.00 up to 4096 (its 16 KB ways made the conflict a
// matter of luck), 1.06 at 8192 and 16384, 1.20 at 65536; Cascade Lake within
// ±2%. Of the gaps tried (256 to 3136 bytes) only 576 lost nowhere: 2112 cost
// 8% at 1024 on both Intel CPUs and gave up Zen 3's gain there, and a 64-byte
// gap was slower than none on Zen 3. Lengths 2^k·3 (6144, 12288) LOST 2–5%
// with any gap on all four CPUs; 1920 and 3840 did not move and 1000 gained
// 3–6% on Intel only. Without a rule that separates those, only powers of
// two take the gap, and only from 1024 points, where a first pass's streams
// are at least 4 KB apart: below it the gap bought nothing anywhere and cost
// 2–3% at 64 and 128 points on Haswell and Cascade Lake.
const (
	setSpan = 4096 / 16 // one 4 KB set period, in complex128
	setGap  = 576       // bytes
)

// takesGap reports whether a transform of length n slides its scratch off
// dst's sets: powers of two from 1024 points.
func takesGap(n int) bool { return n&(n-1) == 0 && n >= 1024 }

// offTheSets returns the n-point window of buf (n+setSpan long) that starts
// setGap bytes past dst modulo 4 KB.
func offTheSets(buf, dst []complex128, n int) []complex128 {
	d := uintptr(unsafe.Pointer(unsafe.SliceData(buf))) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
	shift := int((setGap-d)&4095) / 16
	return buf[shift : shift+n]
}

func (st *skStage) pass(ch, cc []complex128, inverse bool) {
	twX := st.twX
	if inverse {
		twX = st.twXc
	}
	// An interleaved pass calls StockhamPass itself: StockhamPassLayout does
	// not inline, and the extra call cost composites about 1% on Zen 3.
	var ok bool
	if st.split == 0 {
		ok = kernels.StockhamPass(st.r, st.ido, st.l1, cc, ch, twX, inverse, st.wide)
	} else {
		ok = kernels.StockhamPassLayout(st.split, st.r, st.ido, st.l1, cc, ch, twX, inverse, st.wide)
	}
	if !ok {
		st.passScalar(ch, cc, inverse)
	}
}

// passScalar is the pure-Go pass, for every radix and ido.
func (st *skStage) passScalar(ch, cc []complex128, inverse bool) {
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
	case 8:
		pass8(st.ido, st.l1, cc, ch, tw, inverse)
	case 16:
		pass16(st.ido, st.l1, cc, ch, tw, inverse)
	case 12:
		pass12(st.ido, st.l1, cc, ch, tw, inverse)
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
//
// Every pass has the same shape: point i = 0 carries no twiddle and is done
// once per k; the remaining m = ido-1 points run a loop in which every stream
// has been resliced to exactly length m, so the compiler proves every index in
// range and the loop carries no bounds check and no per-point branch. The
// direction enters as a sign s = -1 (forward) or +1 (inverse) folded into the
// rotations, which is exact, instead of a branch per point.

// dirSign is -1 for the forward transform and +1 for the inverse.
func dirSign(inverse bool) float64 {
	if inverse {
		return 1
	}
	return -1
}

// rotS multiplies z by s·i: by -i forward, +i inverse.
func rotS(z complex128, s float64) complex128 {
	return complex(-s*imag(z), s*real(z))
}

// stream returns the length-m run of buf starting at off.
func stream(buf []complex128, off, m int) []complex128 { return buf[off:][:m] }

func pass2(ido, l1 int, cc, ch, tw []complex128) {
	if ido == 1 {
		pass2last(l1, cc, ch)
		return
	}
	m := ido - 1
	w := tw[:m]
	for k := 0; k < l1; k++ {
		a, b := cc[2*ido*k], cc[2*ido*k+ido]
		ch[ido*k] = a + b
		ch[ido*(k+l1)] = a - b
		x0, x1 := stream(cc, 2*ido*k+1, m), stream(cc, 2*ido*k+ido+1, m)
		o0, o1 := stream(ch, ido*k+1, m), stream(ch, ido*(k+l1)+1, m)
		for i := range w {
			a, b := x0[i], x1[i]
			o0[i] = a + b
			o1[i] = (a - b) * w[i]
		}
	}
}

// bfly3 is the size-3 DFT; s is the direction sign.
func bfly3(x0, x1, x2 complex128, s float64) (y0, y1, y2 complex128) {
	t1, t2 := x1+x2, x1-x2
	ca := complex(real(x0)-0.5*real(t1), imag(x0)-0.5*imag(t1))
	cb := rotS(complex(sin120*real(t2), sin120*imag(t2)), s)
	return x0 + t1, ca + cb, ca - cb
}

func pass3(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass3last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	w1, w2 := tw[:m], tw[m:][:m]
	for k := 0; k < l1; k++ {
		b := 3 * ido * k
		y0, y1, y2 := bfly3(cc[b], cc[b+ido], cc[b+2*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)] = y0, y1, y2
		x0, x1, x2 := stream(cc, b+1, m), stream(cc, b+ido+1, m), stream(cc, b+2*ido+1, m)
		o0, o1, o2 := stream(ch, ido*k+1, m), stream(ch, ido*(k+l1)+1, m), stream(ch, ido*(k+2*l1)+1, m)
		for i := range w1 {
			y0, y1, y2 := bfly3(x0[i], x1[i], x2[i], s)
			o0[i] = y0
			o1[i] = y1 * w1[i]
			o2[i] = y2 * w2[i]
		}
	}
}

// bfly4 is the size-4 DFT in output order 0,1,2,3.
func bfly4(x0, x1, x2, x3 complex128, s float64) (y0, y1, y2, y3 complex128) {
	t2, t1 := x0+x2, x0-x2
	t3, t4 := x1+x3, x1-x3
	t4 = rotS(t4, s)
	return t2 + t3, t1 + t4, t2 - t3, t1 - t4
}

func pass4(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass4last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	w1, w2, w3 := tw[:m], tw[m:][:m], tw[2*m:][:m]
	for k := 0; k < l1; k++ {
		b := 4 * ido * k
		y0, y1, y2, y3 := bfly4(cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)], ch[ido*(k+3*l1)] = y0, y1, y2, y3
		x0, x1 := stream(cc, b+1, m), stream(cc, b+ido+1, m)
		x2, x3 := stream(cc, b+2*ido+1, m), stream(cc, b+3*ido+1, m)
		o0, o1 := stream(ch, ido*k+1, m), stream(ch, ido*(k+l1)+1, m)
		o2, o3 := stream(ch, ido*(k+2*l1)+1, m), stream(ch, ido*(k+3*l1)+1, m)
		for i := range w1 {
			y0, y1, y2, y3 := bfly4(x0[i], x1[i], x2[i], x3[i], s)
			o0[i] = y0
			o1[i] = y1 * w1[i]
			o2[i] = y2 * w2[i]
			o3[i] = y3 * w3[i]
		}
	}
}

// Radix-5 constants: cos/sin of 2π/5 and 4π/5.
const (
	c51 = 0.30901699437494742410229341718281905886015458990288
	s51 = 0.95105651629515357211643933337938214340569863412575
	c52 = -0.80901699437494742410229341718281905886015458990289
	s52 = 0.58778525229247312916870595463907276859765243764314
)

// bfly5 is the size-5 DFT in output order 0..4.
func bfly5(a, x1, x2, x3, x4 complex128, s float64) (y0, y1, y2, y3, y4 complex128) {
	t1, t2 := x1+x4, x1-x4
	t3, t4 := x2+x3, x2-x3
	r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
	r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
	i1 := rotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
	i2 := rotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
	return a + t1 + t3, r1 + i1, r2 + i2, r2 - i2, r1 - i1
}

func pass5(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass5last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	w1, w2, w3, w4 := tw[:m], tw[m:][:m], tw[2*m:][:m], tw[3*m:][:m]
	for k := 0; k < l1; k++ {
		b := 5 * ido * k
		y0, y1, y2, y3, y4 := bfly5(cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], s)
		ch[ido*k], ch[ido*(k+l1)], ch[ido*(k+2*l1)] = y0, y1, y2
		ch[ido*(k+3*l1)], ch[ido*(k+4*l1)] = y3, y4
		x0, x1, x2 := stream(cc, b+1, m), stream(cc, b+ido+1, m), stream(cc, b+2*ido+1, m)
		x3, x4 := stream(cc, b+3*ido+1, m), stream(cc, b+4*ido+1, m)
		o0, o1, o2 := stream(ch, ido*k+1, m), stream(ch, ido*(k+l1)+1, m), stream(ch, ido*(k+2*l1)+1, m)
		o3, o4 := stream(ch, ido*(k+3*l1)+1, m), stream(ch, ido*(k+4*l1)+1, m)
		for i := range w1 {
			// bfly5, inlined by hand: it is over the inliner's budget.
			a := x0[i]
			t1, t2 := x1[i]+x4[i], x1[i]-x4[i]
			t3, t4 := x2[i]+x3[i], x2[i]-x3[i]
			r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
			r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
			i1 := rotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
			i2 := rotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
			o0[i] = a + t1 + t3
			o1[i] = (r1 + i1) * w1[i]
			o2[i] = (r2 + i2) * w2[i]
			o3[i] = (r2 - i2) * w3[i]
			o4[i] = (r1 - i1) * w4[i]
		}
	}
}

// Radix-7 constants: cos/sin of 2πk/7, k = 1..3.
const (
	c71 = 0.6234898018587335305250048840042398106322747308237
	s71 = 0.7818314824680298087084445266740577502323345187493
	c72 = -0.2225209339563144042889025644967948758319743752712
	s72 = 0.9749279121818236070181316829939312172327858006199
	c73 = -0.9009688679024191262361023195074450511659191621318
	s73 = 0.4338837391175581204757683328483587546099907277859
)

// bfly7 is the size-7 DFT, written into y (output order 0..6).
func bfly7(y *[7]complex128, a, b, c, d, e, f, g complex128, s float64) {
	s16, d16 := b+g, b-g
	s25, d25 := c+f, c-f
	s34, d34 := d+e, d-e
	y[0] = a + s16 + s25 + s34
	r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
	r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
	r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
	i1 := rotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
	i2 := rotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
	i3 := rotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
	y[1], y[6] = r1+i1, r1-i1
	y[2], y[5] = r2+i2, r2-i2
	y[3], y[4] = r3+i3, r3-i3
}

func pass7(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass7last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	var w [7][]complex128
	for j := 1; j < 7; j++ {
		w[j] = tw[(j-1)*m:][:m]
	}
	var y [7]complex128
	for k := 0; k < l1; k++ {
		b := 7 * ido * k
		bfly7(&y, cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], cc[b+5*ido], cc[b+6*ido], s)
		for j := 0; j < 7; j++ {
			ch[ido*(k+j*l1)] = y[j]
		}
		var x, o [7][]complex128
		for j := 0; j < 7; j++ {
			x[j] = stream(cc, b+j*ido+1, m)
			o[j] = stream(ch, ido*(k+j*l1)+1, m)
		}
		x0, x1, x2, x3, x4, x5, x6 := x[0], x[1][:m], x[2][:m], x[3][:m], x[4][:m], x[5][:m], x[6][:m]
		o0, o1, o2, o3, o4, o5, o6 := o[0][:m], o[1][:m], o[2][:m], o[3][:m], o[4][:m], o[5][:m], o[6][:m]
		w1, w2, w3, w4, w5, w6 := w[1][:m], w[2][:m], w[3][:m], w[4][:m], w[5][:m], w[6][:m]
		for i := range x0 {
			// bfly7, inlined by hand: it is over the inliner's budget.
			a := x0[i]
			s16, d16 := x1[i]+x6[i], x1[i]-x6[i]
			s25, d25 := x2[i]+x5[i], x2[i]-x5[i]
			s34, d34 := x3[i]+x4[i], x3[i]-x4[i]
			r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
			r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
			r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
			i1 := rotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
			i2 := rotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
			i3 := rotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
			o0[i] = a + s16 + s25 + s34
			o1[i] = (r1 + i1) * w1[i]
			o6[i] = (r1 - i1) * w6[i]
			o2[i] = (r2 + i2) * w2[i]
			o5[i] = (r2 - i2) * w5[i]
			o3[i] = (r3 + i3) * w3[i]
			o4[i] = (r3 - i3) * w4[i]
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

// hsqt2 = √2/2, the magnitude of the ±45° and ±135° rotations radix 8 needs.
const hsqt2 = 0.707106781186547524400844362104849

// bfly8 is pocketfft's radix-8 butterfly: a split 2·4 whose odd half is
// rotated by ∓45°/∓135° (pocketfft's ROTX45/ROTX135, direction folded into s)
// instead of multiplied by general roots. Outputs in order 0..7.
func bfly8(y *[8]complex128, x0, x1, x2, x3, x4, x5, x6, x7 complex128, s float64) {
	a1, a5 := x1+x5, x1-x5
	a3, a7 := x3+x7, x3-x7
	a1, a3 = a1+a3, a1-a3
	a3 = rotS(a3, s)
	a7 = rotS(a7, s)
	a5, a7 = a5+a7, a5-a7
	// ×exp(s·iπ/4) and ×exp(s·3iπ/4).
	a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
	a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
	a0, a4 := x0+x4, x0-x4
	a2, a6 := x2+x6, x2-x6
	a0, a2 = a0+a2, a0-a2
	a6 = rotS(a6, s)
	a4, a6 = a4+a6, a4-a6
	y[0], y[4] = a0+a1, a0-a1
	y[2], y[6] = a2+a3, a2-a3
	y[1], y[5] = a4+a5, a4-a5
	y[3], y[7] = a6+a7, a6-a7
}

func pass8(ido, l1 int, cc, ch, tw []complex128, inverse bool) {
	if ido == 1 {
		pass8last(l1, cc, ch, dirSign(inverse))
		return
	}
	s := dirSign(inverse)
	m := ido - 1
	var w [8][]complex128
	for j := 1; j < 8; j++ {
		w[j] = tw[(j-1)*m:][:m]
	}
	var y [8]complex128
	for k := 0; k < l1; k++ {
		b := 8 * ido * k
		bfly8(&y, cc[b], cc[b+ido], cc[b+2*ido], cc[b+3*ido], cc[b+4*ido], cc[b+5*ido], cc[b+6*ido], cc[b+7*ido], s)
		for j := 0; j < 8; j++ {
			ch[ido*(k+j*l1)] = y[j]
		}
		var x, o [8][]complex128
		for j := 0; j < 8; j++ {
			x[j] = stream(cc, b+j*ido+1, m)
			o[j] = stream(ch, ido*(k+j*l1)+1, m)
		}
		x0, x1, x2, x3, x4, x5, x6, x7 := x[0], x[1][:m], x[2][:m], x[3][:m], x[4][:m], x[5][:m], x[6][:m], x[7][:m]
		o0, o1, o2, o3, o4, o5, o6, o7 := o[0][:m], o[1][:m], o[2][:m], o[3][:m], o[4][:m], o[5][:m], o[6][:m], o[7][:m]
		w1, w2, w3, w4, w5, w6, w7 := w[1][:m], w[2][:m], w[3][:m], w[4][:m], w[5][:m], w[6][:m], w[7][:m]
		for i := range x0 {
			// bfly8, inlined by hand: it is over the inliner's budget.
			a1, a5 := x1[i]+x5[i], x1[i]-x5[i]
			a3, a7 := x3[i]+x7[i], x3[i]-x7[i]
			a1, a3 = a1+a3, a1-a3
			a3 = rotS(a3, s)
			a7 = rotS(a7, s)
			a5, a7 = a5+a7, a5-a7
			a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
			a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
			a0, a4 := x0[i]+x4[i], x0[i]-x4[i]
			a2, a6 := x2[i]+x6[i], x2[i]-x6[i]
			a0, a2 = a0+a2, a0-a2
			a6 = rotS(a6, s)
			a4, a6 = a4+a6, a4-a6
			o0[i] = a0 + a1
			o4[i] = (a0 - a1) * w4[i]
			o2[i] = (a2 + a3) * w2[i]
			o6[i] = (a2 - a3) * w6[i]
			o1[i] = (a4 + a5) * w1[i]
			o5[i] = (a4 - a5) * w5[i]
			o3[i] = (a6 + a7) * w3[i]
			o7[i] = (a6 - a7) * w7[i]
		}
	}
}

// The passXlast functions are the ido == 1 case — the final pass, where every
// point is a block of its own. There the per-block setup of the general loop
// runs once per point, so it is replaced by one flat loop over k: input block
// k is cc[r·k : r·k+r] and output j is the unit-stride stream ch[j·l1 : (j+1)·l1].

func pass2last(l1 int, cc, ch []complex128) {
	in := cc[:2*l1]
	o0, o1 := ch[:l1], ch[l1:][:l1]
	for k := range o0 {
		a, b := in[2*k], in[2*k+1]
		o0[k] = a + b
		o1[k] = a - b
	}
}

func pass3last(l1 int, cc, ch []complex128, s float64) {
	in := cc[:3*l1]
	o0, o1, o2 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1]
	for k := range o0 {
		x := in[3*k : 3*k+3]
		o0[k], o1[k], o2[k] = bfly3(x[0], x[1], x[2], s)
	}
}

func pass4last(l1 int, cc, ch []complex128, s float64) {
	in := cc[:4*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	for k := range o0 {
		x := in[4*k : 4*k+4]
		o0[k], o1[k], o2[k], o3[k] = bfly4(x[0], x[1], x[2], x[3], s)
	}
}

func pass5last(l1 int, cc, ch []complex128, s float64) {
	in := cc[:5*l1]
	o0, o1, o2, o3, o4 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1], ch[4*l1:][:l1]
	for k := range o0 {
		x := in[5*k : 5*k+5]
		// bfly5, inlined by hand: it is over the inliner's budget.
		a := x[0]
		t1, t2 := x[1]+x[4], x[1]-x[4]
		t3, t4 := x[2]+x[3], x[2]-x[3]
		r1 := a + complex(c51*real(t1)+c52*real(t3), c51*imag(t1)+c52*imag(t3))
		r2 := a + complex(c52*real(t1)+c51*real(t3), c52*imag(t1)+c51*imag(t3))
		i1 := rotS(complex(s51*real(t2)+s52*real(t4), s51*imag(t2)+s52*imag(t4)), s)
		i2 := rotS(complex(s52*real(t2)-s51*real(t4), s52*imag(t2)-s51*imag(t4)), s)
		o0[k] = a + t1 + t3
		o1[k] = r1 + i1
		o2[k] = r2 + i2
		o3[k] = r2 - i2
		o4[k] = r1 - i1
	}
}

func pass8last(l1 int, cc, ch []complex128, s float64) {
	in := cc[:8*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	o4, o5, o6, o7 := ch[4*l1:][:l1], ch[5*l1:][:l1], ch[6*l1:][:l1], ch[7*l1:][:l1]
	for k := range o0 {
		x := in[8*k : 8*k+8]
		// bfly8, inlined by hand: it is over the inliner's budget.
		a1, a5 := x[1]+x[5], x[1]-x[5]
		a3, a7 := x[3]+x[7], x[3]-x[7]
		a1, a3 = a1+a3, a1-a3
		a3 = rotS(a3, s)
		a7 = rotS(a7, s)
		a5, a7 = a5+a7, a5-a7
		a5 = complex(hsqt2*(real(a5)-s*imag(a5)), hsqt2*(imag(a5)+s*real(a5)))
		a7 = complex(hsqt2*(-s*imag(a7)-real(a7)), hsqt2*(s*real(a7)-imag(a7)))
		a0, a4 := x[0]+x[4], x[0]-x[4]
		a2, a6 := x[2]+x[6], x[2]-x[6]
		a0, a2 = a0+a2, a0-a2
		a6 = rotS(a6, s)
		a4, a6 = a4+a6, a4-a6
		o0[k], o4[k] = a0+a1, a0-a1
		o2[k], o6[k] = a2+a3, a2-a3
		o1[k], o5[k] = a4+a5, a4-a5
		o3[k], o7[k] = a6+a7, a6-a7
	}
}

func pass7last(l1 int, cc, ch []complex128, s float64) {
	in := cc[:7*l1]
	o0, o1, o2, o3 := ch[:l1], ch[l1:][:l1], ch[2*l1:][:l1], ch[3*l1:][:l1]
	o4, o5, o6 := ch[4*l1:][:l1], ch[5*l1:][:l1], ch[6*l1:][:l1]
	for k := range o0 {
		x := in[7*k : 7*k+7]
		// bfly7, inlined by hand: it is over the inliner's budget.
		a := x[0]
		s16, d16 := x[1]+x[6], x[1]-x[6]
		s25, d25 := x[2]+x[5], x[2]-x[5]
		s34, d34 := x[3]+x[4], x[3]-x[4]
		r1 := a + complex(c71*real(s16)+c72*real(s25)+c73*real(s34), c71*imag(s16)+c72*imag(s25)+c73*imag(s34))
		r2 := a + complex(c72*real(s16)+c73*real(s25)+c71*real(s34), c72*imag(s16)+c73*imag(s25)+c71*imag(s34))
		r3 := a + complex(c73*real(s16)+c71*real(s25)+c72*real(s34), c73*imag(s16)+c71*imag(s25)+c72*imag(s34))
		i1 := rotS(complex(s71*real(d16)+s72*real(d25)+s73*real(d34), s71*imag(d16)+s72*imag(d25)+s73*imag(d34)), s)
		i2 := rotS(complex(s72*real(d16)-s73*real(d25)-s71*real(d34), s72*imag(d16)-s73*imag(d25)-s71*imag(d34)), s)
		i3 := rotS(complex(s73*real(d16)-s71*real(d25)+s72*real(d34), s73*imag(d16)-s71*imag(d25)+s72*imag(d34)), s)
		o0[k] = a + s16 + s25 + s34
		o1[k], o6[k] = r1+i1, r1-i1
		o2[k], o5[k] = r2+i2, r2-i2
		o3[k], o4[k] = r3+i3, r3-i3
	}
}

// A batched pass runs one Stockham pass over w transforms side by side: the
// lines of a non-contiguous axis of an N-D array, w neighbouring lines at a
// time. Point p of the batch is the w values cc[p·sIn : p·sIn+w] (and
// ch[p·sOut : p·sOut+w]), so the first pass reads a strip of the array where
// it lies and the last one writes it back, with no gather or scatter.
// Every value of the batch gets the scalar pass's arithmetic, operation for
// operation, so the result is bit-identical to transforming each line alone.

// batchRadix reports whether a batched pass of radix r exists: the radices
// with a SIMD pass kernel.
func batchRadix(r int) bool {
	switch r {
	case 2, 3, 4, 5, 8:
		return true
	}
	return false
}

// batchTwiddles returns the stage's twiddles in the order a batched pass
// reads them, forward and conjugate: for i = 1 .. ido-1, the r-1 twiddles of
// point i one after the other. nil when ido == 1.
func (st *skStage) batchTwiddles() (fwd, conj []complex128) {
	if st.ido == 1 {
		return nil, nil
	}
	m, r := st.ido-1, st.r
	fwd = make([]complex128, m*(r-1))
	conj = make([]complex128, m*(r-1))
	for i := 0; i < m; i++ {
		for j := 0; j < r-1; j++ {
			fwd[i*(r-1)+j] = st.tw[j*m+i]
			conj[i*(r-1)+j] = st.twc[j*m+i]
		}
	}
	return fwd, conj
}

// passBatch runs the stage as a batched pass, on the SIMD kernel when there
// is one. tw is batchTwiddles' table for the direction.
func (st *skStage) passBatch(ch, cc, tw []complex128, w, sIn, sOut int, inverse bool) {
	if !kernels.StockhamBatchPass(st.r, st.ido, st.l1, cc, ch, tw, w, sIn, sOut, inverse) {
		st.passBatchScalar(ch, cc, tw, w, sIn, sOut, inverse)
	}
}

// passBatchScalar is the Go batched pass: each value of the batch through the
// butterfly the scalar pass uses (bfly2..bfly8, which pass2..pass8 inline),
// then, for i > 0, its twiddle.
func (st *skStage) passBatchScalar(ch, cc, tw []complex128, w, sIn, sOut int, inverse bool) {
	r, ido, l1 := st.r, st.ido, st.l1
	s := dirSign(inverse)
	var x, y [8]complex128
	for k := 0; k < l1; k++ {
		for i := 0; i < ido; i++ {
			for c := 0; c < w; c++ {
				for j := 0; j < r; j++ {
					x[j] = cc[(i+ido*(j+r*k))*sIn+c]
				}
				bflyR(&y, &x, r, s)
				for j := 0; j < r; j++ {
					v := y[j]
					if i > 0 && j > 0 {
						v *= tw[(i-1)*(r-1)+j-1]
					}
					ch[(i+ido*(k+l1*j))*sOut+c] = v
				}
			}
		}
	}
}

// bflyR is the size-r DFT of x[:r] into y[:r] for a batched radix.
func bflyR(y, x *[8]complex128, r int, s float64) {
	switch r {
	case 2:
		y[0], y[1] = x[0]+x[1], x[0]-x[1]
	case 3:
		y[0], y[1], y[2] = bfly3(x[0], x[1], x[2], s)
	case 4:
		y[0], y[1], y[2], y[3] = bfly4(x[0], x[1], x[2], x[3], s)
	case 5:
		y[0], y[1], y[2], y[3], y[4] = bfly5(x[0], x[1], x[2], x[3], x[4], s)
	default:
		bfly8(y, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], s)
	}
}
