package fft

import "github.com/go-fft/fft/internal/kernels"

// The blocked schedule for large powers of two (BENCHMARKS.md, Round 20).
//
// Breadth first, every Stockham pass reads and writes the whole array. Once
// dst and the scratch buffer outgrow the L2 cache, each pass is bound by the
// bandwidth of the next level: on Cascade Lake (1 MB of L2 per core), every
// pass of a 65536-point transform cost 1.4–2.4 ns per point against 0.6–0.8
// at 16384, close to what a plain copy between two 1 MB buffers costs there
// (1.3 ns per point), and the six passes added up to 10.4 ns per point.
//
// The passes after the first t only combine points within a block of pass t:
// block k of pass t (m = n/l1_t consecutive points) and everything computed
// from it later never meet another block. So the first t passes run over the
// whole array, as before, and the rest run group by group: b consecutive
// blocks of pass t at a time, as a Stockham transform of b·m points whose l1
// starts at b, between two local buffers that stay in L2. The pass kernels
// are the ones breadth first runs, with l1 = b·r_t·…: a pass's twiddles
// depend on its radix and ido only, so its table serves both. That local
// transform leaves its b transforms of m points interleaved, value f of block
// c at c + b·f, where the whole transform wants it at k0 + c + l1_t·f; the
// final pass (kernels.StockhamLastRun) writes it there directly, in runs of b
// consecutive points. When the first t passes include two that strided
// kernels can run (cascadePair), those two make one sweep over the array
// instead of two. Every value gets the same operations as breadth first, in
// the same order: the result is bit-identical.
//
// On Cascade Lake the six passes over the array of a 65536-point transform
// became two sweeps (the fused pair, then the groups); 7.2 against 9.9 ns per
// point measured pass by pass (Round 20). The groups run in L2 at the speed
// of the pass kernels, which is where AVX-512 is worth its lower clock: with
// the AVX2 kernels, the same schedule tied breadth first. So it runs only with
// the AVX-512 kernels (cascadeMinAMD64).

// cascadeLocal is the number of points a group holds (b·m): two local buffers
// of 128 KB. On Cascade Lake, at 65536 and 2^17 points, groups of 4096 and
// 8192 points ran fastest; 32768 (1 MB for the pair) spilled out of L2.
const cascadeLocal = 8192

// cascadeSkew separates the two local buffers by setGap bytes beyond their
// length, so they do not sit a multiple of 4 KB apart (see offTheSets).
const cascadeSkew = setGap / 16

// cascadeMin is the smallest length that runs the blocked schedule, 0 for
// none; it is per-architecture (route_*.go) and a variable so the tests can
// run the schedule anywhere.
var cascadeMin = cascadeMinDefault()

// cascadeShape returns the number of passes run over the whole array and the
// group width b of the blocked schedule for a power of two n with these
// passes, or 0, 0 when n runs breadth first. t is the first pass whose blocks
// are at most cascadeLocal/4 points (so that b >= 4 and the final pass writes
// whole cache lines), and b fills a group: b·m = cascadeLocal. A transform
// of at most cascadeLocal points is one group: it runs breadth first.
func cascadeShape(n int, stages []skStage, min int) (t, b int) {
	if min == 0 || n < min || n&(n-1) != 0 {
		return 0, 0
	}
	for t = 1; t < len(stages); t++ {
		if m := n / stages[t].l1; 4*m <= cascadeLocal {
			if b = cascadeLocal / m; b > stages[t].l1 {
				break // the whole transform fits in one group
			}
			return t, b
		}
	}
	return 0, 0
}

// cascadeScratch is the scratch length a plan with that shape needs: the
// window that slides off dst's sets (takesGap) and the two local buffers.
func cascadeScratch(n int) int { return n + setSpan + 2*cascadeLocal + cascadeSkew }

// cascadeRun is transform (and run) on the blocked schedule. buf is the plan's
// scratch buffer (cascadeScratch long). dst may alias src.
func (p *skPlan) cascadeRun(dst, src, buf []complex128, inverse bool) {
	n, t, b := p.n, p.cascT, p.cascB
	scr := offTheSets(buf[:n+setSpan], dst, n)
	loc := buf[n+setSpan:]
	// The passes over the whole array alternate so that pass t-1 writes the
	// scratch buffer: the groups read it and write dst. In place, pass 0 must
	// not write the array it reads; with t even it would, so src is copied
	// to the scratch buffer first and read from there.
	in, k := src, 0
	if p.cascPair {
		out := scr
		if t%2 == 1 {
			out = dst
		}
		p.cascadePair(out, in, loc, inverse)
		in, k = out, 2
	}
	if k == 0 && t%2 == 0 && &dst[0] == &src[0] {
		copy(scr, src)
		in = scr
	}
	for ; k < t; k++ {
		out := scr
		if (t-1-k)%2 == 1 {
			out = dst
		}
		p.stages[k].pass(out, in, inverse)
		in = out
	}
	s := len(p.stages)
	l1t := p.stages[t].l1
	m := n / l1t
	g := b * m
	bufs := [2][]complex128{loc[:g], loc[g+cascadeSkew : 2*g+cascadeSkew]}
	last := &p.stages[s-1]
	for k0 := 0; k0 < l1t; k0 += b {
		cur := in[k0*m : k0*m+g]
		l1 := b
		for q := t; q < s-1; q++ {
			st := &p.stages[q]
			out := bufs[(q-t)%2]
			st.cascadePass(out, cur, l1, inverse)
			l1 *= st.r
			cur = out
		}
		last.cascadeLast(dst[k0:], cur, n/last.r, b, l1t, inverse)
	}
}

// cascadeChunk is the number of points of pass 1's ido that cascadePair takes
// at a time: r0·r1·cascadeChunk values (16384 for two radix-8 passes, both
// local buffers) go through the local buffer. On Cascade Lake, chunks of 64 to
// 256 points ran within 3% of each other up to 2^17; from 2^18, where the
// array is in DRAM, 256 (4 KB runs) was fastest and 32 lost 10–30%.
const cascadeChunk = 256

// cascadePairs reports whether cascadePair can run the first two passes of a
// plan with this shape: a strided kernel exists for both radices, the chunks
// tile pass 1's ido, and they fit the local buffers.
func cascadePairs(t int, stages []skStage) bool {
	if t < 2 {
		return false
	}
	r0, r1 := stages[0].r, stages[1].r
	strided := func(r int) bool { return r == 4 || r == 8 }
	return strided(r0) && strided(r1) && stages[1].ido%cascadeChunk == 0 && r0*r1*cascadeChunk <= 2*cascadeLocal
}

// cascadePair runs passes 0 and 1 from in to out (which may be in) in one
// sweep over the array instead of two. Point i' of pass 1 (one of its ido)
// needs pass 0's outputs at the r1 points i' + ido1·m', which need pass 0's
// inputs at i' + ido1·m' + ido0·m: the r0·r1 values that end in pass 1's
// outputs i' + ido1·(k + r0·j). So a chunk of cascadeChunk consecutive i' goes
// through both passes at once: pass 0 on its r1 blocks into the local buffer,
// pass 1 from there into out (kernels.StockhamStrided). A chunk reads and
// writes the same positions, and reads them all before it writes: out may be
// in. Each value gets the operations the two passes give it, so the result is
// bit-identical.
func (p *skPlan) cascadePair(out, in, loc []complex128, inverse bool) {
	s0, s1 := &p.stages[0], &p.stages[1]
	r0, r1, ido0, ido1 := s0.r, s1.r, s0.ido, s1.ido
	w := cascadeChunk
	lb := loc[:r0*r1*w]
	for i0 := 0; i0 < ido1; i0 += w {
		// Pass 0, blocks m' < r1: point i0+i of block m' is pass 0's point
		// i0 + ido1·m' + i; its outputs go to lb[i + w·(m' + r1·j)].
		s0.cascadeStrided(lb, in[i0:], w, r1, ido0, ido1, r1*w, w, i0, ido1, inverse)
		// Pass 1, blocks k < r0, from the local buffer to pass 1's outputs:
		// every block's points are i0 .. i0+w-1.
		s1.cascadeStrided(out[i0:], lb, w, r0, w, r1*w, r0*ido1, ido1, i0, 0, inverse)
	}
}

// cascadeStrided runs the stage on nb blocks of cnt points with the strides
// given (see kernels.StockhamStrided): the first point of block b is the
// stage's point i0 + b·bi.
func (st *skStage) cascadeStrided(ch, cc []complex128, cnt, nb, sin, bin, sout, bout, i0, bi int, inverse bool) {
	r := st.r
	twX := st.twX
	if inverse {
		twX = st.twXc
	}
	if len(twX) == 0 || !kernels.StockhamStrided(r, cc, ch, twX[i0*(r-1):], cnt, nb, sin, bin, sout, bout, bi*(r-1), i0 == 0, i0 == 0 && bi == 0, inverse, st.wide) {
		st.cascadeStridedGo(ch, cc, cnt, nb, sin, bin, sout, bout, i0, bi, inverse)
	}
}

// cascadeStridedGo is cascadeStrided in Go: the butterflies pass{r} runs
// (bfly4, bfly8), then the twiddle of every point but i = 0, as
// passBatchScalar does.
func (st *skStage) cascadeStridedGo(ch, cc []complex128, cnt, nb, sin, bin, sout, bout, i0, bi int, inverse bool) {
	r := st.r
	tw := st.tw
	if inverse {
		tw = st.twc
	}
	m, s := st.ido-1, dirSign(inverse)
	var x, y [8]complex128
	for b := range nb {
		for i := range cnt {
			for j := range r {
				x[j] = cc[b*bin+i+j*sin]
			}
			bflyR(&y, &x, r, s)
			g := i0 + b*bi + i
			for j := range r {
				v := y[j]
				if g > 0 && j > 0 {
					v *= tw[(g-1)+(j-1)*m]
				}
				ch[b*bout+i+j*sout] = v
			}
		}
	}
}

// cascadePass is pass with l1 blocks instead of the stage's own.
func (st *skStage) cascadePass(ch, cc []complex128, l1 int, inverse bool) {
	twX := st.twX
	if inverse {
		twX = st.twXc
	}
	if !kernels.StockhamPass(st.r, st.ido, l1, cc, ch, twX, inverse, st.wide) {
		c := *st
		c.l1 = l1
		c.passScalar(ch, cc, inverse)
	}
}

// cascadeLast runs the final pass (ido == 1) over the blocks of cc, writing
// output j of block q·run + c to dst[q·gap + c + j·os].
func (st *skStage) cascadeLast(dst, cc []complex128, os, run, gap int, inverse bool) {
	if !kernels.StockhamLastRun(st.r, cc, dst, os, len(cc)/(st.r*run), run, gap, inverse, st.wide) {
		st.cascadeLastGo(dst, cc, os, run, gap, inverse)
	}
}

// cascadeLastGo is cascadeLast in Go: the butterflies pass{r}last runs
// (bfly2..bfly8), stored at the run positions.
func (st *skStage) cascadeLastGo(dst, cc []complex128, os, run, gap int, inverse bool) {
	r, s := st.r, dirSign(inverse)
	var x, y [8]complex128
	for k := range len(cc) / r {
		copy(x[:r], cc[r*k:r*k+r])
		bflyR(&y, &x, r, s)
		o := k%run + gap*(k/run)
		for j := range r {
			dst[o+j*os] = y[j]
		}
	}
}
