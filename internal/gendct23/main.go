//go:build ignore

// Command gendct23 writes internal/kernels/dct23_amd64.s: the fused DCT-II and
// DCT-III of 32 and 64 points (go-fft/fft#162) as one straight-line scalar
// program each, entirely in registers (AVX-512 gives 32 xmm registers; a value
// that does not fit is spilled to the frame).
//
// The program is the Makhoul route of dct23.go written out: the reorder is folded
// into the loads, the half-length complex FFT is a split-radix network with its
// twiddles folded into constants, and the untangle (floats.MirrorCombineSplit for
// the DCT-II; CMulZipRev, the z[0] fix-up and floats.MirrorCombine for the DCT-III)
// is folded into the outputs, so a call touches no scratch, no pool and no plan.
//
// It is built as a dataflow graph of real operations (load, add, subtract,
// multiply by a constant) with exact simplifications (a multiplication by 0 or +-1,
// a rotation by +-i, a sum of two multiples of one value), then lowered in an
// order chosen by trying both of them and keeping the one with fewer spills: a
// multiplication used once is fused into the add or subtract that consumes it
// (VFMADD / VFNMADD / VFMSUB, 231 or 132 form, whichever overwrites a dying
// value), a load is a memory operand of the add that needs it, and the registers
// are allocated lowest-free with spills of the value used furthest away.
//
// With -check the program evaluates the graph in float64 and compares it with the
// direct formulas of the route (a naive DFT for the FFT); nothing is written.
//
// Run from the package root with `go generate` (see dct23.go) or
// `go run internal/gendct23/main.go [dir]`.
package main

import (
	"fmt"
	"go/format"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/go-fft/fft/internal/gentwiddle"
)

// ---- the graph ----

type node struct {
	id   int
	op   byte // l load, a add, s sub, m mul by c, n negate, M product of two, S store
	x, y *node
	c    float64
	mem  string // l: the memory operand
	idx  int    // S: the dst index
}

type key struct {
	op   byte
	x, y int
	c    float64
	mem  string
	idx  int
}

type val struct {
	n   *node // nil is zero
	neg bool
}

type builder struct {
	nodes []*node
	cse   map[key]*node
	outs  []*node
}

func newBuilder() *builder { return &builder{cse: map[key]*node{}} }

func (b *builder) mk(op byte, x, y *node, c float64, mem string, idx int) *node {
	k := key{op: op, c: c, mem: mem, idx: idx}
	if x != nil {
		k.x = x.id + 1
	}
	if y != nil {
		k.y = y.id + 1
	}
	if op != 'S' {
		if n, ok := b.cse[k]; ok {
			return n
		}
	}
	n := &node{id: len(b.nodes), op: op, x: x, y: y, c: c, mem: mem, idx: idx}
	b.nodes = append(b.nodes, n)
	if op != 'S' {
		b.cse[k] = n
	}
	return n
}

func (b *builder) load(i int) val {
	return val{n: b.mk('l', nil, nil, 0, fmt.Sprintf("%d(SI)", 8*i), i)}
}
func (b *builder) arg(name string, off int) val {
	return val{n: b.mk('l', nil, nil, 0, fmt.Sprintf("%s+%d(FP)", name, off), -1)}
}

func negv(v val) val {
	if v.n != nil {
		v.neg = !v.neg
	}
	return v
}

func (b *builder) mulc(v val, c float64) val {
	if v.n == nil || c == 0 {
		return val{}
	}
	if c < 0 {
		c = -c
		v.neg = !v.neg
	}
	if c == 1 {
		return v
	}
	if v.n.op == 'm' { // (x*c1)*c2
		return b.mulc(val{n: v.n.x, neg: v.neg}, v.n.c*c)
	}
	return val{n: b.mk('m', v.n, nil, c, "", 0), neg: v.neg}
}

func (b *builder) mul(p, q val) val {
	if p.n == nil || q.n == nil {
		return val{}
	}
	if q.n.id < p.n.id {
		p, q = q, p
	}
	return val{n: b.mk('M', p.n, q.n, 0, "", 0), neg: p.neg != q.neg}
}

func signedc(v val) float64 {
	if v.neg {
		return -v.n.c
	}
	return v.n.c
}

func (b *builder) add(p, q val) val {
	if p.n == nil {
		return q
	}
	if q.n == nil {
		return p
	}
	if p.n == q.n {
		if p.neg != q.neg {
			return val{}
		}
		return b.mulc(p, 2)
	}
	if p.n.op == 'm' && q.n.op == 'm' && p.n.x == q.n.x {
		return b.mulc(val{n: p.n.x}, signedc(p)+signedc(q))
	}
	switch {
	case p.neg && q.neg:
		return val{n: b.mk('a', p.n, q.n, 0, "", 0), neg: true}
	case p.neg:
		return val{n: b.mk('s', q.n, p.n, 0, "", 0)}
	case q.neg:
		return val{n: b.mk('s', p.n, q.n, 0, "", 0)}
	}
	return val{n: b.mk('a', p.n, q.n, 0, "", 0)}
}

func (b *builder) sub(p, q val) val { return b.add(p, negv(q)) }

// real materialises a value with its sign.
func (b *builder) real(v val) *node {
	if !v.neg {
		return v.n
	}
	switch v.n.op {
	case 's':
		return b.mk('s', v.n.y, v.n.x, 0, "", 0)
	case 'm':
		return b.mk('m', v.n.x, nil, -v.n.c, "", 0)
	}
	return b.mk('n', v.n, nil, 0, "", 0)
}

func (b *builder) store(idx int, v val) {
	if v.n == nil {
		panic(fmt.Sprintf("output %d is zero", idx))
	}
	b.outs = append(b.outs, b.mk('S', b.real(v), nil, 0, "", idx))
}

// lin is sum of v[i]*c[i] as a chain, the first product a multiplication and the
// rest fused into it.
func (b *builder) lin(vs []val, cs []float64) val {
	r := val{}
	for i := range vs {
		r = b.add(r, b.mulc(vs[i], cs[i]))
	}
	return r
}

type cx struct{ r, i val }

func (b *builder) cadd(p, q cx) cx { return cx{b.add(p.r, q.r), b.add(p.i, q.i)} }
func (b *builder) csub(p, q cx) cx { return cx{b.sub(p.r, q.r), b.sub(p.i, q.i)} }
func rotNegI(p cx) cx              { return cx{p.i, negv(p.r)} }
func rotPosI(p cx) cx              { return cx{negv(p.i), p.r} }

// cmulc is p * (c + i s).
func (b *builder) cmulc(p cx, c, s float64) cx {
	return cx{
		b.add(b.mulc(p.r, c), b.mulc(p.i, -s)),
		b.add(b.mulc(p.r, s), b.mulc(p.i, c)),
	}
}

// cs is cos and sin of 2*pi*num/den, correctly rounded, the quarter turns exact.
func cs(num, den int) (c, s float64) {
	if num < 0 {
		c, s = cs(-num, den)
		return c, -s
	}
	num %= den
	if (4*num)%den == 0 {
		switch 4 * num / den {
		case 0:
			return 1, 0
		case 1:
			return 0, 1
		case 2:
			return -1, 0
		default:
			return 0, -1
		}
	}
	return gentwiddle.CosSin(num, den)
}

// fft is the split-radix DFT of x (forward when !inv).
func (b *builder) fft(x []cx, inv bool) []cx {
	n := len(x)
	rot := rotNegI
	sgn := -1
	if inv {
		rot, sgn = rotPosI, 1
	}
	switch n {
	case 1:
		return x
	case 2:
		return []cx{b.cadd(x[0], x[1]), b.csub(x[0], x[1])}
	case 4:
		a, bb := b.cadd(x[0], x[2]), b.csub(x[0], x[2])
		c, d := b.cadd(x[1], x[3]), rot(b.csub(x[1], x[3]))
		return []cx{b.cadd(a, c), b.cadd(bb, d), b.csub(a, c), b.csub(bb, d)}
	}
	ev := make([]cx, n/2)
	for m := range ev {
		ev[m] = x[2*m]
	}
	z1, z3 := make([]cx, n/4), make([]cx, n/4)
	for m := range z1 {
		z1[m], z3[m] = x[4*m+1], x[4*m+3]
	}
	u, a1, a3 := b.fft(ev, inv), b.fft(z1, inv), b.fft(z3, inv)
	out := make([]cx, n)
	for k := 0; k < n/4; k++ {
		c1, s1 := cs(sgn*k, n)
		c3, s3 := cs(sgn*3*k, n)
		t1, t2 := b.cmulc(a1[k], c1, s1), b.cmulc(a3[k], c3, s3)
		sum, dif := b.cadd(t1, t2), rot(b.csub(t1, t2))
		out[k], out[k+n/2] = b.cadd(u[k], sum), b.csub(u[k], sum)
		out[k+n/4], out[k+3*n/4] = b.cadd(u[k+n/4], dif), b.csub(u[k+n/4], dif)
	}
	return out
}

// ---- the two transforms of dct23.go ----

// forward is d.forward: dst[0:n] = DCT-II of src[0:n].
func (b *builder) forward(n int) {
	h, q := n/2, n/4
	z := make([]cx, h)
	for j := 0; j < q; j++ {
		z[j] = cx{b.load(4 * j), b.load(4*j + 2)}
		z[q+j] = cx{b.load(2*n - 1 - 4*(q+j)), b.load(2*n - 3 - 4*(q+j))}
	}
	y := b.fft(z, false)
	c1h, _ := cs(h, 4*n)
	x0 := b.mulc(b.add(y[0].r, y[0].i), 2)
	xh := b.mulc(b.sub(y[0].r, y[0].i), 2*c1h)
	b.store(0, x0)
	b.store(h, xh)
	for k := 1; k < h; k++ {
		a, bb := y[k], y[h-k]
		ur, ui := b.add(a.r, bb.r), b.sub(a.i, bb.i)
		vr, vi := b.add(a.i, bb.i), b.sub(bb.r, a.r)
		w1r, w1s := cs(k, 4*n)
		w1i := -w1s
		w2r, w2s := cs(5*k, 4*n)
		w2i := -w2s
		vs := []val{ur, ui, vr, vi}
		b.store(k, b.lin(vs, []float64{w1r, -w1i, w2r, -w2i}))
		b.store(n-k, b.lin(vs, []float64{-w1i, -w1r, -w2i, -w2r}))
	}
}

// inverse is d.inverse: dst[0:n] = DCT-III of src[0:n], src[0] times a0.
func (b *builder) inverse(n int) {
	h, q := n/2, n/4
	a0 := b.arg("a0", 16)
	u := make([]cx, h+1)
	u[0] = cx{b.mul(a0, b.load(0)), val{}}
	for k := 1; k <= h; k++ {
		cr, cs1 := cs(k, 4*n) // cw = conj(c1) = (cos, +sin of the positive angle)
		ci := cs1
		sk, sn := b.load(k), b.load(n-k)
		// (sk - i sn) * (cr + i ci)
		u[k] = cx{
			b.add(b.mulc(sk, cr), b.mulc(sn, ci)),
			b.sub(b.mulc(sk, ci), b.mulc(sn, cr)),
		}
	}
	z := make([]cx, h)
	a, bb := u[0], u[h]
	z[0] = cx{
		b.sub(b.add(a.r, bb.r), b.add(a.i, bb.i)),
		b.add(b.sub(a.i, bb.i), b.sub(a.r, bb.r)),
	}
	for k := 1; k < h; k++ {
		a, bb := u[k], u[h-k]
		ur, ui := b.add(a.r, bb.r), b.sub(a.i, bb.i)
		vr, vi := b.add(a.i, bb.i), b.sub(bb.r, a.r)
		tr, ts := cs(k, n) // ct = exp(+2 pi i k/n); mt = -ct
		w2r, w2i := -tr, -ts
		z[k] = cx{
			b.add(b.add(ur, b.mulc(vr, w2r)), b.mulc(vi, -w2i)),
			b.add(b.add(ui, b.mulc(vi, w2r)), b.mulc(vr, w2i)),
		}
	}
	y := b.fft(z, true)
	for j := 0; j < q; j++ {
		b.store(4*j, y[j].r)
		b.store(4*j+2, y[j].i)
		b.store(2*n-1-4*(q+j), y[q+j].r)
		b.store(2*n-3-4*(q+j), y[q+j].i)
	}
}

// ---- evaluation (-check) ----

func eval(b *builder, src []float64, a0 float64) []float64 {
	memo := map[*node]float64{}
	var ev func(n *node) float64
	ev = func(n *node) float64 {
		if v, ok := memo[n]; ok {
			return v
		}
		var v float64
		switch n.op {
		case 'l':
			if n.idx < 0 {
				v = a0
			} else {
				v = src[n.idx]
			}
		case 'a':
			v = ev(n.x) + ev(n.y)
		case 's':
			v = ev(n.x) - ev(n.y)
		case 'm':
			v = ev(n.x) * n.c
		case 'n':
			v = -ev(n.x)
		case 'M':
			v = ev(n.x) * ev(n.y)
		}
		memo[n] = v
		return v
	}
	dst := make([]float64, len(src))
	for _, s := range b.outs {
		dst[s.idx] = ev(s.x)
	}
	return dst
}

// naive is the route of dct23.go with a naive DFT.
func naiveDFT(z []complex128, inv bool) []complex128 {
	n := len(z)
	out := make([]complex128, n)
	sg := -1.0
	if inv {
		sg = 1
	}
	for k := range out {
		for j := range z {
			s, c := math.Sincos(sg * 2 * math.Pi * float64(j*k%n) / float64(n))
			out[k] += z[j] * complex(c, s)
		}
	}
	return out
}

func naiveForward(src []float64) []float64 {
	n := len(src)
	h, q := n/2, n/4
	z := make([]complex128, h)
	for j := 0; j < q; j++ {
		z[j] = complex(src[4*j], src[4*j+2])
		z[q+j] = complex(src[2*n-1-4*(q+j)], src[2*n-3-4*(q+j)])
	}
	y := naiveDFT(z, false)
	dst := make([]float64, n)
	dst[0] = 2 * (real(y[0]) + imag(y[0]))
	sh, ch := math.Sincos(-math.Pi * float64(h) / float64(2*n))
	_ = sh
	dst[h] = 2 * ch * (real(y[0]) - imag(y[0]))
	for k := 1; k < h; k++ {
		a, b := y[k], y[h-k]
		u := complex(real(a)+real(b), imag(a)-imag(b))
		v := complex(imag(a)+imag(b), real(b)-real(a))
		s1, c1 := math.Sincos(-math.Pi * float64(k) / float64(2*n))
		s2, c2 := math.Sincos(-math.Pi * float64(5*k) / float64(2*n))
		r := complex(c1, s1)*u + complex(c2, s2)*v
		dst[k], dst[n-k] = real(r), -imag(r)
	}
	return dst
}

func naiveInverse(src []float64, a0 float64) []float64 {
	n := len(src)
	h, q := n/2, n/4
	u := make([]complex128, h+1)
	u[0] = complex(a0*src[0], 0)
	for k := 1; k <= h; k++ {
		s, c := math.Sincos(-math.Pi * float64(k) / float64(2*n))
		u[k] = complex(c, -s) * complex(src[k], -src[n-k])
	}
	z := make([]complex128, h)
	a, b := u[0], u[h]
	z[0] = complex(real(a)+real(b)-(imag(a)+imag(b)), imag(a)-imag(b)+(real(a)-real(b)))
	for k := 1; k < h; k++ {
		a, b := u[k], u[h-k]
		p := complex(real(a)+real(b), imag(a)-imag(b))
		v := complex(imag(a)+imag(b), real(b)-real(a))
		s, c := math.Sincos(2 * math.Pi * float64(k) / float64(n))
		z[k] = p + (-complex(c, s))*v
	}
	y := naiveDFT(z, true)
	dst := make([]float64, n)
	for j := 0; j < q; j++ {
		dst[4*j] = real(y[j])
		dst[4*j+2] = imag(y[j])
		dst[2*n-1-4*(q+j)] = real(y[q+j])
		dst[2*n-3-4*(q+j)] = imag(y[q+j])
	}
	return dst
}

func check(n int) bool {
	ok := true
	rng := rand.New(rand.NewSource(1))
	for _, fwd := range []bool{true, false} {
		b := newBuilder()
		if fwd {
			b.forward(n)
		} else {
			b.inverse(n)
		}
		if len(b.outs) != n {
			fmt.Printf("n=%d fwd=%v: %d outputs\n", n, fwd, len(b.outs))
			ok = false
		}
		worst := 0.0
		for t := 0; t < 20; t++ {
			src := make([]float64, n)
			for i := range src {
				src[i] = rng.Float64()*2 - 1
			}
			a0 := 1.0
			if t%2 == 1 {
				a0 = math.Sqrt2
			}
			var want []float64
			if fwd {
				want = naiveForward(src)
			} else {
				want = naiveInverse(src, a0)
			}
			got := eval(b, src, a0)
			for i := range got {
				worst = math.Max(worst, math.Abs(got[i]-want[i]))
			}
		}
		fmt.Printf("n=%d forward=%v: worst |graph - naive| = %.3g\n", n, fwd, worst)
		if !(worst < 1e-12) {
			ok = false
		}
	}
	return ok
}

// ---- lowering ----

type item struct {
	n     *node
	m     *node // a multiplication fused into n
	macc  *node // the other operand of a fused n
	memop *node // a load used as the memory operand of n
}

// plan orders the live nodes, fuses the single-use multiplications and picks the
// memory operands.
func plan(b *builder, order []*node) []item {
	uses := map[*node]int{}
	for _, n := range order {
		for _, o := range []*node{n.x, n.y} {
			if o != nil {
				uses[o]++
			}
		}
	}
	fused := map[*node]bool{}
	var items []item
	pos := map[*node]int{}
	for i, n := range order {
		pos[n] = i
	}
	for _, n := range order {
		if fused[n] {
			continue
		}
		it := item{n: n}
		if n.op == 'a' || n.op == 's' {
			cx, cy := n.x.op == 'm' && uses[n.x] == 1 && !fused[n.x], n.y.op == 'm' && uses[n.y] == 1 && !fused[n.y]
			switch {
			case cx && cy:
				// materialise the earlier one, fuse the later
				if pos[n.x] < pos[n.y] {
					it.m, it.macc = n.y, n.x
				} else {
					it.m, it.macc = n.x, n.y
				}
			case cy:
				it.m, it.macc = n.y, n.x
			case cx:
				it.m, it.macc = n.x, n.y
			}
			if it.m != nil {
				fused[it.m] = true
			} else {
				switch {
				case n.y.op == 'l':
					it.memop = n.y
				case n.x.op == 'l' && n.op == 'a':
					it.memop = n.x
				}
			}
		}
		items = append(items, it)
	}
	// fused multiplications leave the stream (they were emitted before their
	// consumer in order; drop them)
	var out []item
	for _, it := range items {
		if !fused[it.n] {
			out = append(out, it)
		}
	}
	return out
}

// reads is the nodes an item needs in registers.
func (it item) reads() []*node {
	var r []*node
	add := func(n *node) {
		if n != nil && n != it.memop {
			r = append(r, n)
		}
	}
	switch {
	case it.m != nil:
		add(it.m.x)
		add(it.macc)
	default:
		add(it.n.x)
		add(it.n.y)
	}
	return r
}

type lowering struct {
	sb      strings.Builder
	reg     map[*node]int
	slot    map[*node]int
	free    [32]bool
	nslot   int
	spills  int
	reloads int
	ops     int
	consts  map[uint64]bool
	negMask bool
	next    map[*node][]int // item positions that read the node in a register
	cur     int
	pinned  map[*node]bool
	uses    map[*node]int // remaining register reads
	count   map[string]int
}

func (lw *lowering) emit(format string, a ...any) {
	fmt.Fprintf(&lw.sb, "\t"+format+"\n", a...)
	lw.ops++
}

func (lw *lowering) cname(c float64) string {
	bits := math.Float64bits(c)
	lw.consts[bits] = true
	return fmt.Sprintf("d23_%016x<>(SB)", bits)
}

func (lw *lowering) nextUse(n *node) int {
	for _, p := range lw.next[n] {
		if p > lw.cur {
			return p
		}
	}
	return 1 << 30
}

// alloc returns a free register, spilling the unpinned value used furthest away.
func (lw *lowering) alloc() int {
	for r := 0; r < 32; r++ {
		if lw.free[r] {
			lw.free[r] = false
			return r
		}
	}
	var victim *node
	far := -1
	for n, r := range lw.reg {
		if r < 0 || lw.pinned[n] {
			continue
		}
		if d := lw.nextUse(n); d > far || (d == far && n.id < victim.id) {
			victim, far = n, d
		}
	}
	if victim == nil {
		panic("no register to spill")
	}
	r := lw.reg[victim]
	if victim.op != 'l' {
		if _, ok := lw.slot[victim]; !ok {
			lw.slot[victim] = lw.nslot
			lw.nslot++
			lw.emit("VMOVSD X%d, %d(SP)", r, 8*lw.slot[victim])
			lw.spills++
		}
	}
	delete(lw.reg, victim)
	return r
}

// ensure puts n in a register.
func (lw *lowering) ensure(n *node) int {
	if r, ok := lw.reg[n]; ok {
		return r
	}
	r := lw.alloc()
	if n.op == 'l' {
		lw.emit("VMOVSD %s, X%d", n.mem, r)
	} else {
		lw.emit("VMOVSD %d(SP), X%d", 8*lw.slot[n], r)
		lw.reloads++
	}
	lw.reg[n] = r
	return r
}

func (lw *lowering) release(n *node) {
	if r, ok := lw.reg[n]; ok {
		lw.free[r] = true
		delete(lw.reg, n)
	}
}

// dies reports whether n has no register read after the current item.
func (lw *lowering) dies(n *node) bool { return lw.nextUse(n) == 1<<30 }

func lower(b *builder, items []item) (*lowering, bool) {
	lw := &lowering{reg: map[*node]int{}, slot: map[*node]int{}, consts: map[uint64]bool{}, next: map[*node][]int{}, pinned: map[*node]bool{}, count: map[string]int{}}
	for r := range lw.free {
		lw.free[r] = true
	}
	for i, it := range items {
		seen := map[*node]bool{}
		for _, r := range it.reads() {
			if !seen[r] {
				lw.next[r] = append(lw.next[r], i)
				seen[r] = true
			}
		}
	}
	for i, it := range items {
		lw.cur = i
		rd := it.reads()
		for _, r := range rd {
			lw.pinned[r] = true
		}
		for _, r := range rd {
			lw.ensure(r)
		}
		n := it.n
		R := func(x *node) int { return lw.reg[x] }
		// release the operands that die here, but keep their registers readable
		var dying []*node
		seen := map[*node]bool{}
		for _, r := range rd {
			if !seen[r] && lw.dies(r) {
				dying = append(dying, r)
			}
			seen[r] = true
		}
		finish := func() {
			for _, r := range rd {
				delete(lw.pinned, r)
			}
		}
		switch {
		case n.op == 'S':
			x := n.x
			lw.emit("VMOVSD X%d, %d(DI)", R(x), 8*n.idx)
			for _, d := range dying {
				lw.release(d)
			}
			finish()
			continue
		case it.m != nil:
			m, acc := it.m, it.macc
			x := m.x
			cm := lw.cname(m.c)
			// the form of n: acc +/- x*c
			var f231, f132 string
			switch {
			case n.op == 'a':
				f231, f132 = "VFMADD231SD", "VFMADD132SD"
			case n.y == m: // acc - x*c
				f231, f132 = "VFNMADD231SD", "VFNMADD132SD"
			default: // x*c - acc
				f231, f132 = "VFMSUB231SD", "VFMSUB132SD"
			}
			var dst int
			ra, rx := R(acc), R(x)
			switch {
			case acc == x:
				if lw.dies(acc) {
					dst = ra
					delete(lw.reg, acc)
					lw.emit("%s %s, X%d, X%d", f231, cm, rx, dst)
				} else {
					dst = lw.alloc()
					lw.emit("VMOVAPD X%d, X%d", ra, dst)
					lw.emit("%s %s, X%d, X%d", f231, cm, rx, dst)
				}
				lw.count["copy"]++
			case lw.dies(acc):
				dst = ra
				delete(lw.reg, acc)
				lw.emit("%s %s, X%d, X%d", f231, cm, rx, dst)
			case lw.dies(x):
				dst = rx
				delete(lw.reg, x)
				lw.emit("%s %s, X%d, X%d", f132, cm, ra, dst)
			default:
				dst = lw.alloc()
				lw.emit("VMOVAPD X%d, X%d", ra, dst)
				lw.emit("%s %s, X%d, X%d", f231, cm, rx, dst)
				lw.count["copy"]++
			}
			lw.count["fma"]++
			for _, d := range dying {
				lw.release(d)
			}
			lw.reg[n] = dst
			finish()
			continue
		}
		// plain op: read the operand registers before releasing the dying ones
		saved := map[*node]int{}
		for _, r := range rd {
			saved[r] = lw.reg[r]
		}
		R = func(x *node) int { return saved[x] }
		for _, d := range dying {
			lw.release(d)
		}
		dst := lw.alloc()
		switch n.op {
		case 'a', 's':
			mn := "VADDSD"
			if n.op == 's' {
				mn = "VSUBSD"
			}
			switch {
			case it.memop != nil:
				other := n.x
				if it.memop == n.x {
					other = n.y
				}
				lw.emit("%s %s, X%d, X%d", mn, it.memop.mem, R(other), dst)
			default:
				lw.emit("%s X%d, X%d, X%d", mn, R(n.y), R(n.x), dst)
			}
			lw.count["addsub"]++
		case 'm':
			lw.emit("VMULSD %s, X%d, X%d", lw.cname(n.c), R(n.x), dst)
			lw.count["mul"]++
		case 'M':
			lw.emit("VMULSD X%d, X%d, X%d", R(n.y), R(n.x), dst)
			lw.count["mul"]++
		case 'n':
			lw.negMask = true
			lw.emit("VXORPD d23_sign<>(SB), X%d, X%d", R(n.x), dst)
			lw.count["neg"]++
		default:
			panic("op " + string(n.op))
		}
		lw.reg[n] = dst
		finish()
	}
	return lw, true
}

func liveOrder(b *builder, dfs bool) []*node {
	var order []*node
	seen := map[*node]bool{}
	var visit func(n *node)
	visit = func(n *node) {
		if seen[n] || n == nil {
			return
		}
		seen[n] = true
		visit(n.x)
		visit(n.y)
		order = append(order, n)
	}
	if dfs {
		for _, s := range b.outs {
			visit(s)
		}
		return order
	}
	live := map[*node]bool{}
	var mark func(n *node)
	mark = func(n *node) {
		if n == nil || live[n] {
			return
		}
		live[n] = true
		mark(n.x)
		mark(n.y)
	}
	for _, s := range b.outs {
		mark(s)
	}
	for _, n := range b.nodes {
		if live[n] {
			order = append(order, n)
		}
	}
	return order
}

// strip drops the loads from an order (they are emitted on demand).
func strip(order []*node) []*node {
	var o []*node
	for _, n := range order {
		if n.op != 'l' {
			o = append(o, n)
		}
	}
	return o
}

func gen(name string, n int, fwd bool) (text string, consts map[uint64]bool, negMask bool, summary string) {
	b := newBuilder()
	if fwd {
		b.forward(n)
	} else {
		b.inverse(n)
	}
	type cand struct {
		lw    *lowering
		label string
	}
	var best *lowering
	var bestLabel string
	for _, dfs := range []bool{false, true} {
		order := strip(liveOrder(b, dfs))
		items := plan(b, order)
		lw, _ := lower(b, items)
		label := "creation"
		if dfs {
			label = "dfs"
		}
		if best == nil || lw.spills+lw.reloads < best.spills+best.reloads || (lw.spills+lw.reloads == best.spills+best.reloads && lw.ops < best.ops) {
			best, bestLabel = lw, label
		}
	}
	lw := best
	var sb strings.Builder
	frame := 8 * lw.nslot
	if frame%16 != 0 {
		frame += 16 - frame%16
	}
	args := 16
	flags := "NOSPLIT"
	if frame > 256 {
		flags = "0"
	}
	if !fwd {
		args = 24
	}
	sig := "dst, src *float64"
	if !fwd {
		sig += ", a0 float64"
	}
	fmt.Fprintf(&sb, "\n// func %s(%s)\nTEXT ·%s(SB), %s, $%d-%d\n", name, sig, name, flags, frame, args)
	sb.WriteString("\tMOVQ dst+0(FP), DI\n\tMOVQ src+8(FP), SI\n")
	sb.WriteString(lw.sb.String())
	sb.WriteString("\tRET\n")
	summary = fmt.Sprintf("%s: %d instructions (%s order), %d fma, %d add/sub, %d mul, %d neg, %d copies, %d spills, %d reloads, frame %d",
		name, lw.ops, bestLabel, lw.count["fma"], lw.count["addsub"], lw.count["mul"], lw.count["neg"], lw.count["copy"], lw.spills, lw.reloads, frame)
	return sb.String(), lw.consts, lw.negMask, summary
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "-check" {
		ok := true
		for _, n := range []int{32, 64, 128} {
			ok = check(n) && ok
		}
		if !ok {
			os.Exit(1)
		}
		return
	}
	dir := "internal/kernels"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	// Nothing is written unless the dataflow graphs of the kernels' sizes agree
	// with the naive DCT.
	for _, n := range []int{32, 64} {
		if !check(n) {
			fmt.Fprintf(os.Stderr, "gendct23: the graph of n=%d fails its check, nothing written\n", n)
			os.Exit(1)
		}
	}
	var body strings.Builder
	consts := map[uint64]bool{}
	neg := false
	for _, n := range []int{32, 64} {
		for _, fwd := range []bool{true, false} {
			// exported: a function declared without a body has no statement for the
			// coverage gate, so the kernels package needs no Go wrapper that only an
			// AVX-512 CPU would run
			name := fmt.Sprintf("Dct3x%d", n)
			if fwd {
				name = fmt.Sprintf("Dct2x%d", n)
			}
			t, c, nm, sum := gen(name, n, fwd)
			body.WriteString(t)
			for k := range c {
				consts[k] = true
			}
			neg = neg || nm
			fmt.Fprintln(os.Stderr, sum)
		}
	}
	var hdr strings.Builder
	hdr.WriteString("// Code generated by internal/gendct23/main.go; DO NOT EDIT.\n\n//go:build amd64\n\n#include \"textflag.h\"\n\n")
	var keys []uint64
	for k := range consts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, k := range keys {
		fmt.Fprintf(&hdr, "DATA d23_%016x<>+0(SB)/8, $0x%016x\nGLOBL d23_%016x<>(SB), RODATA|NOPTR, $8\n", k, k, k)
	}
	if neg {
		hdr.WriteString("DATA d23_sign<>+0(SB)/8, $0x8000000000000000\nDATA d23_sign<>+8(SB)/8, $0x8000000000000000\nGLOBL d23_sign<>(SB), RODATA|NOPTR, $16\n")
	}
	src := hdr.String() + body.String()
	if err := os.WriteFile(filepath.Join(dir, "dct23_amd64.s"), []byte(src), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_ = format.Source
}
