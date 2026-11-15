//go:build ignore

// Command gendct4 writes the fused DCT-IV of length 64 (go-fft/fft#132):
//
//	dct4_64_gen.go          scalar Go, one straight-line function per variant
//	dct4_64_arm64.s         the pair kernels, NEON, two transforms in the two lanes
//	dct4_64_gen_arm64.go    the constant table the pair kernels read
//
// Both are built from one program: pre-twiddle, a 32-point split-radix complex
// FFT, post-twiddle, as a list of single-assignment instructions (ld, add, sub,
// mul, dot, mulf, st2). The scalar printer turns each into a Go statement; the
// NEON printer maps each to vector instructions on a 2-lane register, with a
// register allocator that spills to a caller-provided buffer, since the 64
// live values of a 32-point FFT do not fit the 32 vector registers.
//
// Run with `go generate` from the package root (see dct4_64.go).
package main

import (
	"bytes"
	"fmt"
	"go/format"
	"math"
	"math/big"
	"os"
	"strconv"
)

const n = 64
const h = n / 2

// ---- IR ----

type op int

const (
	opLd   op = iota // d = in[i]
	opAdd            // d = a + b
	opSub            // d = a - b
	opMul            // d = a * w[k1]
	opDot            // d = a*w[k1] + b*w[k2]
	opMulf           // d = a * f
	opSt2            // out[i] = a, out[i+1] = b
)

type ins struct {
	op        op
	d, a, b   int
	k1, k2, i int
}

// r is a real value: a value id and a pending sign, so that negations cost
// nothing and are absorbed by the next add, subtract or constant.
type r struct {
	id  int
	neg bool
}

func (x r) negated() r { return r{x.id, !x.neg} }

type cx struct{ re, im r }

type prog struct {
	code []ins
	nv   int
}

var (
	consts   []float64
	constIdx = map[float64]int{}
)

func konst(c float64) int {
	i, ok := constIdx[c]
	if !ok {
		i = len(consts)
		consts = append(consts, c)
		constIdx[c] = i
	}
	return i
}

func (p *prog) def(i ins) r {
	i.d = p.nv
	p.nv++
	p.code = append(p.code, i)
	return r{i.d, false}
}

func (p *prog) add(x, y r) r {
	switch {
	case !x.neg && !y.neg:
		return p.def(ins{op: opAdd, a: x.id, b: y.id})
	case x.neg && y.neg:
		v := p.def(ins{op: opAdd, a: x.id, b: y.id})
		return v.negated()
	case x.neg:
		return p.def(ins{op: opSub, a: y.id, b: x.id})
	}
	return p.def(ins{op: opSub, a: x.id, b: y.id})
}

func (p *prog) sub(x, y r) r { return p.add(x, y.negated()) }

func (p *prog) cadd(a, b cx) cx { return cx{p.add(a.re, b.re), p.add(a.im, b.im)} }
func (p *prog) csub(a, b cx) cx { return cx{p.sub(a.re, b.re), p.sub(a.im, b.im)} }

func signed(c float64, neg bool) float64 {
	if neg {
		return -c
	}
	return c
}

func (p *prog) mulk(x r, c float64) r {
	return p.def(ins{op: opMul, a: x.id, k1: konst(signed(c, x.neg))})
}

// dot is x*c1 + y*c2 with the signs of x and y folded into the constants.
func (p *prog) dot(x r, c1 float64, y r, c2 float64) r {
	return p.def(ins{op: opDot, a: x.id, b: y.id, k1: konst(signed(c1, x.neg)), k2: konst(signed(c2, y.neg))})
}

// tw is exp(2πi·num/den) with the multiples of π/4 exact.
func tw(num, den int) complex128 {
	num %= den
	if num < 0 {
		num += den
	}
	if (8*num)%den == 0 {
		o := 8 * num / den
		q := math.Sqrt2 / 2
		return [...]complex128{1, complex(q, q), 1i, complex(-q, q), -1, complex(-q, -q), -1i, complex(q, -q)}[o]
	}
	c, s := bigCosSin(num, den)
	return complex(c, s)
}

// bigPi is π to 400 bits (Machin: 16·atan(1/5) - 4·atan(1/239)).
func bigPi() *big.Float {
	const prec = 400
	atanInv := func(k int64) *big.Float {
		x := new(big.Float).SetPrec(prec).Quo(big.NewFloat(1).SetPrec(prec), new(big.Float).SetPrec(prec).SetInt64(k))
		x2 := new(big.Float).SetPrec(prec).Mul(x, x)
		sum := new(big.Float).SetPrec(prec)
		term := new(big.Float).SetPrec(prec).Set(x)
		for i := int64(0); i < 400; i++ {
			t := new(big.Float).SetPrec(prec).Quo(term, new(big.Float).SetPrec(prec).SetInt64(2*i+1))
			if i%2 == 0 {
				sum.Add(sum, t)
			} else {
				sum.Sub(sum, t)
			}
			term.Mul(term, x2)
		}
		return sum
	}
	a := new(big.Float).SetPrec(prec).Mul(big.NewFloat(16).SetPrec(prec), atanInv(5))
	b := new(big.Float).SetPrec(prec).Mul(big.NewFloat(4).SetPrec(prec), atanInv(239))
	return a.Sub(a, b)
}

// bigCosSin is cos and sin of 2π·num/den, correctly rounded to float64: the
// Taylor series at 400 bits, so the only rounding is the last. (math.Sincos of a
// float64 2πk/n carries the rounding of the angle into the constant.)
func bigCosSin(num, den int) (c, s float64) {
	const prec = 400
	x := new(big.Float).SetPrec(prec).Mul(big.NewFloat(2).SetPrec(prec), bigPi())
	x.Mul(x, new(big.Float).SetPrec(prec).SetInt64(int64(num)))
	x.Quo(x, new(big.Float).SetPrec(prec).SetInt64(int64(den)))
	x2 := new(big.Float).SetPrec(prec).Mul(x, x)
	cs := new(big.Float).SetPrec(prec)
	sn := new(big.Float).SetPrec(prec)
	tc := new(big.Float).SetPrec(prec).SetInt64(1) // x^(2k)/(2k)!
	ts := new(big.Float).SetPrec(prec).Set(x)      // x^(2k+1)/(2k+1)!
	for k := int64(0); k < 120; k++ {
		if k%2 == 0 {
			cs.Add(cs, tc)
			sn.Add(sn, ts)
		} else {
			cs.Sub(cs, tc)
			sn.Sub(sn, ts)
		}
		tc.Mul(tc, x2)
		tc.Quo(tc, new(big.Float).SetPrec(prec).SetInt64((2*k+1)*(2*k+2)))
		ts.Mul(ts, x2)
		ts.Quo(ts, new(big.Float).SetPrec(prec).SetInt64((2*k+2)*(2*k+3)))
	}
	c, _ = cs.Float64()
	s, _ = sn.Float64()
	return c, s
}

// mulc returns a*w for a constant w.
func (p *prog) mulc(a cx, w complex128) cx {
	wr, wi := real(w), imag(w)
	switch {
	case wi == 0 && wr == 1:
		return a
	case wi == 0 && wr == -1:
		return cx{a.re.negated(), a.im.negated()}
	case wr == 0 && wi == 1:
		return cx{a.im.negated(), a.re}
	case wr == 0 && wi == -1:
		return cx{a.im, a.re.negated()}
	case wr == wi:
		d := p.sub(a.re, a.im)
		sm := p.add(a.re, a.im)
		return cx{p.mulk(d, wr), p.mulk(sm, wr)}
	case wr == -wi:
		sm := p.add(a.re, a.im)
		d := p.sub(a.im, a.re)
		return cx{p.mulk(sm, wr), p.mulk(d, wr)}
	}
	return cx{p.dot(a.re, wr, a.im, -wi), p.dot(a.re, wi, a.im, wr)}
}

// thunk is a value computed when first needed, so a leaf's load and
// pre-twiddle are emitted next to its use and live ranges stay short.
type thunk func() cx

// fft is the forward split-radix DFT of x. A non-nil sink receives the outputs
// instead of the result, in the order the pair stores want them.
func (p *prog) fft(x []thunk, sink func(k int, v cx)) []cx {
	m := len(x)
	switch m {
	case 1:
		return []cx{x[0]()}
	case 2:
		a, b := x[0](), x[1]()
		return []cx{p.cadd(a, b), p.csub(a, b)}
	}
	var ev, o1, o3 []thunk
	for i := 0; i < m; i += 2 {
		ev = append(ev, x[i])
	}
	for i := 1; i < m; i += 4 {
		o1 = append(o1, x[i])
		o3 = append(o3, x[i+2])
	}
	E := p.fft(ev, nil)
	Z1 := p.fft(o1, nil)
	Z3 := p.fft(o3, nil)
	out := make([]cx, m)
	order := make([]int, 0, m/4)
	for k := 0; k < m/4; k++ {
		order = append(order, k)
	}
	if sink != nil {
		// 0, 7, 1, 6, ...: output k pairs with output m-1-k, which is in
		// the iteration q-1-k, for the pair stores.
		order = order[:0]
		for k := 0; k < m/8; k++ {
			order = append(order, k, m/4-1-k)
		}
	}
	for _, k := range order {
		a := p.mulc(Z1[k], tw(-k, m))
		b := p.mulc(Z3[k], tw(-3*k, m))
		sum := p.cadd(a, b)
		d := p.csub(a, b)
		nid := cx{d.im, d.re.negated()} // -i·d
		o0 := p.cadd(E[k], sum)
		o2 := p.csub(E[k], sum)
		o1 := p.cadd(E[k+m/4], nid)
		o3 := p.csub(E[k+m/4], nid)
		if sink != nil {
			sink(k, o0)
			sink(k+m/4, o1)
			sink(k+m/2, o2)
			sink(k+3*m/4, o3)
			continue
		}
		out[k], out[k+m/2], out[k+m/4], out[k+3*m/4] = o0, o2, o1, o3
	}
	return out
}

// build returns the instruction list of one DCT-IV(64); scaled multiplies every
// output by the run-time f.
func build(scaled bool) *prog {
	p := &prog{}
	x := make([]thunk, h)
	for m := 0; m < h; m++ {
		m := m
		x[m] = func() cx {
			a := p.def(ins{op: opLd, i: 2 * m})
			b := p.def(ins{op: opLd, i: n - 1 - 2*m})
			return p.mulc(cx{a, b}, tw(-m, 2*n))
		}
	}
	type pair struct{ e, o r }
	pend := map[int]pair{}
	sink := func(k int, z cx) {
		pt := tw(-(4*k + 1), 8*n)
		pr, pi := 2*real(pt), 2*imag(pt)
		e := p.dot(z.re, pr, z.im, -pi)  // y[2k]
		o := p.dot(z.re, -pi, z.im, -pr) // y[n-1-2k]
		if scaled {
			e = p.def(ins{op: opMulf, a: e.id})
			o = p.def(ins{op: opMulf, a: o.id})
		}
		q := h - 1 - k
		if pq, ok := pend[q]; ok {
			delete(pend, q)
			lo, hi := k, q
			pl, ph := pair{e, o}, pq
			if lo > hi {
				lo, hi, pl, ph = hi, lo, ph, pl
			}
			// y[2lo], y[2lo+1] = y[n-1-2hi]; y[2hi], y[2hi+1]... see below.
			p.code = append(p.code, ins{op: opSt2, a: pl.e.id, b: ph.o.id, i: 2 * lo})
			p.code = append(p.code, ins{op: opSt2, a: ph.e.id, b: pl.o.id, i: 2 * hi})
			return
		}
		pend[k] = pair{e, o}
	}
	p.fft(x, sink)
	if len(pend) != 0 {
		panic("unpaired outputs")
	}
	return p
}

// ---- scalar Go printer ----

func (p *prog) goFunc(name, kind string, scaled bool) string {
	var b bytes.Buffer
	ty := "float64"
	if kind == "f32" {
		ty = "float32"
	}
	sig := "dst, src []" + ty
	if scaled {
		sig += ", f float64"
		name += "s"
	}
	fmt.Fprintf(&b, "// %s is the unrolled DCT-IV of length 64 (dct4_64.go).\nfunc %s(%s) {\n\t_, _ = dst[63], src[63]\n", name, name, sig)
	v := func(id int) string { return fmt.Sprintf("v%d", id) }
	// Literals, not a table: gc loads a float64 constant with a load from the
	// literal pool, which measured 43 ns for the kernel against 60 ns with the
	// constants in a table read through a register (M4 Max).
	wk := func(k int) string { return "(" + strconv.FormatFloat(consts[k], 'g', -1, 64) + ")" }
	for _, c := range p.code {
		switch c.op {
		case opLd:
			if kind == "f32" {
				fmt.Fprintf(&b, "\t%s := float64(src[%d])\n", v(c.d), c.i)
			} else {
				fmt.Fprintf(&b, "\t%s := src[%d]\n", v(c.d), c.i)
			}
		case opAdd:
			fmt.Fprintf(&b, "\t%s := %s + %s\n", v(c.d), v(c.a), v(c.b))
		case opSub:
			fmt.Fprintf(&b, "\t%s := %s - %s\n", v(c.d), v(c.a), v(c.b))
		case opMul:
			fmt.Fprintf(&b, "\t%s := %s * %s\n", v(c.d), v(c.a), wk(c.k1))
		case opDot:
			fmt.Fprintf(&b, "\t%s := %s*%s + %s*%s\n", v(c.d), v(c.a), wk(c.k1), v(c.b), wk(c.k2))
		case opMulf:
			fmt.Fprintf(&b, "\t%s := %s * f\n", v(c.d), v(c.a))
		case opSt2:
			for j, id := range []int{c.a, c.b} {
				e := v(id)
				if kind == "f32" {
					e = "float32(" + e + ")"
				}
				fmt.Fprintf(&b, "\tdst[%d] = %s\n", c.i+j, e)
			}
		}
	}
	b.WriteString("}\n\n")
	return b.String()
}

// ---- NEON printer ----

const (
	nAlloc   = 26 // V0..V25 hold values
	tmpC1    = 26
	tmpC2    = 27
	tmpMul   = 28 // unused
	regF     = 29
	tmpLo    = 30
	tmpHi    = 31
	noReg    = -1
	slotNone = -1
)

type regalloc struct {
	b      *bytes.Buffer
	reg    []int // value -> register or noReg
	slot   []int // value -> spill slot or slotNone
	holder [nAlloc]int
	uses   [][]int // value -> instruction indices that read it
	pos    int
	nslots int
}

func (ra *regalloc) nextUse(v int) int {
	for _, u := range ra.uses[v] {
		if u > ra.pos {
			return u
		}
	}
	return math.MaxInt
}

// alloc returns a register for v, evicting the value whose next use is farthest,
// never one of pinned.
func (ra *regalloc) alloc(v int, pinned ...int) int {
	best, bestUse := -1, -1
	for r := 0; r < nAlloc; r++ {
		h := ra.holder[r]
		if h < 0 {
			best = r
			bestUse = math.MaxInt
			break
		}
		skip := false
		for _, p := range pinned {
			if p == h {
				skip = true
			}
		}
		if skip {
			continue
		}
		if u := ra.nextUse(h); u > bestUse {
			best, bestUse = r, u
		}
	}
	if best < 0 {
		panic("no register to evict")
	}
	if h := ra.holder[best]; h >= 0 {
		if bestUse != math.MaxInt { // still needed: keep it in memory
			if ra.slot[h] == slotNone {
				ra.slot[h] = ra.nslots
				ra.nslots++
				fmt.Fprintf(ra.b, "\tFMOVQ F%d, %d(R5)\n", best, 16*ra.slot[h])
			}
		}
		ra.reg[h] = noReg
	}
	ra.holder[best] = v
	ra.reg[v] = best
	return best
}

// use makes sure v is in a register (reloading it) and returns it.
func (ra *regalloc) use(v int, pinned ...int) int {
	if r := ra.reg[v]; r != noReg {
		return r
	}
	r := ra.alloc(v, pinned...)
	fmt.Fprintf(ra.b, "\tFMOVQ %d(R5), F%d\n", 16*ra.slot[v], r)
	return r
}

func (ra *regalloc) release(v int) {
	if r := ra.reg[v]; r != noReg && ra.nextUse(v) == math.MaxInt {
		ra.holder[r] = -1
		ra.reg[v] = noReg
	}
}

// The vector table holds one row per distinct pair of constants (c1, c2): a
// kernel loads a row into one register with a single 128-bit load and multiplies
// by its two lanes with FMUL/FMLA by element (the Go assembler has no mnemonic for
// them, so they are WORDs). Loading each constant into both lanes of a register
// took two loads per complex product; that was the kernel's bottleneck, as
// 640 of its 1500 instructions were loads on a core with three load ports.
var (
	pairRows [][2]float64
	pairIdx  = map[[2]float64]int{}
)

func pairRow(c1, c2 float64) int {
	k := [2]float64{c1, c2}
	i, ok := pairIdx[k]
	if !ok {
		i = len(pairRows)
		pairRows = append(pairRows, k)
		pairIdx[k] = i
	}
	return i
}

// elem encodes FMUL (base 0x4FC09000) or FMLA (base 0x4FC01000) by element, 2D:
// Vd = Vn * Vm.D[idx] (+ Vd for FMLA).
func elem(base uint32, d, n, m, idx int) uint32 {
	return base | uint32(m>>4&1)<<20 | uint32(m&15)<<16 | uint32(idx)<<11 | uint32(n)<<5 | uint32(d)
}

// constPairs hands out the two constant registers V26 and V27, alternately, and
// skips a load when a register already holds the row.
type constPairs struct {
	held [2]int
	next int
	init bool
}

func (c *constPairs) load(b *bytes.Buffer, c1, c2 float64) int {
	if !c.init {
		c.held = [2]int{-1, -1}
		c.init = true
	}
	row := pairRow(c1, c2)
	for i, h := range c.held {
		if h == row {
			return tmpC1 + i
		}
	}
	i := c.next
	c.next ^= 1
	c.held[i] = row
	fmt.Fprintf(b, "\tFMOVQ %d(R4), F%d\n", 16*row, tmpC1+i)
	return tmpC1 + i
}

// asmFunc prints one NEON pair kernel and returns it with its spill slot count.
func (p *prog) asmFunc(name, kind string, scaled bool) (string, int) {
	var b bytes.Buffer
	ra := &regalloc{b: &b, reg: make([]int, p.nv), slot: make([]int, p.nv), uses: make([][]int, p.nv)}
	for i := range ra.reg {
		ra.reg[i], ra.slot[i] = noReg, slotNone
	}
	for i := range ra.holder {
		ra.holder[i] = -1
	}
	for i, c := range p.code {
		switch c.op {
		case opAdd, opSub, opDot, opSt2:
			ra.uses[c.a] = append(ra.uses[c.a], i)
			ra.uses[c.b] = append(ra.uses[c.b], i)
		case opMul, opMulf:
			ra.uses[c.a] = append(ra.uses[c.a], i)
		}
	}
	argSize := 40
	if scaled {
		name += "S"
		argSize = 48
	}
	var pro bytes.Buffer
	pro.WriteString("\tMOVD dst0+0(FP), R0\n\tMOVD dst1+8(FP), R1\n\tMOVD src0+16(FP), R2\n\tMOVD src1+24(FP), R3\n\tMOVD w+32(FP), R4\n\tMOVD RSP, R5\n\tADD $16, R5\n")
	if scaled {
		pro.WriteString("\tFMOVD f+40(FP), F29\n\tVDUP V29.D[0], V29.D2\n")
	}
	b.WriteString(pro.String())
	esz := 8
	if kind == "f32" {
		esz = 4
	}
	cp := &constPairs{}
	for i, c := range p.code {
		ra.pos = i
		switch c.op {
		case opLd:
			d := ra.alloc(c.d)
			if kind == "f32" {
				fmt.Fprintf(&b, "\tFMOVS %d(R2), F30\n\tFMOVS %d(R3), F31\n\tFCVTSD F30, F30\n\tFCVTSD F31, F31\n", esz*c.i, esz*c.i)
			} else {
				fmt.Fprintf(&b, "\tFMOVD %d(R2), F30\n\tFMOVD %d(R3), F31\n", esz*c.i, esz*c.i)
			}
			fmt.Fprintf(&b, "\tVZIP1 V31.D2, V30.D2, V%d.D2\n", d)
		case opAdd, opSub, opDot:
			a := ra.use(c.a, c.b)
			bb := ra.use(c.b, c.a)
			d := ra.alloc(c.d, c.a, c.b)
			// a or b may have been evicted by alloc only if not pinned; they are.
			switch c.op {
			case opAdd:
				fmt.Fprintf(&b, "\tVFADD V%d.D2, V%d.D2, V%d.D2\n", bb, a, d)
			case opSub:
				fmt.Fprintf(&b, "\tVFSUB V%d.D2, V%d.D2, V%d.D2\n", bb, a, d)
			default:
				// d = a*w[k1] + b*w[k2], both constants in one register
				// (two lanes of the table row), read by element.
				t := cp.load(&b, consts[c.k1], consts[c.k2])
				fmt.Fprintf(&b, "\tWORD $0x%08X // FMUL V%d.2D, V%d.2D, V%d.D[0]\n", elem(0x4FC09000, d, a, t, 0), d, a, t)
				fmt.Fprintf(&b, "\tWORD $0x%08X // FMLA V%d.2D, V%d.2D, V%d.D[1]\n", elem(0x4FC01000, d, bb, t, 1), d, bb, t)
			}
		case opMul:
			a := ra.use(c.a)
			d := ra.alloc(c.d, c.a)
			t := cp.load(&b, consts[c.k1], consts[c.k1])
			fmt.Fprintf(&b, "\tWORD $0x%08X // FMUL V%d.2D, V%d.2D, V%d.D[0]\n", elem(0x4FC09000, d, a, t, 0), d, a, t)
		case opMulf:
			a := ra.use(c.a)
			d := ra.alloc(c.d, c.a)
			fmt.Fprintf(&b, "\tVFMUL V29.D2, V%d.D2, V%d.D2\n", a, d)
		case opSt2:
			a := ra.use(c.a, c.b)
			bb := ra.use(c.b, c.a)
			fmt.Fprintf(&b, "\tVZIP1 V%d.D2, V%d.D2, V30.D2\n\tVZIP2 V%d.D2, V%d.D2, V31.D2\n", bb, a, bb, a)
			if kind == "f32" {
				fmt.Fprintf(&b, "\tWORD $0x%08X // FCVTN V30.2S, V30.2D\n\tWORD $0x%08X // FCVTN V31.2S, V31.2D\n", 0x0E616800|30<<5|30, 0x0E616800|31<<5|31)
				fmt.Fprintf(&b, "\tFMOVD F30, %d(R0)\n\tFMOVD F31, %d(R1)\n", esz*c.i, esz*c.i)
			} else {
				fmt.Fprintf(&b, "\tFMOVQ F30, %d(R0)\n\tFMOVQ F31, %d(R1)\n", esz*c.i, esz*c.i)
			}
		}
		// free what died here
		switch c.op {
		case opAdd, opSub, opDot, opSt2:
			ra.release(c.a)
			ra.release(c.b)
		case opMul, opMulf:
			ra.release(c.a)
		}
	}
	b.WriteString("\tRET\n\n")
	// The spill slots are the function's own frame, from 16(RSP) up: below the
	// saved link register at the top of the frame, and never zeroed by a caller.
	head := fmt.Sprintf("TEXT ·%s(SB), $%d-%d\n", name, 16+16*ra.nslots+16, argSize)
	return head + b.String(), ra.nslots
}

func main() {
	var goSrc, sSrc bytes.Buffer
	maxSlots := 0
	type variant struct {
		kind   string
		scaled bool
	}
	for _, v := range []variant{{"f64", false}, {"f64", true}, {"f32", false}, {"f32", true}} {
		p := build(v.scaled)
		goSrc.WriteString(p.goFunc("dct4n64"+v.kind, v.kind, v.scaled))
		pairName := "dct4n64Pair" + map[string]string{"f64": "F64", "f32": "F32"}[v.kind]
		s, ns := p.asmFunc(pairName, v.kind, v.scaled)
		sSrc.WriteString(s)
		if ns > maxSlots {
			maxSlots = ns
		}
	}

	write("dct4_64_gen.go", append([]byte("// Code generated by internal/gendct4; DO NOT EDIT.\n\npackage fft\n\n"), goSrc.Bytes()...), true)

	var vt bytes.Buffer
	vt.WriteString("// Code generated by internal/gendct4; DO NOT EDIT.\n\npackage fft\n\n")
	fmt.Fprintf(&vt, "// dct4n64VTab holds the %d distinct pairs of twiddle constants the pair kernels read, one 128-bit row each.\ntype dct4n64VTab [%d][2]float64\n\n", len(pairRows), len(pairRows))
	fmt.Fprintf(&vt, "// dct4n64Spill is the number of 128-bit spill slots a pair kernel uses.\nconst dct4n64Spill = %d\n\n", maxSlots)
	vt.WriteString("var dct4n64V = dct4n64VTab{\n")
	for _, c := range pairRows {
		vt.WriteString("{" + strconv.FormatFloat(c[0], 'g', -1, 64) + ", " + strconv.FormatFloat(c[1], 'g', -1, 64) + "},\n")
	}
	vt.WriteString("}\n")
	write("dct4_64_gen_arm64.go", vt.Bytes(), true)

	asm := "// Code generated by internal/gendct4; DO NOT EDIT.\n\n#include \"textflag.h\"\n\n" + sSrc.String()
	write("dct4_64_arm64.s", []byte(asm), false)
}

func write(name string, b []byte, gofmt bool) {
	if gofmt {
		var err error
		if b, err = format.Source(b); err != nil {
			fmt.Fprintln(os.Stderr, name, err)
			os.Exit(1)
		}
	}
	if err := os.WriteFile(name, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
