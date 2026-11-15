//go:build ignore

// The amd64 pair kernels of the fused DCT-IV of length 64 (go-fft/fft#132).
//
// Layout. One 256-bit register holds one complex value of BOTH transforms: lanes
// (re, im, re', im') of the first and the second stream. A complex add or subtract
// is one instruction for the two transforms; a twiddle product a*w is
//
//	VPERMILPD $5       swap re and im of each stream
//	VMULPD  (wr,wr,wr,wr)
//	VFMADD231PD (-wi,wi,-wi,wi)
//
// that is three instructions for both transforms (the real-lane layout of the
// arm64 kernel needs four, and twice the registers). The post-twiddle uses the
// constants (pr,-pr,pr,-pr) and (-pi,-pi,-pi,-pi), which deliver y[2k] and
// y[n-1-2k] directly, the second already negated; the pair stores pick the
// even output of one k and the odd output of its partner with a blend.
//
// The kernels need AVX2 and FMA3 (kernels.UseFMA) and use no AVX-512: they have
// two transforms to fill four lanes and a 512-bit register would hold eight.
package main

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
)

type cop int

const (
	cLd   cop = iota // d = (in0[2m], in0[n-1-2m], in1[2m], in1[n-1-2m]): i = m
	cAdd             // d = a + b
	cSub             // d = a - b
	cMul             // d = a*rows[k1] + swap(a)*rows[k2]
	cNegI            // d = -i·a = (a.im, -a.re) per stream
	cMulf            // d = a * f
	cSt2             // the two pair stores of Y_a (lower k) and Y_b (its partner): i, j are the dst indices
)

type cins struct {
	op      cop
	d, a, b int
	k1, k2  int
	i, j    int
}

type cprog struct {
	code []cins
	nv   int
}

var (
	yRows   [][4]float64
	yRowIdx = map[[4]float64]int{}
)

func yrow(r [4]float64) int {
	i, ok := yRowIdx[r]
	if !ok {
		i = len(yRows)
		yRows = append(yRows, r)
		yRowIdx[r] = i
	}
	return i
}

func (p *cprog) def(i cins) int {
	i.d = p.nv
	p.nv++
	p.code = append(p.code, i)
	return i.d
}

func (p *cprog) add(a, b int) int { return p.def(cins{op: cAdd, a: a, b: b}) }
func (p *cprog) sub(a, b int) int { return p.def(cins{op: cSub, a: a, b: b}) }

// mul is a*w for a constant w.
func (p *cprog) mul(a int, w complex128) int {
	wr, wi := real(w), imag(w)
	switch {
	case wi == 0 && wr == 1:
		return a
	case wr == 0 && wi == -1:
		return p.def(cins{op: cNegI, a: a, k1: yrow([4]float64{0, math.Copysign(0, -1), 0, math.Copysign(0, -1)})})
	}
	return p.def(cins{op: cMul, a: a,
		k1: yrow([4]float64{wr, wr, wr, wr}),
		k2: yrow([4]float64{-wi, wi, -wi, wi})})
}

type cthunk func() int

// fft is the forward split-radix DFT of x (see main.go): the same recursion on
// complex values.
func (p *cprog) fft(x []cthunk, sink func(k, v int)) []int {
	m := len(x)
	switch m {
	case 1:
		return []int{x[0]()}
	case 2:
		a, b := x[0](), x[1]()
		return []int{p.add(a, b), p.sub(a, b)}
	}
	var ev, o1, o3 []cthunk
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
	out := make([]int, m)
	order := make([]int, 0, m/4)
	for k := 0; k < m/4; k++ {
		order = append(order, k)
	}
	if sink != nil {
		order = order[:0]
		for k := 0; k < m/8; k++ {
			order = append(order, k, m/4-1-k)
		}
	}
	for _, k := range order {
		a := p.mul(Z1[k], tw(-k, m))
		b := p.mul(Z3[k], tw(-3*k, m))
		sum := p.add(a, b)
		d := p.sub(a, b)
		nid := p.def(cins{op: cNegI, a: d, k1: yrow([4]float64{0, math.Copysign(0, -1), 0, math.Copysign(0, -1)})})
		o0 := p.add(E[k], sum)
		o2 := p.sub(E[k], sum)
		o1v := p.add(E[k+m/4], nid)
		o3v := p.sub(E[k+m/4], nid)
		if sink != nil {
			sink(k, o0)
			sink(k+m/4, o1v)
			sink(k+m/2, o2)
			sink(k+3*m/4, o3v)
			continue
		}
		out[k], out[k+m/2], out[k+m/4], out[k+3*m/4] = o0, o2, o1v, o3v
	}
	return out
}

func cbuild(scaled bool) *cprog {
	p := &cprog{}
	x := make([]cthunk, h)
	for m := 0; m < h; m++ {
		m := m
		x[m] = func() int {
			return p.mul(p.def(cins{op: cLd, i: m}), tw(-m, 2*n))
		}
	}
	pend := map[int]int{}
	sink := func(k, z int) {
		pt := tw(-(4*k + 1), 8*n)
		pr, pi := 2*real(pt), 2*imag(pt)
		y := p.def(cins{op: cMul, a: z,
			k1: yrow([4]float64{pr, -pr, pr, -pr}),
			k2: yrow([4]float64{-pi, -pi, -pi, -pi})})
		if scaled {
			y = p.def(cins{op: cMulf, a: y})
		}
		q := h - 1 - k
		if yq, ok := pend[q]; ok {
			delete(pend, q)
			lo, hi, ylo, yhi := k, q, y, yq
			if lo > hi {
				lo, hi, ylo, yhi = hi, lo, yhi, ylo
			}
			p.code = append(p.code, cins{op: cSt2, a: ylo, b: yhi, i: 2 * lo, j: 2 * hi})
			return
		}
		pend[k] = y
	}
	p.fft(x, sink)
	if len(pend) != 0 {
		panic("unpaired outputs")
	}
	return p
}

// ---- allocator and printer ----

const (
	yAlloc = 14 // Y0..Y13 hold values; Y14 holds f, Y15 is the scratch
	yF     = 14
	yTmp   = 15
)

// fmaEvery: every fmaEvery-th eligible add or subtract becomes an FMA (2: half of
// them, 1: all, 0: none). Zero here: measured on a Zen 4 (Ryzen 9 7950X, one
// pinned core, 10 interleaved rounds against the kernel without the rewrite), half
// of them gave +2.6% (f64 pair) and +0.6% (Mixed pair), all of them +6.7% and
// +2.1%: the 256-point complex kernel of go-fft/fft#162 gains 6.4% from the same
// rewrite, this one, whose adds are a third of its instructions and whose
// critical path is the twiddle products, does not.
var fmaEvery = 0

type yalloc struct {
	fmaCount int
	b        *bytes.Buffer
	reg      []int
	slot     []int
	holder   [yAlloc]int
	uses     [][]int
	pos      int
	nslots   int
}

func (ra *yalloc) nextUse(v int) int {
	for _, u := range ra.uses[v] {
		if u > ra.pos {
			return u
		}
	}
	return math.MaxInt
}

func (ra *yalloc) alloc(v int, pinned ...int) int {
	best, bestUse := -1, -1
	for r := 0; r < yAlloc; r++ {
		h := ra.holder[r]
		if h < 0 {
			best, bestUse = r, math.MaxInt
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
		if bestUse != math.MaxInt && ra.slot[h] < 0 {
			ra.slot[h] = ra.nslots
			ra.nslots++
			fmt.Fprintf(ra.b, "\tVMOVUPD Y%d, %d(SP)\n", best, 32*ra.slot[h])
		}
		ra.reg[h] = -1
	}
	ra.holder[best] = v
	ra.reg[v] = best
	return best
}

func (ra *yalloc) use(v int, pinned ...int) int {
	if r := ra.reg[v]; r >= 0 {
		return r
	}
	r := ra.alloc(v, pinned...)
	fmt.Fprintf(ra.b, "\tVMOVUPD %d(SP), Y%d\n", 32*ra.slot[v], r)
	return r
}

func (ra *yalloc) release(v int) {
	if r := ra.reg[v]; r >= 0 && ra.nextUse(v) == math.MaxInt {
		ra.holder[r] = -1
		ra.reg[v] = -1
	}
}

// dying reports whether v has no use after the current instruction.
func (ra *yalloc) dying(v int) bool { return ra.nextUse(v) == math.MaxInt }

// take gives d the register of a when a dies here (the instruction reads a
// before it writes d), else a free or evicted one.
func (ra *yalloc) take(d, a int, pinned ...int) int {
	if r := ra.reg[a]; r >= 0 && ra.dying(a) {
		ra.holder[r] = d
		ra.reg[a] = -1
		ra.reg[d] = r
		return r
	}
	return ra.alloc(d, pinned...)
}

func (p *cprog) asmFunc(name, kind string, scaled bool) (string, int) {
	var b bytes.Buffer
	ra := &yalloc{b: &b, reg: make([]int, p.nv), slot: make([]int, p.nv), uses: make([][]int, p.nv)}
	for i := range ra.reg {
		ra.reg[i], ra.slot[i] = -1, -1
	}
	for i := range ra.holder {
		ra.holder[i] = -1
	}
	for i, c := range p.code {
		switch c.op {
		case cAdd, cSub, cSt2:
			ra.uses[c.a] = append(ra.uses[c.a], i)
			ra.uses[c.b] = append(ra.uses[c.b], i)
		case cMul, cNegI, cMulf:
			ra.uses[c.a] = append(ra.uses[c.a], i)
		}
	}
	argSize := 40
	if scaled {
		name += "S"
		argSize = 48
	}
	b.WriteString("\tMOVQ dst0+0(FP), DI\n\tMOVQ dst1+8(FP), SI\n\tMOVQ src0+16(FP), AX\n\tMOVQ src1+24(FP), BX\n\tMOVQ w+32(FP), DX\n")
	if scaled {
		b.WriteString("\tVBROADCASTSD f+40(FP), Y14\n")
	}
	esz := 8
	if kind == "f32" {
		esz = 4
	}
	row := func(k int) string { return fmt.Sprintf("%d(DX)", 32*k) }
	for i, c := range p.code {
		ra.pos = i
		switch c.op {
		case cLd:
			d := ra.alloc(c.d)
			ia, ib := 2*c.i, n-1-2*c.i
			if kind == "f32" {
				// two floats into elements 0 and 1, then both to double
				fmt.Fprintf(&b, "\tVMOVSS %d(AX), X%d\n\tVINSERTPS $0x10, %d(AX), X%d, X%d\n\tVCVTPS2PD X%d, X%d\n", esz*ia, d, esz*ib, d, d, d, d)
				fmt.Fprintf(&b, "\tVMOVSS %d(BX), X%d\n\tVINSERTPS $0x10, %d(BX), X%d, X%d\n\tVCVTPS2PD X%d, X%d\n", esz*ia, yTmp, esz*ib, yTmp, yTmp, yTmp, yTmp)
			} else {
				fmt.Fprintf(&b, "\tVMOVSD %d(AX), X%d\n\tVMOVHPD %d(AX), X%d, X%d\n", esz*ia, d, esz*ib, d, d)
				fmt.Fprintf(&b, "\tVMOVSD %d(BX), X%d\n\tVMOVHPD %d(BX), X%d, X%d\n", esz*ia, yTmp, esz*ib, yTmp, yTmp)
			}
			fmt.Fprintf(&b, "\tVINSERTF128 $1, X%d, Y%d, Y%d\n", yTmp, d, d)
		case cAdd, cSub:
			a, bb := c.a, c.b
			if c.op == cAdd && ra.reg[a] < 0 && ra.reg[bb] >= 0 {
				a, bb = bb, a // commutative: the operand in a register goes first
			}
			ya := ra.use(a, bb)
			var bt string
			if r := ra.reg[bb]; r >= 0 {
				bt = fmt.Sprintf("Y%d", r)
			} else {
				bt = fmt.Sprintf("%d(SP)", 32*ra.slot[bb])
			}
			// The FMA rewrite (go-fft/fft#162): a + b is fma(b, 1, a) and a - b is
			// fma(b, -1, a), the same bits, issued on the FMA pipes instead of the
			// add pipes. Zen 4 issues add and FMA on different pipes, and about
			// half the adds moved gave 6.4% on the 256-point kernel (all of them,
			// less). Only where a dies here (the sum takes its register) and b is in
			// a register.
			if fmaEvery > 0 && ra.dying(a) && ra.reg[bb] >= 0 && bb != a {
				ra.fmaCount++
				if ra.fmaCount%fmaEvery == 0 {
					one := yrow([4]float64{1, 1, 1, 1})
					if c.op == cSub {
						one = yrow([4]float64{-1, -1, -1, -1})
					}
					d := ra.take(c.d, a, a, bb)
					fmt.Fprintf(&b, "\tVFMADD231PD %s, Y%d, Y%d\n", row(one), ra.reg[bb], d)
					break
				}
			}
			d := ra.take(c.d, a, a, bb)
			mn := "VADDPD"
			if c.op == cSub {
				mn = "VSUBPD"
			}
			fmt.Fprintf(&b, "\t%s %s, Y%d, Y%d\n", mn, bt, ya, d)
		case cMul:
			ya := ra.use(c.a)
			fmt.Fprintf(&b, "\tVPERMILPD $5, Y%d, Y%d\n", ya, yTmp)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVMULPD %s, Y%d, Y%d\n\tVFMADD231PD %s, Y%d, Y%d\n", row(c.k1), ya, d, row(c.k2), yTmp, d)
		case cNegI:
			ya := ra.use(c.a)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVPERMILPD $5, Y%d, Y%d\n\tVXORPD %s, Y%d, Y%d\n", ya, d, row(c.k1), d, d)
		case cMulf:
			ya := ra.use(c.a)
			d := ra.take(c.d, c.a, c.a)
			fmt.Fprintf(&b, "\tVMULPD Y%d, Y%d, Y%d\n", yF, ya, d)
		case cSt2:
			ya := ra.use(c.a, c.b)
			yb := ra.use(c.b, c.a)
			store := func(idx int) {
				if kind == "f32" {
					fmt.Fprintf(&b, "\tVCVTPD2PSY Y%d, X%d\n\tVMOVQ X%d, %d(DI)\n\tVMOVHPD X%d, %d(SI)\n", yTmp, yTmp, yTmp, esz*idx, yTmp, esz*idx)
				} else {
					fmt.Fprintf(&b, "\tVMOVUPD X%d, %d(DI)\n\tVEXTRACTF128 $1, Y%d, %d(SI)\n", yTmp, esz*idx, yTmp, esz*idx)
				}
			}
			// lanes 0,2 from the first register, 1,3 from the second
			fmt.Fprintf(&b, "\tVBLENDPD $10, Y%d, Y%d, Y%d\n", yb, ya, yTmp)
			store(c.i)
			fmt.Fprintf(&b, "\tVBLENDPD $10, Y%d, Y%d, Y%d\n", ya, yb, yTmp)
			store(c.j)
		}
		switch c.op {
		case cAdd, cSub, cSt2:
			ra.release(c.a)
			ra.release(c.b)
		case cMul, cNegI, cMulf:
			ra.release(c.a)
		}
	}
	b.WriteString("\tVZEROUPPER\n\tRET\n\n")
	head := fmt.Sprintf("TEXT ·%s(SB), 0, $%d-%d\n", name, 32*ra.nslots+32, argSize)
	return head + b.String(), ra.nslots
}

// amd64Gen writes dct4_64_amd64.s and dct4_64_gen_amd64.go.
func amd64Gen() {
	var sSrc bytes.Buffer
	for _, v := range []struct {
		kind   string
		scaled bool
	}{{"f64", false}, {"f64", true}, {"f32", false}, {"f32", true}} {
		p := cbuild(v.scaled)
		name := "dct4n64YPair" + map[string]string{"f64": "F64", "f32": "F32"}[v.kind]
		s, _ := p.asmFunc(name, v.kind, v.scaled)
		sSrc.WriteString(s)
	}
	var vt bytes.Buffer
	vt.WriteString("// Code generated by internal/gendct4; DO NOT EDIT.\n\npackage fft\n\n")
	fmt.Fprintf(&vt, "// dct4n64YTab holds the %d distinct 256-bit constant rows the amd64 pair kernels read.\ntype dct4n64YTab [%d][4]float64\n\n", len(yRows), len(yRows))
	vt.WriteString("var dct4n64Y = dct4n64YTab{\n")
	for _, r := range yRows {
		var s [4]string
		for i, x := range r {
			s[i] = strconv.FormatFloat(x, 'g', -1, 64)
			if x == 0 && math.Signbit(x) {
				s[i] = "math.Copysign(0, -1)"
			}
		}
		vt.WriteString("{" + s[0] + ", " + s[1] + ", " + s[2] + ", " + s[3] + "},\n")
	}
	vt.WriteString("}\n")
	write("dct4_64_gen_amd64.go", bytes.Replace(vt.Bytes(), []byte("package fft\n"), []byte("package fft\n\nimport \"math\"\n"), 1), true)
	write("dct4_64_amd64.s", []byte("// Code generated by internal/gendct4; DO NOT EDIT.\n\n#include \"textflag.h\"\n\n"+sSrc.String()), false)
}
