//go:build ignore

// Command variants writes N schedules of the fused 256-point kernel into one
// assembly file (go-fft/fft#162, proposal 1's falsifying experiment). Every
// variant computes the same dataflow as internal/genfused256, so each result bit
// is the same; they differ only in:
//
//   - the instruction ORDER: a list schedule of the dependency DAG with priority
//     = program position + N(0, sigma), under a 32-register cap;
//   - the REGISTER ASSIGNMENT policy (lowest free, random free, oldest freed);
//   - the share of add/sub rewritten as an FMA with a constant 1 or -1
//     (fma(a, 1, b) = a+b, fma(b, -1, a) = a-b: one rounding of the exact sum, so
//     the same bits), done where one source dies at the instruction;
//   - the TRANSPOSE: shuffles in registers (VSHUFF64X2) or a store of whole rows
//     and 128-bit reloads (VMOVUPD xmm + VINSERTF32X4 from memory).
//
// usage: go run internal/genfused256/variants/main.go -n 200 -seed 1 -out internal/schedexp
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/go-fft/fft/internal/gentwiddle"
)

type kind int

const (
	kLoad kind = iota
	kStore
	kAdd
	kSub
	kPerm
	kFmaB // mn.BCST const, x, acc->dst
	kFmaT // mn tablemem, x, acc->dst
	kMulT
	kMulB
	kXorB
	kShuf
	kLoadX
	kIns
)

type ins struct {
	k    kind
	mn   string
	imm  string
	mem  string // memory operand text
	src  []int
	acc  int // RMW accumulator source, -1 none
	dst  int // -1 for stores
	flag bool
	// memory dependency info
	base   string
	lo, hi int
	load   bool
	store  bool
	// filled by the scheduler
	rewrite int // -1 none, else the vreg whose register the FMA overwrites
}

var (
	ops    []ins
	nv     int
	consts = map[string]float64{}
	inv    bool
	memTr  bool
	pFlag  float64
	rng    *rand.Rand
)

func newV() int { nv++; return nv - 1 }

func cname(v float64) string {
	n := fmt.Sprintf("v256_%016x<>", math.Float64bits(v))
	consts[n] = v
	return n
}

func add1(o ins) int {
	o.rewrite = -1
	if o.k != kStore {
		o.dst = newV()
	} else {
		o.dst = -1
	}
	ops = append(ops, o)
	return o.dst
}

func ld(base string, off int) int {
	return add1(ins{k: kLoad, mem: fmt.Sprintf("%d(%s)", off, base), acc: -1, base: base, lo: off, hi: off + 64, load: true})
}
func st(v int, base string, off int) {
	add1(ins{k: kStore, mem: fmt.Sprintf("%d(%s)", off, base), src: []int{v}, acc: -1, base: base, lo: off, hi: off + 64, store: true})
}
func addv(a, b int) int {
	return add1(ins{k: kAdd, src: []int{a, b}, acc: -1, flag: rng.Float64() < pFlag})
}
func subv(a, b int) int {
	return add1(ins{k: kSub, src: []int{a, b}, acc: -1, flag: rng.Float64() < pFlag})
}
func perm(a int) int { return add1(ins{k: kPerm, src: []int{a}, acc: -1}) }
func fmaB(mn string, c float64, x, acc int) int {
	return add1(ins{k: kFmaB, mn: mn, mem: cname(c), src: []int{x}, acc: acc})
}

func cs(num, den int) (c, s float64) {
	num %= den
	if (4*num)%den == 0 {
		switch 4 * num / den {
		case 0:
			return 1, 0
		case 1:
			return 0, 1
		case 2:
			return -1, 0
		}
		return 0, -1
	}
	return gentwiddle.CosSin(num, den)
}

func radix4(a0, a1, a2, a3 int) (o0, o1, o2, o3 int) {
	t0 := addv(a0, a2)
	t1 := subv(a0, a2)
	t2 := addv(a1, a3)
	t3 := subv(a1, a3)
	o0 = addv(t0, t2)
	o2 = subv(t0, t2)
	sw1 := perm(t3)
	sw3 := perm(t3)
	m1, m3 := "VFMSUBADD231PD", "VFMADDSUB231PD"
	if inv {
		m1, m3 = m3, m1
	}
	o1 = fmaB(m1, 1, t1, sw1)
	o3 = fmaB(m3, 1, t1, sw3)
	return
}

func cmulConst(r, m int) int {
	m &= 15
	if m == 0 {
		return r
	}
	c, s := cs(m, 16)
	if !inv {
		s = -s
	}
	if m == 8 {
		return add1(ins{k: kXorB, mn: "VXORPD", mem: cname(math.Copysign(0, -1)), src: []int{r}, acc: -1})
	}
	t := perm(r)
	t2 := add1(ins{k: kMulB, mn: "VMULPD", mem: cname(s), src: []int{t}, acc: -1})
	return fmaB("VFMADDSUB231PD", c, r, t2)
}

func dft16(in [16]int) (out [16]int) {
	var u [4][4]int
	for j2 := 0; j2 < 4; j2++ {
		u[0][j2], u[1][j2], u[2][j2], u[3][j2] = radix4(in[j2], in[4+j2], in[8+j2], in[12+j2])
	}
	for k1 := 1; k1 < 4; k1++ {
		for j2 := 1; j2 < 4; j2++ {
			u[k1][j2] = cmulConst(u[k1][j2], j2*k1)
		}
	}
	for k1 := 0; k1 < 4; k1++ {
		out[k1], out[k1+4], out[k1+8], out[k1+12] = radix4(u[k1][0], u[k1][1], u[k1][2], u[k1][3])
	}
	return
}

func shuf(imm string, a, b int) int {
	return add1(ins{k: kShuf, imm: imm, src: []int{a, b}, acc: -1})
}

func transpose4(a0, a1, a2, a3 int) (o0, o1, o2, o3 int) {
	t0 := shuf("$0x88", a0, a1)
	t1 := shuf("$0xDD", a0, a1)
	t2 := shuf("$0x88", a2, a3)
	t3 := shuf("$0xDD", a2, a3)
	o0 = shuf("$0x88", t0, t2)
	o1 = shuf("$0x88", t1, t3)
	o2 = shuf("$0xDD", t0, t2)
	o3 = shuf("$0xDD", t1, t3)
	return
}

// build makes the whole forward transform as SSA over virtual registers.
func build() {
	ops, nv = nil, 0
	for g := 0; g < 4; g++ {
		var in [16]int
		for a := 0; a < 16; a++ {
			in[a] = ld("SI", (16*a+4*g)*16)
		}
		y := dft16(in)
		for k1 := 1; k1 < 16; k1++ {
			off := (k1*4 + g) * 128
			t := perm(y[k1])
			t2 := add1(ins{k: kMulT, mn: "VMULPD", mem: fmt.Sprintf("%d(DX)", off+64), src: []int{t}, acc: -1})
			y[k1] = add1(ins{k: kFmaT, mn: "VFMADDSUB231PD", mem: fmt.Sprintf("%d(DX)", off), src: []int{y[k1]}, acc: t2})
		}
		if memTr {
			for k1 := 0; k1 < 16; k1++ {
				st(y[k1], "R8", (k1*4+g)*64)
			}
			continue
		}
		for h := 0; h < 4; h++ {
			o := [4]int{}
			o[0], o[1], o[2], o[3] = transpose4(y[4*h], y[4*h+1], y[4*h+2], y[4*h+3])
			for j := 0; j < 4; j++ {
				st(o[j], "R8", (h*16+4*g+j)*64)
			}
		}
	}
	for h := 0; h < 4; h++ {
		var in [16]int
		for b := 0; b < 16; b++ {
			if !memTr {
				in[b] = ld("R8", (h*16+b)*64)
				continue
			}
			g, l := b/4, b%4
			chunk := func(j int) (string, int) {
				off := ((4*h+j)*4+g)*64 + l*16
				return fmt.Sprintf("%d(R8)", off), off
			}
			m0, o0 := chunk(0)
			d := add1(ins{k: kLoadX, mn: "VMOVUPD", mem: m0, acc: -1, base: "R8", lo: o0, hi: o0 + 16, load: true})
			for j := 1; j < 4; j++ {
				m, o := chunk(j)
				d = add1(ins{k: kIns, mn: "VINSERTF32X4", imm: fmt.Sprintf("$%d", j), mem: m, src: []int{d}, acc: -1, base: "R8", lo: o, hi: o + 16, load: true})
			}
			in[b] = d
		}
		x := dft16(in)
		for k2 := 0; k2 < 16; k2++ {
			st(x[k2], "DI", (16*k2+4*h)*16)
		}
	}
}

// ---- scheduling and register allocation ----

type result struct {
	text   []string
	maxLiv int
}

func schedule(sigma float64, policy int, seed int64) (*result, bool) {
	n := len(ops)
	r := rand.New(rand.NewSource(seed))
	// predecessors
	defOp := make([]int, nv)
	for i := range defOp {
		defOp[i] = -1
	}
	for i, o := range ops {
		if o.dst >= 0 {
			defOp[o.dst] = i
		}
	}
	preds := make([][]int, n)
	uses := make([]int, nv)
	for i, o := range ops {
		for _, s := range o.src {
			preds[i] = append(preds[i], defOp[s])
			uses[s]++
		}
		if o.acc >= 0 {
			preds[i] = append(preds[i], defOp[o.acc])
			uses[o.acc]++
		}
		for j := 0; j < i; j++ {
			p := ops[j]
			switch {
			case p.store && o.load && p.base == o.base && p.lo < o.hi && o.lo < p.hi:
				preds[i] = append(preds[i], j) // RAW through memory
			case o.store && o.base == "DI" && p.load && p.base == "SI":
				preds[i] = append(preds[i], j) // dst may alias src: stores after every src load
			}
		}
	}
	succs := make([][]int, n)
	npred := make([]int, n)
	for i := range preds {
		seen := map[int]bool{}
		for _, p := range preds[i] {
			if !seen[p] {
				seen[p] = true
				succs[p] = append(succs[p], i)
				npred[i]++
			}
		}
	}
	pri := make([]float64, n)
	for i := range pri {
		pri[i] = float64(i) + r.NormFloat64()*sigma
	}
	remaining := append([]int(nil), uses...)
	phys := make([]int, nv)
	for i := range phys {
		phys[i] = -1
	}
	var free []int
	for p := 0; p < 32; p++ {
		free = append(free, p)
	}
	ready := []int{}
	for i := 0; i < n; i++ {
		if npred[i] == 0 {
			ready = append(ready, i)
		}
	}
	live, maxLive := 0, 0
	res := &result{}
	zn := func(p int) string { return fmt.Sprintf("Z%d", p) }
	for done := 0; done < n; done++ {
		// cost after scheduling each ready op
		okI, okPos := -1, -1   // lowest priority among ops that keep the cap
		anyI, anyPos := -1, -1 // fewest live registers, if none does
		anyNew := 1 << 30
		for pos, i := range ready {
			o := ops[i]
			dying := map[int]bool{}
			for _, s := range o.src {
				if remaining[s] == countIn(o, s) {
					dying[s] = true
				}
			}
			if o.acc >= 0 && remaining[o.acc] == 1 {
				dying[o.acc] = true
			}
			nl := live
			if o.dst >= 0 {
				nl++
			}
			nl -= len(dying)
			if nl <= 32 && (okI < 0 || pri[i] < pri[okI]) {
				okI, okPos = i, pos
			}
			if nl < anyNew {
				anyI, anyPos, anyNew = i, pos, nl
			}
		}
		bestI, bestPos := okI, okPos
		if bestI < 0 {
			if anyNew > 32 {
				return nil, false
			}
			bestI, bestPos = anyI, anyPos
		}
		i := bestI
		ready = append(ready[:bestPos], ready[bestPos+1:]...)
		o := ops[i]
		// release dying sources' registers
		dyingList := []int{}
		seen := map[int]bool{}
		consume := func(v int) {
			remaining[v]--
		}
		for _, s := range o.src {
			consume(s)
		}
		if o.acc >= 0 {
			consume(o.acc)
		}
		for _, s := range append(append([]int(nil), o.src...), o.acc) {
			if s >= 0 && remaining[s] == 0 && !seen[s] {
				seen[s] = true
				dyingList = append(dyingList, s)
			}
		}
		live -= len(dyingList)
		// pick the destination register
		forced := -1
		rewrite := -1
		if o.acc >= 0 {
			forced = phys[o.acc]
		} else if (o.k == kAdd || o.k == kSub) && o.flag && len(dyingList) > 0 {
			rewrite = dyingList[r.Intn(len(dyingList))]
			forced = phys[rewrite]
		}
		for _, s := range dyingList {
			if phys[s] != forced {
				free = append(free, phys[s])
			}
		}
		dstP := -1
		if o.dst >= 0 {
			if forced >= 0 {
				dstP = forced
			} else {
				idx := 0
				switch policy {
				case 0: // lowest number
					for k := range free {
						if free[k] < free[idx] {
							idx = k
						}
					}
				case 1:
					idx = r.Intn(len(free))
				case 2: // oldest freed (front of the list)
					idx = 0
				}
				dstP = free[idx]
				free = append(free[:idx], free[idx+1:]...)
			}
			phys[o.dst] = dstP
			live++
			if live > maxLive {
				maxLive = live
			}
		}
		res.text = append(res.text, render(o, phys, rewrite, zn))
		for _, s := range succs[i] {
			npred[s]--
			if npred[s] == 0 {
				ready = append(ready, s)
			}
		}
	}
	res.maxLiv = maxLive
	return res, true
}

func countIn(o ins, s int) int {
	c := 0
	for _, x := range o.src {
		if x == s {
			c++
		}
	}
	return c
}

func render(o ins, phys []int, rewrite int, zn func(int) string) string {
	z := func(v int) string { return zn(phys[v]) }
	switch o.k {
	case kLoad:
		return fmt.Sprintf("VMOVUPD %s, %s", o.mem, z(o.dst))
	case kStore:
		return fmt.Sprintf("VMOVUPD %s, %s", z(o.src[0]), o.mem)
	case kAdd, kSub:
		a, b := o.src[0], o.src[1]
		if rewrite >= 0 {
			one, neg := cname(1), cname(-1)
			if o.k == kAdd {
				if rewrite == a { // dst = a: a*1 + b
					return fmt.Sprintf("VFMADD132PD.BCST %s(SB), %s, %s", one, z(b), z(o.dst))
				}
				return fmt.Sprintf("VFMADD231PD.BCST %s(SB), %s, %s", one, z(a), z(o.dst)) // dst = b: a*1 + b
			}
			if rewrite == a { // a - b with dst = a: b*(-1) + a
				return fmt.Sprintf("VFMADD231PD.BCST %s(SB), %s, %s", neg, z(b), z(o.dst))
			}
			return fmt.Sprintf("VFMADD132PD.BCST %s(SB), %s, %s", neg, z(a), z(o.dst)) // dst = b: b*(-1) + a
		}
		mn := "VADDPD"
		if o.k == kSub {
			mn = "VSUBPD"
		}
		return fmt.Sprintf("%s %s, %s, %s", mn, z(b), z(a), z(o.dst))
	case kPerm:
		return fmt.Sprintf("VPERMILPD $0x55, %s, %s", z(o.src[0]), z(o.dst))
	case kFmaB:
		return fmt.Sprintf("%s.BCST %s(SB), %s, %s", o.mn, o.mem, z(o.src[0]), z(o.dst))
	case kFmaT:
		return fmt.Sprintf("%s %s, %s, %s", o.mn, o.mem, z(o.src[0]), z(o.dst))
	case kMulT:
		return fmt.Sprintf("VMULPD %s, %s, %s", o.mem, z(o.src[0]), z(o.dst))
	case kMulB:
		return fmt.Sprintf("VMULPD.BCST %s(SB), %s, %s", o.mem, z(o.src[0]), z(o.dst))
	case kXorB:
		return fmt.Sprintf("VXORPD.BCST %s(SB), %s, %s", o.mem, z(o.src[0]), z(o.dst))
	case kShuf:
		return fmt.Sprintf("VSHUFF64X2 %s, %s, %s, %s", o.imm, z(o.src[1]), z(o.src[0]), z(o.dst))
	case kLoadX:
		return fmt.Sprintf("VMOVUPD %s, X%d", o.mem, phys[o.dst])
	case kIns:
		return fmt.Sprintf("VINSERTF32X4 %s, %s, %s, %s", o.imm, o.mem, z(o.src[0]), z(o.dst))
	}
	panic("render")
}

func main() {
	n := flag.Int("n", 200, "variants")
	seed := flag.Int64("seed", 1, "seed")
	out := flag.String("out", "internal/schedexp", "output directory")
	pfs := flag.String("pf", "", "comma-separated FMA-rewrite probabilities (default 0,0.25,0.5)")
	shufOnly := flag.Bool("shuf", false, "register transposes only")
	flag.Parse()
	master := rand.New(rand.NewSource(*seed))
	var asm strings.Builder
	var decl strings.Builder
	var names, descr []string
	sigmas := []float64{0, 2, 6, 15, 40, 120}
	flags := []float64{0, 0.25, 0.5}
	if *pfs != "" {
		flags = nil
		for _, s := range strings.Split(*pfs, ",") {
			f, _ := strconv.ParseFloat(s, 64)
			flags = append(flags, f)
		}
	}
	policies := []string{"lowest", "random", "oldest"}
	made := 0
	for tries := 0; made < *n && tries < *n*50; tries++ {
		sigma, pf, tm, pol := sigmas[master.Intn(len(sigmas))], flags[master.Intn(len(flags))], master.Intn(2) == 1, master.Intn(3)
		if *shufOnly {
			tm = false
		}
		if made == 0 { // the control: program order, no rewrite, shuffles, lowest register
			sigma, pf, tm, pol = 0, 0, false, 0
		}
		vseed := master.Int63()
		rng = rand.New(rand.NewSource(vseed))
		pFlag, memTr, inv = pf, tm, false
		build()
		res, ok := schedule(sigma, pol, vseed)
		if !ok {
			continue
		}
		name := fmt.Sprintf("fusedv%03d", made)
		tr := "shuf"
		if tm {
			tr = "mem"
		}
		descr = append(descr, fmt.Sprintf("sigma=%g fma=%.0f%% transpose=%s regs=%s maxlive=%d", sigma, pf*100, tr, policies[pol], res.maxLiv))
		names = append(names, name)
		fmt.Fprintf(&asm, "\n// func %s(dst, src, tw *complex128)\nTEXT ·%s(SB), $4160-24\n", name, name)
		asm.WriteString("\tMOVQ dst+0(FP), DI\n\tMOVQ src+8(FP), SI\n\tMOVQ tw+16(FP), DX\n\tMOVQ SP, R8\n\tADDQ $63, R8\n\tANDQ $-64, R8\n")
		for _, l := range res.text {
			asm.WriteString("\t" + l + "\n")
		}
		asm.WriteString("\tVZEROUPPER\n\tRET\n")
		fmt.Fprintf(&decl, "//go:noescape\nfunc %s(dst, src, tw *complex128)\n\n", name)
		made++
	}
	names2 := append([]string(nil), names...)
	sort.Strings(names2)
	var hdr strings.Builder
	hdr.WriteString("// Code generated by internal/genfused256/variants; experiment only, do not commit.\n\n//go:build amd64\n\n#include \"textflag.h\"\n\n")
	cn := make([]string, 0, len(consts))
	for k := range consts {
		cn = append(cn, k)
	}
	sort.Strings(cn)
	for _, k := range cn {
		fmt.Fprintf(&hdr, "DATA %s+0(SB)/8, $%#x\nGLOBL %s(SB), RODATA|NOPTR, $8\n", k, math.Float64bits(consts[k]), k)
	}
	var g strings.Builder
	g.WriteString("package schedexp\n\n")
	g.WriteString(decl.String())
	g.WriteString("var variantFns = [...]func(dst, src, tw *complex128){\n")
	for _, nme := range names {
		fmt.Fprintf(&g, "\t%s,\n", nme)
	}
	g.WriteString("}\n\nvar variantDescr = [...]string{\n")
	for _, d := range descr {
		fmt.Fprintf(&g, "\t%q,\n", d)
	}
	g.WriteString("}\n")
	os.MkdirAll(*out, 0o755)
	os.WriteFile(filepath.Join(*out, "variants_amd64.s"), []byte(hdr.String()+asm.String()), 0o644)
	os.WriteFile(filepath.Join(*out, "variants_amd64.go"), []byte(g.String()), 0o644)
	fmt.Printf("%d variants\n", made)
}
