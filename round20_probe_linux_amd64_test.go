package fft

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

// r20Alloc maps n complex128 values on 2 MiB-aligned memory advised huge
// (MADV_HUGEPAGE) or not (MADV_NOHUGEPAGE), plus pad bytes of offset.
func r20Alloc(n int, huge bool, pad int) []complex128 {
	const H = 2 << 20
	nb := n*16 + pad
	sz := (nb + H - 1) / H * H
	b, err := syscall.Mmap(-1, 0, sz+H, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	off := (H - int(uintptr(unsafe.Pointer(&b[0])))%H) % H
	b = b[off : off+sz]
	adv := 15
	if huge {
		adv = 14
	}
	if err := syscall.Madvise(b, adv); err != nil {
		panic(err)
	}
	for i := range b {
		b[i] = 0
	}
	return unsafe.Slice((*complex128)(unsafe.Pointer(&b[pad])), n)
}

func r20Median(x []float64) (med, lo, hi float64) {
	y := append([]float64(nil), x...)
	sort.Float64s(y)
	return y[len(y)/2], y[0], y[len(y)-1]
}

// r20Time returns ns per point of f, one sample: f is repeated for >= 30 ms.
func r20Time(f func(), n int) float64 {
	f()
	it := 1
	for {
		t := time.Now()
		for i := 0; i < it; i++ {
			f()
		}
		d := time.Since(t)
		if d > 30*time.Millisecond {
			return float64(d.Nanoseconds()) / float64(it) / float64(n)
		}
		it *= 2
	}
}

// r20MemPass moves the data of a radix-r pass without arithmetic: the same
// loads and stores in the same order, a lower bound on the memory cost.
func r20MemPass(r, ido, l1 int, cc, ch []complex128) {
	for k := 0; k < l1; k++ {
		for j := 0; j < r; j++ {
			src := cc[ido*(j+r*k):][:ido]
			dst := ch[ido*(k+l1*j):][:ido]
			copy(dst, src)
		}
	}
}

func r20Chunks() []int {
	var ls []int
	for _, f := range strings.Split(os.Getenv("R20C"), ",") {
		if v, err := strconv.Atoi(f); err == nil {
			ls = append(ls, v)
		}
	}
	if ls == nil {
		ls = []int{cascadeChunk}
	}
	return ls
}

func r20Locals() []int {
	var ls []int
	for _, f := range strings.Split(os.Getenv("R20L"), ",") {
		if v, err := strconv.Atoi(f); err == nil {
			ls = append(ls, v)
		}
	}
	if ls == nil {
		ls = []int{4096, 8192, 16384}
	}
	return ls
}

func r20Sizes() []int {
	s := os.Getenv("R20N")
	if s == "" {
		s = "4096,8192,16384,32768,65536,131072,262144,524288,1048576"
	}
	var ns []int
	for _, f := range strings.Split(s, ",") {
		v, _ := strconv.Atoi(f)
		ns = append(ns, v)
	}
	return ns
}

func log2i(n int) int {
	e := 0
	for 1<<e < n {
		e++
	}
	return e
}

// runPasses runs the plan's passes as transform does, with the scratch given.
func r20Run(p *skPlan, dst, src, scr []complex128) {
	s := len(p.stages)
	in := src
	for k := range p.stages {
		out := scr
		if (s-1-k)%2 == 0 {
			out = dst
		}
		p.stages[k].pass(out, in, false)
		in = out
	}
}

// r20Blocked runs passes 0..t-1 over the whole array, then the remaining
// passes group by group: B consecutive blocks of pass t (m = n/l1_t points
// each) as a local Stockham of B·m points with l1 starting at B, in loc, and
// scatters the result: X[K0 + Kl + l1_t·f] = local[Kl + B·f]. Out of place.
func r20Blocked(p *skPlan, t, B int, dst, src, scr, loc []complex128) {
	r20BlockedX(p, t, B, dst, src, scr, loc, 0)
}

// r20BlockedX is r20Blocked with a timing variant: how 1 writes the last
// local pass straight into dst at the group's own (wrong) positions, the
// traffic a fused scatter would have; how 2 runs only the global passes; how
// 3 only the groups.
func r20BlockedX(p *skPlan, t, B int, dst, src, scr, loc []complex128, how int) {
	n, s := p.n, len(p.stages)
	in := src
	for k := 0; k < t; k++ {
		out := scr
		if (t-1-k)%2 == 1 {
			out = dst
		}
		if how != 3 {
			p.stages[k].pass(out, in, false)
		}
		in = out
	}
	if how == 2 {
		return
	}
	l1t := p.stages[t].l1
	m := n / l1t
	for K0 := 0; K0 < l1t; K0 += B {
		cur := in[K0*m : (K0+B)*m]
		bufs := [2][]complex128{loc[:B*m], loc[B*m : 2*B*m]}
		l1 := B
		for q := t; q < s; q++ {
			st := &p.stages[q]
			out := bufs[(q-t)%2]
			if how == 1 && q == s-1 {
				out = dst[K0*m : (K0+B)*m]
			}
			if !kernels.StockhamPass(st.r, st.ido, l1, cur, out, st.twX, false, st.wide) {
				panic("no kernel")
			}
			l1 *= st.r
			cur = out
		}
		if how == 1 {
			continue
		}
		for f := 0; f < m; f++ {
			copy(dst[K0+l1t*f:][:B], cur[B*f:][:B])
		}
	}
}

var r20MaxLoc, r20MinLoc = r20Env("R20MAXLOC", 65536), r20Env("R20MINLOC", 1024)

func r20Env(k string, d int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return v
	}
	return d
}

// TestRound20Probe is the experiment driver of Round 20; it runs only with
// R20MODE set.
func TestRound20Probe(t *testing.T) {
	mode := os.Getenv("R20MODE")
	if mode == "" {
		t.Skip("R20MODE not set")
	}
	rounds := 5
	if v, err := strconv.Atoi(os.Getenv("R20ROUNDS")); err == nil {
		rounds = v
	}
	avx512 := kernels.UseStockhamAVX512
	defer func() { kernels.UseStockhamAVX512 = avx512 }()
	switch mode {
	case "passes":
		// Each pass alone at every size, AVX-512 and AVX2, plus its memory-only twin.
		for _, n := range r20Sizes() {
			p := newSKPlanFactors(n, radix8Maximal(log2i(n)))
			src := benchComplex(n)
			dst := make([]complex128, n)
			bp := p.scratch.Get().(*[]complex128)
			scr := offTheSets(*bp, dst, n)
			for _, w := range []bool{true, false} {
				kernels.UseStockhamAVX512 = w && avx512
				res := map[string][]float64{}
				var keys []string
				add := func(k string, v float64) {
					if _, ok := res[k]; !ok {
						keys = append(keys, k)
					}
					res[k] = append(res[k], v)
				}
				for r := 0; r < rounds; r++ {
					add("whole", r20Time(func() { r20Run(p, dst, src, scr) }, n))
					s := len(p.stages)
					in := src
					sum, msum := 0.0, 0.0
					for k := range p.stages {
						out := scr
						if (s-1-k)%2 == 0 {
							out = dst
						}
						st, i0, o0 := &p.stages[k], in, out
						v := r20Time(func() { st.pass(o0, i0, false) }, n)
						m := r20Time(func() { r20MemPass(st.r, st.ido, st.l1, i0, o0) }, n)
						name := fmt.Sprintf("pass%d_r%d_ido%d_l1%d", k, st.r, st.ido, st.l1)
						add(name, v)
						add(name+"_mem", m)
						sum += v
						msum += m
						in = out
					}
					add("sumpasses", sum)
					add("summem", msum)
				}
				for _, k := range keys {
					med, lo, hi := r20Median(res[k])
					fmt.Printf("passes n=%d avx512=%v %-28s %.3f ns/pt [%.3f %.3f]\n", n, w && avx512, k, med, lo, hi)
				}
			}
			p.scratch.Put(bp)
		}
	case "blocked":
		// The blocked schedule against the current one, checked bit for bit.
		for _, n := range r20Sizes() {
			e := log2i(n)
			p := newSKPlanFactors(n, radix8Maximal(e))
			src := benchComplex(n)
			ref := make([]complex128, n)
			dst := make([]complex128, n)
			bp := p.scratch.Get().(*[]complex128)
			scr := offTheSets(*bp, dst, n)
			r20Run(p, ref, src, scr)
			loc := make([]complex128, 1<<20)
			type variant struct {
				name string
				f    func()
			}
			vs := []variant{{"base", func() { r20Run(p, dst, src, scr) }}}
			for tt := 1; tt < len(p.stages)-1; tt++ {
				m := n / p.stages[tt].l1
				for _, B := range []int{4, 8, 16, 32, 64, 128} {
					if B > p.stages[tt].l1 || 2*B*m > len(loc) || B*m > r20MaxLoc || B*m < r20MinLoc {
						continue
					}
					tt, B := tt, B
					r20Blocked(p, tt, B, dst, src, scr, loc)
					for i := range dst {
						if dst[i] != ref[i] {
							t.Fatalf("n=%d t=%d B=%d: index %d differs", n, tt, B, i)
						}
					}
					vs = append(vs, variant{fmt.Sprintf("t%d_B%d_loc%d", tt, B, B*m), func() { r20Blocked(p, tt, B, dst, src, scr, loc) }})
					if os.Getenv("R20X") != "" {
						for how := 1; how <= 3; how++ {
							how := how
							vs = append(vs, variant{fmt.Sprintf("t%d_B%d_x%d", tt, B, how), func() { r20BlockedX(p, tt, B, dst, src, scr, loc, how) }})
						}
					}
				}
			}
			res := make([][]float64, len(vs))
			for r := 0; r < rounds; r++ {
				for i, v := range vs {
					res[i] = append(res[i], r20Time(v.f, n))
				}
			}
			for i, v := range vs {
				med, lo, hi := r20Median(res[i])
				fmt.Printf("blocked n=%d %-22s %.3f ns/pt [%.3f %.3f]\n", n, v.name, med, lo, hi)
			}
			p.scratch.Put(bp)
		}
	case "cascade":
		// The shipped blocked schedule against breadth first, interleaved.
		for _, n := range r20Sizes() {
			e := log2i(n)
			src := benchComplex(n)
			dst := make([]complex128, n)
			p := newSKPlanFactors(n, radix8Maximal(e))
			buf := make([]complex128, n+setSpan+2*65536+cascadeSkew)
			type variant struct {
				name string
				f    func()
			}
			plain := func() {
				p.cascT = 0
				p.run(dst, src, buf, false)
			}
			vs := []variant{{"plain", plain}}
			for _, L := range r20Locals() {
				t0, b0 := 0, 0
				for t := 1; t < len(p.stages); t++ {
					if m := n / p.stages[t].l1; 4*m <= L && L/m <= p.stages[t].l1 {
						t0, b0 = t, L/m
						break
					}
				}
				if t0 == 0 {
					continue
				}
				for _, w := range []bool{true, false} {
					if !w && os.Getenv("R20NOAVX2") != "" {
						continue
					}
					for _, chunk := range append([]int{0}, r20Chunks()...) {
						w, chunk := w, chunk
						pair := chunk > 0
						if pair {
							if chunk != cascadeChunk || !cascadePairs(t0, p.stages) {
								continue
							}
						}
						vs = append(vs, variant{fmt.Sprintf("L%d_t%d_b%d_w%v_c%d", L, t0, b0, w, chunk), func() {
							kernels.UseStockhamAVX512 = w && avx512
							p.cascT, p.cascB, p.cascPair = t0, b0, pair
							p.cascadeRun(dst, src, buf, false)
							kernels.UseStockhamAVX512 = avx512
						}})
					}
				}
			}
			vs = append(vs, variant{"plain_avx2", func() {
				kernels.UseStockhamAVX512 = false
				plain()
				kernels.UseStockhamAVX512 = avx512
			}})
			res := make([][]float64, len(vs))
			for r := 0; r < rounds; r++ {
				for i, v := range vs {
					res[i] = append(res[i], r20Time(v.f, n))
				}
			}
			for i, v := range vs {
				med, lo, hi := r20Median(res[i])
				fmt.Printf("cascade n=%d %-24s %.3f ns/pt [%.3f %.3f]\n", n, v.name, med, lo, hi)
			}
		}
	case "pair":
		// Timing only (wrong twiddles): passes 0 and 1 fused, chunk by chunk
		// of w points of pass 1's ido: gather the 64 runs, two local passes,
		// scatter the 64 runs; against the two passes over the whole array.
		for _, n := range r20Sizes() {
			e := log2i(n)
			p := newSKPlanFactors(n, radix8Maximal(e))
			src := benchComplex(n)
			dst := make([]complex128, n)
			bp := p.scratch.Get().(*[]complex128)
			scr := offTheSets((*bp)[:n+setSpan], dst, n)
			ido0, ido1 := n/8, n/64
			type variant struct {
				name string
				f    func()
			}
			vs := []variant{{"two_passes", func() {
				p.stages[0].pass(dst, src, false)
				p.stages[1].pass(scr, dst, false)
			}}}
			for _, w := range []int{16, 32, 64, 128} {
				w := w
				la := make([]complex128, 64*w+64)
				lb := make([]complex128, 64*w+64)
				tw := make([]complex128, 7*w)
				for i := range tw {
					tw[i] = complex(0.6, 0.8)
				}
				vs = append(vs, variant{fmt.Sprintf("fused_w%d", w), func() {
					for i0 := 0; i0 < ido1; i0 += w {
						// gather: la[i + w*(m + 8*mp)] = src[i0+i + ido1*mp + ido0*m]
						for mp := 0; mp < 8; mp++ {
							for m := 0; m < 8; m++ {
								copy(la[w*(m+8*mp):][:w], src[i0+ido1*mp+ido0*m:][:w])
							}
						}
						kernels.StockhamPass(8, w, 8, la[:64*w], lb[:64*w], tw, false, true)
						kernels.StockhamPass(8, w, 8, lb[:64*w], la[:64*w], tw, false, true)
						// scatter: scr[i0+i + ido1*(k + 8*j2)] = la[i + w*(k + 8*j2)]
						for q := 0; q < 64; q++ {
							copy(scr[i0+ido1*q:][:w], la[w*q:][:w])
						}
					}
				}})
			}
			res := make([][]float64, len(vs))
			for r := 0; r < rounds; r++ {
				for i, v := range vs {
					res[i] = append(res[i], r20Time(v.f, n))
				}
			}
			for i, v := range vs {
				med, lo, hi := r20Median(res[i])
				fmt.Printf("pair n=%d %-12s %.3f ns/pt [%.3f %.3f]\n", n, v.name, med, lo, hi)
			}
			p.scratch.Put(bp)
		}
	case "whole":
		// Variants of the whole transform, interleaved round by round.
		for _, n := range r20Sizes() {
			e := log2i(n)
			type variant struct {
				name string
				f    func()
			}
			src := benchComplex(n)
			dst := make([]complex128, n)
			pA := newSKPlanFactors(n, radix8Maximal(e))
			pB := newSKPlanFactors(n, pow2Radices(e, false))
			pC := newSKPlanFactors(n, pow2Radices(e, true))
			bp := pA.scratch.Get().(*[]complex128)
			mk := func(gap int) []complex128 {
				d := uintptr(unsafe.Pointer(unsafe.SliceData(*bp))) - uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
				shift := int((uintptr(gap)-d)&4095) / 16
				return (*bp)[shift : shift+n]
			}
			itp := newITPlan(n)
			hs, hd, hscr := r20Alloc(n, true, 0), r20Alloc(n, true, 0), r20Alloc(n, true, setGap)
			ss, sd, sscr := r20Alloc(n, false, 0), r20Alloc(n, false, 0), r20Alloc(n, false, setGap)
			copy(hs, src)
			copy(ss, src)
			vs := []variant{
				{"A_gap576", func() { r20Run(pA, dst, src, mk(576)) }},
				{"A_gap0", func() { r20Run(pA, dst, src, mk(0)) }},
				{"A_gap1088", func() { r20Run(pA, dst, src, mk(1088)) }},
				{"A_gap2112", func() { r20Run(pA, dst, src, mk(2112)) }},
				{"A_huge", func() { r20Run(pA, hd, hs, hscr) }},
				{"A_small", func() { r20Run(pA, sd, ss, sscr) }},
				{"A_inplace", func() { pA.transform(dst, dst, false) }},
				{"B_gap576", func() { r20Run(pB, dst, src, mk(576)) }},
				{"C_gap576", func() { r20Run(pC, dst, src, mk(576)) }},
				{"A_avx2", func() {
					kernels.UseStockhamAVX512 = false
					r20Run(pA, dst, src, mk(576))
					kernels.UseStockhamAVX512 = avx512
				}},
				{"B_avx2", func() {
					kernels.UseStockhamAVX512 = false
					r20Run(pB, dst, src, mk(576))
					kernels.UseStockhamAVX512 = avx512
				}},
				{"itplan", func() { itp.transform(dst, src, false) }},
			}
			res := make([][]float64, len(vs))
			for r := 0; r < rounds; r++ {
				for i, v := range vs {
					res[i] = append(res[i], r20Time(v.f, n))
				}
			}
			for i, v := range vs {
				med, lo, hi := r20Median(res[i])
				fmt.Printf("whole n=%d %-10s %.3f ns/pt [%.3f %.3f]\n", n, v.name, med, lo, hi)
			}
			pA.scratch.Put(bp)
		}
	}
}
