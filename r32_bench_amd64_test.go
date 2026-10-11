//go:build amd64

package fft

// The measurements behind BENCHMARKS.md's Round 38: whole transforms through
// the exported plans, each row built once per factorization variant (the
// variant handed to skFactorize through r32Table, so the plan is the one the
// library would build with that rule: split layouts, frame kernels and
// scratch gap included), timed in one process in rounds whose variant order
// rotates.

import (
	"flag"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-fft/fft/internal/kernels"
)

var (
	r32AB   = flag.String("r32.ab", "", "TestR32AB: rounds,target ns per sample (e.g. 9,2000000)")
	r32Rows = flag.String("r32.rows", "c1024:cur,32x32", "TestR32AB: rows kind+n:variant,variant;... (kind c complex, r RFFT, i IRFFT; a variant is cur or a factorization of the complex length, n or n/2)")
	r32Pass = flag.String("r32.pass", "", "TestR32Passes: rounds,target ns per sample")
	r32PNs  = flag.String("r32.pns", "1024:32x32,8x8x16", "TestR32Passes: n:factorization;...")
)

// r32Variant builds the plan for row kind of length n with the complex
// length m = n (or n/2 for the real rows) factored as name says, through
// r32Table so that newSKPlan sees it, and takes it out of the plan cache.
func r32Variant(t *testing.T, kind byte, n int, name string) (fn func(), desc string) {
	m := n
	if kind != 'c' {
		m = n / 2
	}
	var f []int
	if name != "cur" {
		for _, x := range strings.Split(name, "x") {
			r, err := strconv.Atoi(x)
			if err != nil {
				t.Fatal(err)
			}
			f = append(f, r)
		}
		if product(f) != m {
			t.Fatalf("%c%d: %s multiplies to %d, not %d", kind, n, name, product(f), m)
		}
	}
	saved := r32Table
	defer func() { r32Table = saved }()
	if f != nil {
		r32Table = map[int][]int{m: f}
	}
	forget := func() { planMu.Lock(); delete(planCache, m); planMu.Unlock() }
	forget()
	defer forget()
	src := benchComplex(n)
	switch kind {
	case 'c':
		p := NewPlan(n)
		dst := make([]complex128, n)
		return func() { p.FFT(dst, src) }, r32Desc(p)
	case 'r':
		p := NewRealPlan(n)
		x := make([]float64, n)
		for i := range x {
			x[i] = real(src[i])
		}
		dst := make([]complex128, n/2+1)
		return func() { p.RFFT(dst, x) }, r32Desc(p.half)
	case 'i':
		p := NewRealPlan(n)
		x := make([]float64, n)
		for i := range x {
			x[i] = real(src[i])
		}
		spec := make([]complex128, n/2+1)
		p.RFFT(spec, x)
		dst := make([]float64, n)
		return func() { p.IRFFT(dst, spec) }, r32Desc(p.half)
	}
	t.Fatalf("row kind %c", kind)
	return nil, ""
}

// r32Desc names a plan's passes: radix, then s for a split layout mode, and
// +zn when the frame kernels run it.
func r32Desc(p *Plan) string {
	if p.sk == nil {
		return "not Stockham"
	}
	var parts []string
	for _, st := range p.sk.stages {
		s := strconv.Itoa(st.r)
		if st.split != 0 {
			s += "s"
		}
		parts = append(parts, s)
	}
	d := strings.Join(parts, "·")
	if p.sk.zn != nil {
		d += "+zn"
	}
	return d
}

// TestR32AB times every row of -r32.rows, variant by variant, in rotated
// rounds; the rows run in the order given, each round all of them.
func TestR32AB(t *testing.T) {
	if *r32AB == "" {
		t.Skip("set -r32.ab=rounds,target-ns")
	}
	var rounds, target int
	if _, err := fmt.Sscanf(*r32AB, "%d,%d", &rounds, &target); err != nil {
		t.Fatal(err)
	}
	type row struct {
		name  string
		vars  []string
		fns   []func()
		descs []string
		iters int
		ns    [][]float64
	}
	var rows []*row
	for _, spec := range strings.Split(*r32Rows, ";") {
		head, vs, ok := strings.Cut(spec, ":")
		if !ok || len(head) < 2 {
			t.Fatalf("row %q: want kind+n:variants", spec)
		}
		n, err := strconv.Atoi(head[1:])
		if err != nil {
			t.Fatal(err)
		}
		r := &row{name: head, vars: strings.Split(vs, ",")}
		for _, v := range r.vars {
			fn, d := r32Variant(t, head[0], n, v)
			r.fns = append(r.fns, fn)
			r.descs = append(r.descs, d)
		}
		// Calibrate on the first variant.
		r.fns[0]()
		t0 := time.Now()
		k := 0
		for time.Since(t0) < 20*time.Millisecond {
			r.fns[0]()
			k++
		}
		per := float64(time.Since(t0).Nanoseconds()) / float64(k)
		r.iters = max(1, int(float64(target)/per))
		r.ns = make([][]float64, len(r.vars))
		rows = append(rows, r)
	}
	fmt.Printf("R32AB host avx2=%v avx512=%v intel=%v split=%v rounds=%d target=%d ns\n",
		kernels.UseStockhamAVX2, kernels.UseStockhamAVX512, kernels.IntelCPU, kernels.UseStockhamSplit, rounds, target)
	r32Load("start")
	for round := range rounds {
		for _, r := range rows {
			for q := range r.vars {
				v := (q + round) % len(r.vars)
				fn := r.fns[v]
				t0 := time.Now()
				for range r.iters {
					fn()
				}
				r.ns[v] = append(r.ns[v], float64(time.Since(t0).Nanoseconds())/float64(r.iters))
			}
		}
	}
	r32Load("end")
	for _, r := range rows {
		for v := range r.vars {
			ratios := make([]float64, rounds)
			for i := range ratios {
				ratios[i] = r.ns[0][i] / r.ns[v][i]
			}
			fmt.Printf("R32AB %-7s %-10s %-22s %10.1f ns  first/this %.3f  range %.3f-%.3f  spread %.3f\n",
				r.name, r.vars[v], r.descs[v], median(r.ns[v]), median(ratios), slices.Min(ratios), slices.Max(ratios),
				slices.Max(r.ns[v])/slices.Min(r.ns[v]))
		}
	}
}

// r32Load prints the host's load average, when it has /proc/loadavg.
func r32Load(when string) {
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		fmt.Printf("R32LOAD %s %s", when, b)
	}
}

// TestR32Passes times each pass of the plans of -r32.pns alone, src to dst
// (two n-point buffers 576 bytes apart modulo 4 KB, as offTheSets places
// them), in rotated rounds.
func TestR32Passes(t *testing.T) {
	if *r32Pass == "" {
		t.Skip("set -r32.pass=rounds,target-ns")
	}
	var rounds, target int
	if _, err := fmt.Sscanf(*r32Pass, "%d,%d", &rounds, &target); err != nil {
		t.Fatal(err)
	}
	type item struct {
		name  string
		fn    func()
		iters int
		ns    []float64
	}
	var items []*item
	for _, spec := range strings.Split(*r32PNs, ";") {
		head, fs, _ := strings.Cut(spec, ":")
		n, err := strconv.Atoi(head)
		if err != nil {
			t.Fatal(err)
		}
		var f []int
		for _, x := range strings.Split(fs, "x") {
			r, _ := strconv.Atoi(x)
			f = append(f, r)
		}
		p := newSKPlanFactors(n, f)
		a := benchComplex(n)
		buf := make([]complex128, n+setSpan)
		b := offTheSets(buf, a, n)
		for k := range p.stages {
			st := &p.stages[k]
			in, out := a, b
			if k%2 == 1 {
				in, out = b, a
			}
			items = append(items, &item{name: fmt.Sprintf("%d:%s pass %d (r%d ido %d split %d)", n, fs, k, st.r, st.ido, st.split),
				fn: func() { st.pass(out, in, false) }})
		}
	}
	for _, it := range items {
		it.fn()
		t0 := time.Now()
		k := 0
		for time.Since(t0) < 10*time.Millisecond {
			it.fn()
			k++
		}
		it.iters = max(1, int(float64(target)/(float64(time.Since(t0).Nanoseconds())/float64(k))))
	}
	r32Load("start")
	for round := range rounds {
		for q := range items {
			it := items[(q+round)%len(items)]
			t0 := time.Now()
			for range it.iters {
				it.fn()
			}
			it.ns = append(it.ns, float64(time.Since(t0).Nanoseconds())/float64(it.iters))
		}
	}
	r32Load("end")
	for _, it := range items {
		fmt.Printf("R32PASS %-48s %10.1f ns  spread %.3f\n", it.name, median(it.ns), slices.Max(it.ns)/slices.Min(it.ns))
	}
}
