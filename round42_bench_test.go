package fft

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// Round 42: which passes of an FMA plan pay. Every helper here is prefixed
// r42.

var (
	r42AB   = flag.String("r42.ab", "", "rounds,ms,placements: run Round 42's per-pass FMA A/B (TestR42Masks)")
	r42Rows = flag.String("r42.rows", "c256,c1024,c4096,c65536,c1048576,c1000,c1080,c1920,c1296,r256,r1024,r4096,r65536,r1000,r1080,r1920",
		"TestR42Masks: rows, c<n> complex, r<n> RFFT (the mask applies to the half plan); an optional /f1.f2... sets the factors")
	r42Passes = flag.String("r42.passes", "", "rounds,ms: run Round 42's single-pass timing (TestR42Passes)")
)

// r42Spec parses a row: kind, n, and the factors after a slash (nil: the
// library's).
func r42Spec(t *testing.T, spec string) (byte, int, []int) {
	kind, arg := spec[0], spec[1:]
	var fs []int
	if i := strings.IndexByte(arg, '/'); i >= 0 {
		for _, f := range strings.Split(arg[i+1:], ".") {
			v, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			fs = append(fs, v)
		}
		arg = arg[:i]
	}
	n, err := strconv.Atoi(arg)
	if err != nil {
		t.Fatal(err)
	}
	return kind, n, fs
}

// r42Plan builds a complex plan of n points (factors fs, or the library's),
// an FMA plan whose stage k fuses where bit k of mask is set (mask < 0: the
// default plan).
func r42Plan(n int, fs []int, mask int) *Plan {
	if fs == nil {
		p := newPlanF(n, mask >= 0)
		if mask >= 0 && p.sk != nil {
			r42Apply(p.sk, mask)
		}
		return p
	}
	p := &Plan{n: n, sk: newSKPlanFactorsOpt(n, fs, mask >= 0)}
	if mask >= 0 {
		r42Apply(p.sk, mask)
	}
	return p
}

// r42Apply makes stage k of an FMA plan fuse where bit k of mask is set,
// and rebinds its frame kernels.
func r42Apply(p *skPlan, mask int) {
	for k := range p.stages {
		st := &p.stages[k]
		st.fma = fmaRadix(st.r) && (mask>>k&1 == 1 || st.r == 32)
	}
	p.zn = znPlanFor(p.n, p.stages, true)
}

// r42Stages is the number of stages of a row's (half) plan.
func r42Stages(kind byte, n int, fs []int) int {
	m := n
	if kind == 'r' {
		m = n / 2
	}
	if fs != nil {
		return len(fs)
	}
	p := newPlanF(m, true)
	if p.sk == nil {
		return 0
	}
	return len(p.sk.stages)
}

func r42Build(t *testing.T, spec string, mask int) func() {
	kind, n, fs := r42Spec(t, spec)
	switch kind {
	case 'c':
		p := r42Plan(n, fs, mask)
		src, dst := cmplxSignal(n), make([]complex128, n)
		return func() { p.FFT(dst, src) }
	case 'r':
		p := newRealPlanF(n, mask >= 0)
		p.half = r42Plan(n/2, fs, mask)
		src, dst := realSignal(n), make([]complex128, n/2+1)
		return func() { p.RFFT(dst, src) }
	}
	t.Fatalf("row %q", spec)
	return nil
}

// r42Masks are the variants of a plan of s stages: -1 (default), then every
// mask when s <= 4, else all on, all off and every mask with one stage off.
func r42Masks(s int) []int {
	all := 1<<s - 1
	m := []int{-1}
	if s <= 4 {
		for x := all; x >= 0; x-- {
			m = append(m, x)
		}
		return m
	}
	m = append(m, all)
	for k := range s {
		m = append(m, all&^(1<<k))
	}
	return append(m, 0)
}

func r42MaskName(mask, s int) string {
	if mask < 0 {
		return "default"
	}
	b := []byte{}
	for k := range s {
		if mask>>k&1 == 1 {
			b = append(b, 'F')
		} else {
			b = append(b, '.')
		}
	}
	return string(b)
}

// TestR42Masks times, for every row, the default plan and FMA plans fusing
// only some of their passes, at several heap placements, in rounds with the
// variant order rotated, and prints per variant the median time and the
// median over rounds of default ÷ variant with its range.
func TestR42Masks(t *testing.T) {
	if *r42AB == "" {
		t.Skip("set -r42.ab=rounds,ms,placements")
	}
	var rounds, ms, places int
	if _, err := fmt.Sscanf(*r42AB, "%d,%d,%d", &rounds, &ms, &places); err != nil {
		t.Fatal(err)
	}
	t.Logf("fmaAvailable=%v rounds=%d ms=%d placements=%d", fmaAvailable, rounds, ms, places)
	for _, spec := range strings.Split(*r42Rows, ",") {
		kind, n, fs := r42Spec(t, spec)
		s := r42Stages(kind, n, fs)
		masks := r42Masks(s)
		nv := len(masks)
		runs := make([][]func(), places) // [placement][variant]
		for j := range places {
			r41Pad(j)
			runs[j] = make([]func(), nv)
			for k := range nv {
				w := (k + j) % nv
				runs[j][w] = r42Build(t, spec, masks[w])
			}
		}
		times := make([][]float64, nv)  // per round, median over placements
		ratios := make([][]float64, nv) // default ÷ variant per round
		for round := range rounds {
			per := make([][]float64, nv)
			for j := range places {
				for k := range nv {
					w := (k + round + j) % nv
					per[w] = append(per[w], r41Time(runs[j][w], ms))
				}
			}
			for w := range nv {
				times[w] = append(times[w], r41Median(per[w]))
			}
			for w := range nv {
				ratios[w] = append(ratios[w], times[0][round]/times[w][round])
			}
		}
		desc := ""
		if p := r42Plan(map[byte]int{'c': n, 'r': n / 2}[kind], fs, 0); p.sk != nil {
			for _, st := range p.sk.stages {
				desc += fmt.Sprintf(" %d(ido %d, split %d)", st.r, st.ido, st.split)
			}
			if p.sk.zn != nil {
				desc += " +zn"
			}
		}
		t.Logf("%s:%s", spec, desc)
		for w, m := range masks {
			r := slices.Clone(ratios[w])
			slices.Sort(r)
			t.Logf("  %-12s %-8s %11.1f ns  default/variant %.3f  (%.3f–%.3f)  rounds %s",
				spec, r42MaskName(m, s), r41Median(times[w]), r41Median(ratios[w]), r[0], r[len(r)-1], r41Fmt(ratios[w]))
		}
	}
}

// TestR42Passes times each pass of every complex row alone, default and FMA
// kernel, on buffers of their own (in rounds, the order rotated), and
// prints ns per point and the default ÷ FMA ratio per pass.
func TestR42Passes(t *testing.T) {
	if *r42Passes == "" {
		t.Skip("set -r42.passes=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*r42Passes, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	for _, spec := range strings.Split(*r42Rows, ",") {
		kind, n, fs := r42Spec(t, spec)
		if kind == 'r' {
			n /= 2
		}
		pf := r42Plan(n, fs, 1<<20-1)
		pd := pf // a plan of given factors is timed as an FMA plan only
		if fs == nil {
			pd = r42Plan(n, fs, -1)
		}
		if pd.sk == nil {
			continue
		}
		in := cmplxSignal(n)
		buf := make([]complex128, n+setSpan)
		out := offTheSets(buf, in, n)
		for k := range pd.sk.stages {
			sd, sf := &pd.sk.stages[k], &pf.sk.stages[k]
			f := [2]func(){func() { sd.pass(out, in, false) }, func() { sf.pass(out, in, false) }}
			var tm [2][]float64
			for round := range rounds {
				for v := range 2 {
					w := (v + round) % 2
					tm[w] = append(tm[w], r41Time(f[w], ms))
				}
			}
			d, fm := r41Median(tm[0]), r41Median(tm[1])
			t.Logf("%-10s pass %d r %2d ido %5d l1 %5d split %d fma %v: default %8.1f ns (%.3f ns/pt)  fma %8.1f ns  default/fma %.3f",
				spec, k, sd.r, sd.ido, sd.l1, sd.split, sf.fma, d, d/float64(n), fm, d/fm)
		}
	}
}

var r42Vars = flag.String("r42.vars", "", "TestR42Vars: row:variant,variant;row:... — c<n> or r<n>; variants def (default plan), fma (the library's FMA plan), or factors a.b.c (an FMA plan of those passes) with an optional @mask of fused stages (F and .)")

// r42Variant builds one variant of a row (see r42Vars).
func r42Variant(t *testing.T, kind byte, n int, v string) func() {
	m := n
	if kind == 'r' {
		m = n / 2
	}
	var p *Plan
	switch {
	case v == "def":
		p = newPlanF(m, false)
	case v == "fma":
		p = newPlanF(m, true)
	default:
		spec, mask, _ := strings.Cut(v, "@")
		var fs []int
		for _, f := range strings.Split(spec, ".") {
			x, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			fs = append(fs, x)
		}
		if factorProduct(fs) != m {
			t.Fatalf("variant %s: product %d, want %d", v, factorProduct(fs), m)
		}
		p = &Plan{n: m, sk: newSKPlanFactorsOpt(m, fs, true)}
		if mask != "" {
			bits := 0
			for k, c := range mask {
				if c == 'F' {
					bits |= 1 << k
				}
			}
			r42Apply(p.sk, bits)
		}
	}
	if kind == 'c' {
		src, dst := cmplxSignal(n), make([]complex128, n)
		return func() { p.FFT(dst, src) }
	}
	rp := newRealPlanF(n, v != "def")
	rp.half = p
	src, dst := realSignal(n), make([]complex128, n/2+1)
	return func() { rp.RFFT(dst, src) }
}

// TestR42Vars times the variants of every row (r42Vars) at several heap
// placements, in rounds with the order rotated, and prints per variant the
// median time and the median over rounds of first variant ÷ variant.
func TestR42Vars(t *testing.T) {
	if *r42AB == "" || *r42Vars == "" {
		t.Skip("set -r42.ab=rounds,ms,placements and -r42.vars")
	}
	var rounds, ms, places int
	if _, err := fmt.Sscanf(*r42AB, "%d,%d,%d", &rounds, &ms, &places); err != nil {
		t.Fatal(err)
	}
	t.Logf("fmaAvailable=%v rounds=%d ms=%d placements=%d", fmaAvailable, rounds, ms, places)
	for _, row := range strings.Split(*r42Vars, ";") {
		spec, list, ok := strings.Cut(row, ":")
		if !ok {
			t.Fatalf("row %q", row)
		}
		kind, n, _ := r42Spec(t, spec)
		vars := strings.Split(list, ",")
		nv := len(vars)
		runs := make([][]func(), places)
		for j := range places {
			r41Pad(j)
			runs[j] = make([]func(), nv)
			for k := range nv {
				w := (k + j) % nv
				runs[j][w] = r42Variant(t, kind, n, vars[w])
			}
		}
		times := make([][]float64, nv)
		ratios := make([][]float64, nv)
		for round := range rounds {
			per := make([][]float64, nv)
			for j := range places {
				for k := range nv {
					w := (k + round + j) % nv
					per[w] = append(per[w], r41Time(runs[j][w], ms))
				}
			}
			for w := range nv {
				times[w] = append(times[w], r41Median(per[w]))
			}
			for w := range nv {
				ratios[w] = append(ratios[w], times[0][round]/times[w][round])
			}
		}
		for w, v := range vars {
			r := slices.Clone(ratios[w])
			slices.Sort(r)
			t.Logf("  %-8s %-16s %11.1f ns  first/variant %.3f  (%.3f–%.3f)  rounds %s",
				spec, v, r41Median(times[w]), r41Median(ratios[w]), r[0], r[len(r)-1], r41Fmt(ratios[w]))
		}
	}
}

var r42Over = flag.String("r42.over", "", "rounds,ms: run TestR42Overhead")

// TestR42Overhead times, for every row's FMA plan, the whole transform
// against its passes run in sequence on fixed buffers (the transform's own
// ping-pong, no pool), and each pass alone in that sequence's buffers.
func TestR42Overhead(t *testing.T) {
	if *r42Over == "" {
		t.Skip("set -r42.over=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*r42Over, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	for _, spec := range strings.Split(*r42Rows, ",") {
		_, n, fs := r42Spec(t, spec)
		p := r42Plan(n, fs, 1<<20-1)
		sk := p.sk
		src, dst := cmplxSignal(n), make([]complex128, n)
		buf := make([]complex128, n+setSpan)
		scr := offTheSets(buf, dst, n)
		s := len(sk.stages)
		outs := make([][]complex128, s)
		ins := make([][]complex128, s)
		in := src
		for k := range sk.stages {
			out := scr
			if (s-1-k)%2 == 0 {
				out = dst
			}
			ins[k], outs[k] = in, out
			in = out
		}
		f := []func(){
			func() { p.FFT(dst, src) },
			func() {
				for k := range sk.stages {
					sk.stages[k].pass(outs[k], ins[k], false)
				}
			},
		}
		for k := range sk.stages {
			f = append(f, func() { sk.stages[k].pass(outs[k], ins[k], false) })
		}
		tm := make([][]float64, len(f))
		for round := range rounds {
			for v := range f {
				w := (v + round) % len(f)
				tm[w] = append(tm[w], r41Time(f[w], ms))
			}
		}
		line := fmt.Sprintf("%-14s transform %8.1f  passes %8.1f  each", spec, r41Median(tm[0]), r41Median(tm[1]))
		for k := range sk.stages {
			line += fmt.Sprintf(" %7.1f", r41Median(tm[2+k]))
		}
		t.Log(line)
	}
}
