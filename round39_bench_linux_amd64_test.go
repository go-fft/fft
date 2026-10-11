package fft

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

// Round 39 (Cascade Lake): the 2-D and large real rows, located. Every
// helper here is prefixed cl3.

var (
	cl3AB     = flag.String("cl3.ab", "", "rounds,ms: run Round 39's in-process tests")
	cl3Shapes = flag.String("cl3.shapes", "512x512", "Round 39: N-D shapes, comma separated")
	cl3Pairs  = flag.Int("cl3.pairs", 4, "TestCl3Pages: Go-heap buffer pairs")
	cl3Scav   = flag.Bool("cl3.scav", false, "TestCl3Pages: churn and return the heap to the OS before allocating the Go-heap pairs")
	cl3Parts  = flag.Bool("cl3.parts", true, "TestCl3Pages: also time each axis alone")
)

func cl3Rounds(t *testing.T) (rounds, ms int) {
	if *cl3AB == "" {
		t.Skip("set -cl3.ab=rounds,ms")
	}
	if _, err := fmt.Sscanf(*cl3AB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	return rounds, ms
}

func cl3Shape(t *testing.T, s string) []int {
	var shape []int
	for _, f := range strings.Split(s, "x") {
		n, err := strconv.Atoi(f)
		if err != nil {
			t.Fatal(err)
		}
		shape = append(shape, n)
	}
	return shape
}

const (
	cl3MadvDontneed  = 4
	cl3MadvHugepage  = 14
	cl3MadvNohuge    = 15
	cl3MadvCollapse  = 25
	cl3MadvDontdump  = 16
	cl3MadvDodump    = 17
	cl3Huge          = 2 << 20
	cl3PageAlignMask = 4095
)

func cl3Madvise(addr, length uintptr, advice int) error {
	_, _, e := syscall.Syscall(syscall.SYS_MADVISE, addr, length, uintptr(advice))
	if e != 0 {
		return e
	}
	return nil
}

// cl3Mmap returns n complex128 outside the Go heap, 2 MB aligned, backed by
// 2 MB pages (huge, MADV_HUGEPAGE then MADV_COLLAPSE) or by 4 KB pages
// (MADV_NOHUGEPAGE), touched.
func cl3Mmap(t *testing.T, n int, huge bool) []complex128 {
	size := uintptr(n * 16)
	b, err := syscall.Mmap(-1, 0, int(size)+cl3Huge, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	base := uintptr(unsafe.Pointer(&b[0]))
	al := (base + cl3Huge - 1) &^ (cl3Huge - 1)
	adv := cl3MadvNohuge
	if huge {
		adv = cl3MadvHugepage
	}
	if err := cl3Madvise(al, size, adv); err != nil {
		t.Fatal(err)
	}
	s := unsafe.Slice((*complex128)(unsafe.Pointer(&b[al-base])), n)
	for i := range s {
		s[i] = complex(float64(i%13)*0.1, float64(i%7)*0.2)
	}
	if huge {
		if err := cl3Madvise(al, size, cl3MadvCollapse); err != nil {
			t.Logf("MADV_COLLAPSE: %v", err)
		}
	}
	return s
}

// cl3HugeKB reports how many KB of b's page-aligned interior the kernel maps
// with huge pages. It marks that range MADV_DONTDUMP, which splits it into a
// VMA of its own and changes nothing but core dumps, reads that VMA's AnonHugePages from /proc/self/smaps.
// Call it after the timing.
func cl3HugeKB(t *testing.T, b []complex128) (huge, total int) {
	lo := uintptr(unsafe.Pointer(&b[0]))
	hi := lo + uintptr(len(b)*16)
	lo = (lo + cl3PageAlignMask) &^ cl3PageAlignMask
	hi &^= cl3PageAlignMask
	if err := cl3Madvise(lo, hi-lo, cl3MadvDontdump); err != nil {
		t.Fatal(err)
	}
	defer cl3Madvise(lo, hi-lo, cl3MadvDodump)
	f, err := os.Open("/proc/self/smaps")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	want := fmt.Sprintf("%x-%x ", lo, hi)
	sc := bufio.NewScanner(f)
	in := false
	for sc.Scan() {
		l := sc.Text()
		if fs := strings.Fields(l); len(fs) > 0 && !strings.HasSuffix(fs[0], ":") && strings.Contains(fs[0], "-") {
			in = strings.HasPrefix(l, want)
			continue
		}
		if in && strings.HasPrefix(l, "AnonHugePages:") {
			fs := strings.Fields(l)
			huge, _ = strconv.Atoi(fs[1])
			return huge, int(hi-lo) >> 10
		}
	}
	return -1, int(hi-lo) >> 10
}

// TestCl3Pages times an N-D plan on Go-heap buffer pairs (several
// allocations) and on buffers whose page size is forced: 4 KB or 2 MB for
// src and dst independently. With -cl3.parts each variant's columns (src to
// dst) and rows (in place on dst) are also timed alone. After the timing it
// reports how much of each Go-heap buffer the kernel had on huge pages.
func TestCl3Pages(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	for _, s := range strings.Split(*cl3Shapes, ",") {
		shape := cl3Shape(t, s)
		p := NewPlanN(shape...)
		size := p.size
		type vr struct {
			name     string
			src, dst []complex128
		}
		var vs []vr
		if *cl3Scav {
			junk := make([][]byte, 0, 64)
			for range 64 {
				junk = append(junk, make([]byte, 1<<20))
			}
			junk = nil
			runtime.GC()
			debug.FreeOSMemory()
		}
		var keep [][]byte
		for j := range *cl3Pairs {
			keep = append(keep, make([]byte, (j*37%11+1)<<16))
			src := benchComplex(size)
			dst := make([]complex128, size)
			vs = append(vs, vr{fmt.Sprintf("heap%d", j), src, dst})
		}
		_ = keep
		m4s, m4d := cl3Mmap(t, size, false), cl3Mmap(t, size, false)
		m2s, m2d := cl3Mmap(t, size, true), cl3Mmap(t, size, true)
		vs = append(vs, vr{"4K/4K", m4s, m4d}, vr{"2M/2M", m2s, m2d}, vr{"src4K/dst2M", m4s, m2d}, vr{"src2M/dst4K", m2s, m4d})
		var fns []func()
		for _, v := range vs {
			fns = append(fns, func() { p.FFT(v.dst, v.src) })
		}
		if *cl3Parts {
			for _, v := range vs {
				fns = append(fns, func() { p.transformAxis(v.dst, v.src, 0, false) })
			}
			for _, v := range vs {
				last := len(shape) - 1
				fns = append(fns, func() { p.transformAxis(v.dst, v.dst, last, false) })
			}
		}
		times := comp2Rotate(fns, rounds, ms)
		part := func(k int) string {
			if !*cl3Parts {
				return ""
			}
			c, r := median(times[len(vs)+k]), median(times[2*len(vs)+k])
			return fmt.Sprintf("cols %9.0f rows %9.0f", c, r)
		}
		for k, v := range vs {
			hs, ts := cl3HugeKB(t, v.src)
			hd, td := cl3HugeKB(t, v.dst)
			fmt.Printf("PAGES %-9s %-12s whole %9.0f ns [%9.0f %9.0f] %s  src huge %6d/%6d KB dst huge %6d/%6d KB src%%2M=%7x dst%%2M=%7x\n",
				s, v.name, median(times[k]), slices.Min(times[k]), slices.Max(times[k]), part(k), hs, ts, hd, td,
				uintptr(unsafe.Pointer(&v.src[0]))%cl3Huge, uintptr(unsafe.Pointer(&v.dst[0]))%cl3Huge)
		}
	}
}

// TestCl3FrameRowsAB times N-D plans with their last axis on the frame kernels
// (zmFrameRows, Round 36) and off it, in one process, rotated.
func TestCl3FrameRowsAB(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	defer func(v bool) { zmFrameRows = v }(zmFrameRows)
	for _, s := range strings.Split(*cl3Shapes, ",") {
		shape := cl3Shape(t, s)
		p := NewPlanN(shape...)
		src, d0, d1 := benchComplex(p.size), make([]complex128, p.size), make([]complex128, p.size)
		fns := []func(){
			func() { zmFrameRows = false; p.FFT(d0, src) },
			func() { zmFrameRows = true; p.FFT(d1, src) },
		}
		times := comp2Rotate(fns, rounds, ms)
		var q []float64
		for r := range times[0] {
			q = append(q, times[0][r]/times[1][r])
		}
		fmt.Printf("FRAMEROWS %-9s off %10.0f ns  on %10.0f ns  off/on %.3f [%.3f %.3f]  rows zn=%v\n", s,
			median(times[0]), median(times[1]), median(q), slices.Min(q), slices.Max(q), p.axes[len(shape)-1].sk != nil && p.axes[len(shape)-1].sk.zn != nil)
	}
}

// cl3StripLines is stripLines with its scratch given.
func (p *PlanN) cl3StripLines(dst, src []complex128, ax int, buf []complex128) {
	n, st := p.shape[ax], p.stride[ax]
	stages, tws := p.axes[ax].sk.stages, p.strips[ax]
	sw := stripWidth(n)
	per := (st + sw - 1) / sw
	a, b := buf[:n*sw], buf[n*sw:2*n*sw]
	last := len(stages) - 1
	for s := 0; s < p.size/(n*st)*per; s++ {
		c0 := s % per * sw
		w := min(sw, st-c0)
		base := s/per*n*st + c0
		in, sIn := src[base:], st
		for k := range stages {
			out, sOut := a, w
			if k == last {
				out, sOut = dst[base:], st
			}
			stages[k].passBatch(out, in, tws[k][0], w, sIn, sOut, false)
			in, sIn = out, sOut
			a, b = b, a
		}
	}
}

var cl3Offs = flag.String("cl3.offs", "0,64,4096,8192,32768,65536,131072,262144", "TestCl3Scratch: byte offsets of the strip scratch in a 2 MB page")

// TestCl3Scratch times the strip (column) step of an N-D plan, src and dst
// on 2 MB pages, with its scratch at given offsets in a 2 MB page, on 4 KB
// pages, and from the Go heap.
func TestCl3Scratch(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	if *cl3Scav {
		junk := make([][]byte, 0, 64)
		for range 64 {
			junk = append(junk, make([]byte, 1<<20))
		}
		junk = nil
		runtime.GC()
		debug.FreeOSMemory()
	}
	for _, s := range strings.Split(*cl3Shapes, ",") {
		shape := cl3Shape(t, s)
		p := NewPlanN(shape...)
		n := shape[0]
		need := 2 * n * stripWidth(n)
		src, dst := cl3Mmap(t, p.size, true), cl3Mmap(t, p.size, true)
		big := cl3Mmap(t, 2*cl3Huge/16, true)
		small := cl3Mmap(t, need, false)
		heap := make([]complex128, need)
		pool := p.bufs.Get().(*[]complex128)
		names := []string{"heap", "pool", "4K"}
		bufs := [][]complex128{heap, (*pool)[:need], small}
		for _, o := range csInts(t, *cl3Offs) {
			names = append(names, fmt.Sprintf("2M+%d", o))
			bufs = append(bufs, big[o/16:o/16+need])
		}
		var fns []func()
		for _, b := range bufs {
			fns = append(fns, func() { p.cl3StripLines(dst, src, 0, b) })
		}
		times := comp2Rotate(fns, rounds, ms)
		for k, name := range names {
			fmt.Printf("SCRATCH %-9s factors %v %-10s cols %9.0f ns [%9.0f %9.0f] buf%%2M=%7x\n", s, p.axes[0].sk.factors(), name,
				median(times[k]), slices.Min(times[k]), slices.Max(times[k]), uintptr(unsafe.Pointer(&bufs[k][0]))%cl3Huge)
		}
	}
}

// cl3Chase follows a random cycle through idx and returns where it ends.
func cl3Chase(idx []int32, steps int) int32 {
	j := int32(0)
	for range steps {
		j = idx[j]
	}
	return j
}

var cl3Sink float64

// TestCl3Mem times, in the same rotated rounds as a 512² plan's columns and
// rows, three probes of the memory system with no FFT arithmetic: a random
// pointer chase over 64 MB (latency), a sequential read of 64 MB
// (bandwidth), and the columns' own access pattern: 256 bytes of every
// 8 KB row of a 4 MB array, strip after strip.
func TestCl3Mem(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	p := NewPlanN(512, 512)
	src, dst := cl3Mmap(t, p.size, true), cl3Mmap(t, p.size, true)
	big := cl3Mmap(t, 4<<20, true) // 64 MB
	idx := unsafe.Slice((*int32)(unsafe.Pointer(&big[0])), 16<<20)
	perm := make([]int32, len(idx))
	for i := range perm {
		perm[i] = int32(i)
	}
	r := uint64(88172645463325252)
	for i := len(perm) - 1; i > 0; i-- {
		r ^= r << 13
		r ^= r >> 7
		r ^= r << 17
		j := int(r % uint64(i+1))
		perm[i], perm[j] = perm[j], perm[i]
	}
	for i := range perm {
		idx[perm[i]] = perm[(i+1)%len(perm)]
	}
	const chase = 1 << 16
	fns := []func(){
		func() { p.transformAxis(dst, src, 0, false) },
		func() { p.transformAxis(dst, dst, 1, false) },
		func() { cl3Sink += float64(cl3Chase(idx, chase)) },
		func() {
			s := 0.0
			for _, v := range big {
				s += real(v)
			}
			cl3Sink += s
		},
		func() {
			s := 0.0
			for c := 0; c < 512; c += 16 {
				for row := range 512 {
					for _, v := range src[row*512+c : row*512+c+16] {
						s += real(v)
					}
				}
			}
			cl3Sink += s
		},
	}
	names := []string{"cols", "rows", "chase ns/load", "seq GB/s", "colread"}
	times := comp2Rotate(fns, rounds, ms)
	for k, name := range names {
		v := median(times[k])
		switch k {
		case 2:
			v /= chase
		case 3:
			v = float64(len(big)*16) / v
		}
		fmt.Printf("MEM %-14s %12.2f  [%.0f %.0f ns]\n", name, v, slices.Min(times[k]), slices.Max(times[k]))
	}
}

// TestCl3Resident times sweeps over buffers of 16 KB to 4 MB that stay in
// one level of the cache hierarchy: read and write back every 64-byte line,
// ns per line.
func TestCl3Resident(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	sizes := []int{16 << 10, 128 << 10, 512 << 10, 768 << 10, 4 << 20}
	buf := cl3Mmap(t, 4<<20/16*2, true)
	var fns []func()
	for _, sz := range sizes {
		b := buf[:sz/16]
		fns = append(fns, func() {
			for i := 0; i < len(b); i += 4 {
				b[i] += 1
			}
		})
	}
	// equal work per call: repeat smaller buffers
	for k, sz := range sizes {
		rep := (4 << 20) / sz
		f := fns[k]
		fns[k] = func() {
			for range rep {
				f()
			}
		}
	}
	times := comp2Rotate(fns, rounds, ms)
	for k, sz := range sizes {
		fmt.Printf("RESIDENT %7d KB %6.2f ns/line [%.2f %.2f]\n", sz>>10, median(times[k])/float64(4<<20/64),
			slices.Min(times[k])/float64(4<<20/64), slices.Max(times[k])/float64(4<<20/64))
	}
}

// cl3StripPasses is cl3StripLines with each pass's time accumulated in
// acc[k].
func (p *PlanN) cl3StripPasses(dst, src []complex128, ax int, buf []complex128, acc []time.Duration) {
	n, st := p.shape[ax], p.stride[ax]
	stages, tws := p.axes[ax].sk.stages, p.strips[ax]
	sw := stripWidth(n)
	per := (st + sw - 1) / sw
	a, b := buf[:n*sw], buf[n*sw:2*n*sw]
	last := len(stages) - 1
	for s := 0; s < p.size/(n*st)*per; s++ {
		c0 := s % per * sw
		w := min(sw, st-c0)
		base := s/per*n*st + c0
		in, sIn := src[base:], st
		for k := range stages {
			out, sOut := a, w
			if k == last {
				out, sOut = dst[base:], st
			}
			t0 := time.Now()
			stages[k].passBatch(out, in, tws[k][0], w, sIn, sOut, false)
			acc[k] += time.Since(t0)
			in, sIn = out, sOut
			a, b = b, a
		}
	}
}

// TestCl3StripPasses times the column step of each -cl3.shapes plan pass by
// pass, src and dst on 2 MB pages, with a column-write probe beside it.
func TestCl3StripPasses(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	_ = ms
	for _, s := range strings.Split(*cl3Shapes, ",") {
		shape := cl3Shape(t, s)
		p := NewPlanN(shape...)
		n := shape[0]
		src, dst := cl3Mmap(t, p.size, true), cl3Mmap(t, p.size, true)
		buf := make([]complex128, 2*n*stripWidth(n))
		ns := len(p.axes[0].sk.stages)
		per := make([][]float64, ns+2)
		for range rounds {
			acc := make([]time.Duration, ns)
			const it = 20
			for range it {
				p.cl3StripPasses(dst, src, 0, buf, acc)
			}
			for k := range ns {
				per[k] = append(per[k], float64(acc[k].Nanoseconds())/it)
			}
			t0 := time.Now()
			for range it {
				for c := 0; c < shape[1]; c += 16 {
					for row := range n {
						d := dst[row*shape[1]+c : row*shape[1]+c+16]
						for i := range d {
							d[i] = complex(float64(i), 0)
						}
					}
				}
			}
			per[ns] = append(per[ns], float64(time.Since(t0).Nanoseconds())/it)
			t0 = time.Now()
			for range it {
				p.cl3StripLines(dst, src, 0, buf)
			}
			per[ns+1] = append(per[ns+1], float64(time.Since(t0).Nanoseconds())/it)
		}
		for k := range ns {
			st := p.axes[0].sk.stages[k]
			fmt.Printf("STRIPPASS %-9s pass %d r=%d l1=%d %9.0f ns [%9.0f %9.0f]\n", s, k, st.r, st.l1, median(per[k]), slices.Min(per[k]), slices.Max(per[k]))
		}
		fmt.Printf("STRIPPASS %-9s colwrite      %9.0f ns [%9.0f %9.0f]\n", s, median(per[ns]), slices.Min(per[ns]), slices.Max(per[ns]))
		fmt.Printf("STRIPPASS %-9s whole step    %9.0f ns [%9.0f %9.0f]\n", s, median(per[ns+1]), slices.Min(per[ns+1]), slices.Max(per[ns+1]))
	}
}

// cl3StripVariant is stripLines with the strip width sw, and optionally the
// strip copied out of src into the scratch first (gather: the first pass
// then reads it at stride sw) and the last pass written into the scratch and
// copied into dst after it (scatter).
func (p *PlanN) cl3StripVariant(dst, src []complex128, ax int, buf []complex128, sw int, gather, scatter bool) {
	n, st := p.shape[ax], p.stride[ax]
	stages, tws := p.axes[ax].sk.stages, p.strips[ax]
	per := (st + sw - 1) / sw
	last := len(stages) - 1
	for s := 0; s < p.size/(n*st)*per; s++ {
		a, b := buf[:n*sw], buf[n*sw:2*n*sw]
		c0 := s % per * sw
		w := min(sw, st-c0)
		base := s/per*n*st + c0
		in, sIn := src[base:], st
		if gather {
			for i := range n {
				copy(b[i*w:(i+1)*w], src[base+i*st:base+i*st+w])
			}
			in, sIn = b, w
		}
		for k := range stages {
			out, sOut := a, w
			if k == last && !scatter {
				out, sOut = dst[base:], st
			}
			stages[k].passBatch(out, in, tws[k][0], w, sIn, sOut, false)
			in, sIn = out, sOut
			a, b = b, a
		}
		if scatter {
			for i := range n {
				copy(dst[base+i*st:base+i*st+w], in[i*w:(i+1)*w])
			}
		}
	}
}

var cl3Vars = flag.String("cl3.vars", "16,32,8,16g,16gs,32g,32gs,16s", "TestCl3StripVars: strip widths, g = gather in, s = scatter out")

// TestCl3StripVars times the column step of each -cl3.shapes plan with the
// -cl3.vars strip variants, src and dst on the Go heap, rotated; the first
// variant is the reference, which must be main's.
func TestCl3StripVars(t *testing.T) {
	rounds, ms := cl3Rounds(t)
	for _, s := range strings.Split(*cl3Shapes, ",") {
		shape := cl3Shape(t, s)
		p := NewPlanN(shape...)
		src, dst := benchComplex(p.size), make([]complex128, p.size)
		names := strings.Split(*cl3Vars, ",")
		for ax := range len(shape) - 1 {
			if p.strips[ax] == nil {
				continue
			}
			n := shape[ax]
			var fns []func()
			for _, v := range names {
				w, _ := strconv.Atoi(strings.TrimRight(v, "gs"))
				g, sc := strings.Contains(v, "g"), strings.Contains(v, "s")
				buf := make([]complex128, 2*n*w)
				fns = append(fns, func() { p.cl3StripVariant(dst, src, ax, buf, w, g, sc) })
			}
			ref := make([]complex128, p.size)
			p.stripLines(ref, src, ax, 0, (p.size/(n*p.stride[ax]))*((p.stride[ax]+stripWidth(n)-1)/stripWidth(n)), false)
			for k, f := range fns {
				clear(dst)
				f()
				if !slices.Equal(dst, ref) {
					t.Fatalf("%s variant %s differs from stripLines", s, names[k])
				}
			}
			times := comp2Rotate(fns, rounds, ms)
			for k, name := range names {
				var q []float64
				for r := range times[0] {
					q = append(q, times[0][r]/times[k][r])
				}
				fmt.Printf("STRIPVAR %-11s ax %d %-6s %9.0f ns [%9.0f %9.0f]  ref/this %.3f [%.3f %.3f]\n", s, ax, name, median(times[k]),
					slices.Min(times[k]), slices.Max(times[k]), median(q), slices.Min(q), slices.Max(q))
			}
		}
	}
}

var cl3Export = flag.String("cl3.export", "", "TestCl3Export: directory")

// TestCl3Export writes, for N-D shapes this round changes, a random input
// and go-fft's FFTN and IFFTN of it with this round's changes off ("old":
// stripWidth, the rows off the frame) and on ("new"), and the same for
// RealPlan2's RFFT, as raw little-endian float64, for numpy to check on the
// same bytes (acc39.py).
func TestCl3Export(t *testing.T) {
	if *cl3Export == "" {
		t.Skip("set -cl3.export=dir")
	}
	r := rand.New(rand.NewPCG(39, 2026))
	write := func(name string, x []float64) {
		b := make([]byte, 0, 8*len(x))
		for _, v := range x {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v))
		}
		if err := os.WriteFile(filepath.Join(*cl3Export, name), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cplx := func(x []complex128) []float64 { return unsafe.Slice((*float64)(unsafe.Pointer(&x[0])), 2*len(x)) }
	defer func(w, f bool) { cl3WideStrips, zmFrameRows = w, f }(cl3WideStrips, zmFrameRows)
	set := func(on bool) { cl3WideStrips, zmFrameRows = on, on }
	tags := map[bool]string{false: "old", true: "new"}
	for _, s := range []string{"32x32", "64x64", "128x128", "256x256", "512x512", "1024x1024", "512x1024", "300x1000", "64x4096", "64x64x64", "8x64x64"} {
		shape := cl3Shape(t, s)
		size := shapeProduct(shape...)
		x := make([]complex128, size)
		for i := range x {
			x[i] = complex(r.NormFloat64(), r.NormFloat64())
		}
		write("in_nd_"+s+".bin", cplx(x))
		write("in_ind_"+s+".bin", cplx(x))
		for _, on := range []bool{false, true} {
			set(on)
			p := NewPlanN(shape...)
			write(fmt.Sprintf("%s_nd_%s.bin", tags[on], s), cplx(p.FFT(make([]complex128, size), x)))
			write(fmt.Sprintf("%s_ind_%s.bin", tags[on], s), cplx(p.IFFT(make([]complex128, size), x)))
			fmt.Printf("EXPORT %s %s widths %v\n", s, tags[on], p.sw)
		}
	}
	for _, s := range []string{"512x1024", "1024x1024"} {
		shape := cl3Shape(t, s)
		x := make([]float64, shape[0]*shape[1])
		for i := range x {
			x[i] = r.NormFloat64()
		}
		write("in_r2_"+s+".bin", x)
		for _, on := range []bool{false, true} {
			set(on)
			p := NewRealPlan2(shape[0], shape[1])
			write(fmt.Sprintf("%s_r2_%s.bin", tags[on], s), cplx(p.RFFT(make([]complex128, shape[0]*(shape[1]/2+1)), x)))
			fmt.Printf("EXPORT r2 %s %s widths %v\n", s, tags[on], p.col.sw)
		}
	}
}
