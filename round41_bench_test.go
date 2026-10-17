package fft

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Round 41: the opt-in FMA plans against the default ones, in one process.
// Every helper here is prefixed r41.

var (
	r41AB   = flag.String("r41.ab", "", "rounds,ms,placements: run Round 41's in-process A/B (TestR41AB)")
	r41Rows = flag.String("r41.rows", "c256,c1024,c4096,c65536,c1048576,c1000,c1080,c1920,c1009,c1296,c10007,r256,r1024,r4096,r65536,r1048576,r1000,r1080,r1920,n64x64,n128x128,n256x256,n512x512,n1024x1024",
		"TestR41AB: rows, c<n> complex, r<n> RFFT, i<n> IRFFT, n<a>x<b> 2-D complex PlanN")
)

// r41Sink keeps the padding allocations alive.
var r41Sink [][]byte

// r41Pad shifts every small size class's next allocation by a pseudo-random
// number of objects (placement 0: none), Round 32's znPad, so that the plan
// built next lands elsewhere in the heap.
func r41Pad(j int) {
	if j == 0 {
		return
	}
	r := rand.New(rand.NewPCG(uint64(j), 41))
	for _, s := range []int{16, 32, 48, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2304, 2688, 3072, 4096, 5376, 6144, 8192, 16384, 32768} {
		for range r.IntN(8) {
			r41Sink = append(r41Sink, make([]byte, s))
		}
	}
}

// r41Build returns one placement of a row's transform in a variant: its own
// plan (the half plan of a real one included, never the cached one) and
// buffers.
func r41Build(t *testing.T, spec string, fma bool) func() {
	kind, arg := spec[0], spec[1:]
	switch kind {
	case 'c', 'r', 'i':
		n, err := strconv.Atoi(arg)
		if err != nil {
			t.Fatal(err)
		}
		switch kind {
		case 'c':
			p := newPlanF(n, fma)
			src, dst := cmplxSignal(n), make([]complex128, n)
			return func() { p.FFT(dst, src) }
		case 'r':
			p := r41RealPlan(n, fma)
			src, dst := realSignal(n), make([]complex128, n/2+1)
			return func() { p.RFFT(dst, src) }
		default:
			p := r41RealPlan(n, fma)
			spec := p.RFFT(make([]complex128, n/2+1), realSignal(n))
			dst := make([]float64, n)
			return func() { p.IRFFT(dst, spec) }
		}
	case 'n':
		var shape []int
		for _, f := range strings.Split(arg, "x") {
			n, err := strconv.Atoi(f)
			if err != nil {
				t.Fatal(err)
			}
			shape = append(shape, n)
		}
		p := newPlanNF(shape, fma)
		src, dst := cmplxSignal(p.size), make([]complex128, p.size)
		return func() { p.FFT(dst, src) }
	}
	t.Fatalf("row %q", spec)
	return nil
}

// r41RealPlan is newRealPlanF with a half (or full) plan of its own.
func r41RealPlan(n int, fma bool) *RealPlan {
	p := newRealPlanF(n, fma)
	if p.half != nil {
		p.half = newPlanF(n/2, fma)
	} else {
		p.full = newPlanF(n, fma)
	}
	return p
}

// r41Time runs f for about ms milliseconds and returns its time per call in
// ns: a calibration call count first, then timed batches, the median batch.
func r41Time(f func(), ms int) float64 {
	f()
	reps := 1
	for {
		t0 := time.Now()
		for range reps {
			f()
		}
		if time.Since(t0) > 200*time.Microsecond {
			break
		}
		reps *= 2
	}
	var per []float64
	end := time.Now().Add(time.Duration(ms) * time.Millisecond)
	for time.Now().Before(end) || len(per) < 3 {
		t0 := time.Now()
		for range reps {
			f()
		}
		per = append(per, float64(time.Since(t0).Nanoseconds())/float64(reps))
	}
	return r41Median(per)
}

func r41Median(x []float64) float64 {
	y := slices.Clone(x)
	slices.Sort(y)
	if len(y)%2 == 1 {
		return y[len(y)/2]
	}
	return (y[len(y)/2-1] + y[len(y)/2]) / 2
}

// TestR41AB builds every row's default and FMA transforms at several heap
// placements, then times them in rounds, the variant order rotated every
// round and placement, and prints per row the median over rounds of
// default ÷ FMA (each round's ratio being the ratio of its medians over the
// placements), its range, and each variant's median time.
func TestR41AB(t *testing.T) {
	if *r41AB == "" {
		t.Skip("set -r41.ab=rounds,ms,placements")
	}
	var rounds, ms, places int
	if _, err := fmt.Sscanf(*r41AB, "%d,%d,%d", &rounds, &ms, &places); err != nil {
		t.Fatal(err)
	}
	t.Logf("fmaAvailable=%v rounds=%d ms=%d placements=%d", fmaAvailable, rounds, ms, places)
	specs := strings.Split(*r41Rows, ",")
	runs := make([][][2]func(), len(specs)) // [row][placement][variant]
	for j := range places {
		for i, s := range specs {
			r41Pad(j)
			var v [2]func()
			// Alternate which variant is allocated first.
			for k := range 2 {
				w := (k + j) % 2
				v[w] = r41Build(t, s, w == 1)
			}
			runs[i] = append(runs[i], v)
		}
	}
	ratios := make([][]float64, len(specs))
	times := make([][2][]float64, len(specs))
	for round := range rounds {
		for i := range specs {
			var per [2][]float64
			for j, v := range runs[i] {
				for k := range 2 {
					w := (k + round + j) % 2
					per[w] = append(per[w], r41Time(v[w], ms))
				}
			}
			d, f := r41Median(per[0]), r41Median(per[1])
			ratios[i] = append(ratios[i], d/f)
			times[i][0] = append(times[i][0], d)
			times[i][1] = append(times[i][1], f)
		}
	}
	for i, s := range specs {
		r := slices.Clone(ratios[i])
		slices.Sort(r)
		t.Logf("%-12s default %11.1f ns  fma %11.1f ns  default/fma %.3f  (%.3f–%.3f)  rounds %s",
			s, r41Median(times[i][0]), r41Median(times[i][1]), r41Median(ratios[i]), r[0], r[len(r)-1], r41Fmt(ratios[i]))
	}
}

func r41Fmt(x []float64) string {
	s := make([]string, len(x))
	for i, v := range x {
		s[i] = strconv.FormatFloat(v, 'f', 3, 64)
	}
	return strings.Join(s, " ")
}
