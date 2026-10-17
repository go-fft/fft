package fft

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Round 41: the opt-in FMA plans against the default ones, in one process.
// Every helper here is prefixed r41.

var (
	r41AB   = flag.String("r41.ab", "", "rounds,ms: run Round 41's in-process A/B (TestR41AB)")
	r41Rows = flag.String("r41.rows", "c256,c1024,c4096,c65536,c1048576,c1000,c1080,c1920,c1009,c1296,c10007,r256,r1024,r4096,r65536,r1048576,r1000,r1080,r1920,n64x64,n128x128,n256x256,n512x512,n1024x1024",
		"TestR41AB: rows, c<n> complex, r<n> RFFT, i<n> IRFFT, n<a>x<b> 2-D complex PlanN")
)

// r41Row is one timed transform in its two variants.
type r41Row struct {
	name string
	run  [2]func() // default, FMA
}

func r41MakeRow(t *testing.T, spec string) r41Row {
	kind, arg := spec[0], spec[1:]
	row := r41Row{name: spec}
	switch kind {
	case 'c', 'r', 'i':
		n, err := strconv.Atoi(arg)
		if err != nil {
			t.Fatal(err)
		}
		for v, fma := range []bool{false, true} {
			switch kind {
			case 'c':
				p := newPlanF(n, fma)
				src, dst := cmplxSignal(n), make([]complex128, n)
				row.run[v] = func() { p.FFT(dst, src) }
			case 'r':
				p := newRealPlanF(n, fma)
				src, dst := realSignal(n), make([]complex128, n/2+1)
				row.run[v] = func() { p.RFFT(dst, src) }
			default:
				p := newRealPlanF(n, fma)
				spec := p.RFFT(make([]complex128, n/2+1), realSignal(n))
				dst := make([]float64, n)
				row.run[v] = func() { p.IRFFT(dst, spec) }
			}
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
		for v, fma := range []bool{false, true} {
			p := newPlanNF(shape, fma)
			src, dst := cmplxSignal(p.size), make([]complex128, p.size)
			row.run[v] = func() { p.FFT(dst, src) }
		}
	default:
		t.Fatalf("row %q", spec)
	}
	return row
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
	slices.Sort(per)
	return per[len(per)/2]
}

// TestR41AB times every row's default and FMA plans, the order rotated every
// round, and prints default ÷ FMA per round and its median.
func TestR41AB(t *testing.T) {
	if *r41AB == "" {
		t.Skip("set -r41.ab=rounds,ms")
	}
	var rounds, ms int
	if _, err := fmt.Sscanf(*r41AB, "%d,%d", &rounds, &ms); err != nil {
		t.Fatal(err)
	}
	t.Logf("fmaAvailable=%v", fmaAvailable)
	var rows []r41Row
	for _, s := range strings.Split(*r41Rows, ",") {
		rows = append(rows, r41MakeRow(t, s))
	}
	ratios := make([][]float64, len(rows))
	times := make([][2][]float64, len(rows))
	for round := range rounds {
		for i, row := range rows {
			var ns [2]float64
			for k := range 2 {
				v := (k + round) % 2
				ns[v] = r41Time(row.run[v], ms)
			}
			ratios[i] = append(ratios[i], ns[0]/ns[1])
			times[i][0] = append(times[i][0], ns[0])
			times[i][1] = append(times[i][1], ns[1])
		}
	}
	med := func(x []float64) float64 {
		y := slices.Clone(x)
		slices.Sort(y)
		return y[len(y)/2]
	}
	for i, row := range rows {
		r := slices.Clone(ratios[i])
		slices.Sort(r)
		t.Logf("%-12s default %10.1f ns  fma %10.1f ns  default/fma %.3f  (%.3f–%.3f)  rounds %v",
			row.name, med(times[i][0]), med(times[i][1]), med(ratios[i]), r[0], r[len(r)-1], fmtRatios(ratios[i]))
	}
}

func fmtRatios(x []float64) string {
	s := make([]string, len(x))
	for i, v := range x {
		s[i] = strconv.FormatFloat(v, 'f', 3, 64)
	}
	return strings.Join(s, " ")
}
