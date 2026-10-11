package kernels

// Round 37 (Haswell): what the 512-bit untangle's call with quads = 0
// (Round 31) costs a CPU without AVX-512, in one process, rotated.

import (
	goflag "flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	hwCall     = goflag.String("hw.call", "", "TestHwUntangleCall: rounds,iters")
	hwCallLens = goflag.String("hw.calllens", "128,256,512,540", "TestHwUntangleCall: half lengths m")
)

// TestHwUntangleCall times Untangle and Retangle (the 512-bit kernel called
// with quads = 0, then the AVX2 one) against the AVX2 kernel called alone.
func TestHwUntangleCall(t *testing.T) {
	if *hwCall == "" {
		t.Skip("set -hw.call=rounds,iters")
	}
	var rounds, iters int
	if _, err := fmt.Sscanf(*hwCall, "%d,%d", &rounds, &iters); err != nil {
		t.Fatal(err)
	}
	for _, s := range strings.Split(*hwCallLens, ",") {
		m, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		pairs := (m - 1) / 2 / 2
		dst := make([]complex128, m+1)
		z := make([]complex128, m)
		tw := make([]complex128, m+2)
		for i := range z {
			z[i] = complex(float64(i%7), float64(i%5))
		}
		for i := range tw {
			tw[i] = complex(float64(i%3)*0.25, float64(i%4)*0.125)
		}
		hv := [4]float64{0.5, 0.5, 0.5, 0.5}
		fns := []func(){
			func() { Untangle(dst, z, tw, m) },
			func() { smallUntangleAVX2(&dst[0], &z[0], &tw[0], &skFwd[0][0], m, pairs) },
			func() { Retangle(z, dst, tw, m, 0.5) },
			func() { smallRetangleAVX2(&z[0], &dst[0], &tw[0], &skFwd[0][0], &hv, m, pairs) },
		}
		times := make([][]float64, len(fns))
		for r := range rounds {
			for q := range fns {
				v := (q + r) % len(fns)
				t0 := time.Now()
				for range iters {
					fns[v]()
				}
				times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(iters))
			}
		}
		med := func(x []float64) float64 { y := slices.Clone(x); slices.Sort(y); return y[len(y)/2] }
		ratio := func(a, b []float64) string {
			r := make([]float64, len(a))
			for i := range r {
				r[i] = a[i] / b[i]
			}
			return fmt.Sprintf("%.3f (%.3f-%.3f)", med(r), slices.Min(r), slices.Max(r))
		}
		fmt.Printf("HWCALL m=%d untangle %.1f / AVX2 alone %.1f  ratio %s | retangle %.1f / %.1f  ratio %s\n", m,
			med(times[0]), med(times[1]), ratio(times[0], times[1]), med(times[2]), med(times[3]), ratio(times[2], times[3]))
	}
}
