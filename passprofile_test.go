package fft

// Measurement-only harness (branch exp/passprofile, never for main): lists the
// Stockham passes of a plan and runs ONE pass, or the whole plan, in a warm
// loop so that `perf stat` (two iteration counts subtracted) and the program's
// own ns/op can be attributed per pass. Driven by ppdrv.

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

func ppPlan(n int) *skPlan {
	p := NewPlan(n)
	if p.sk == nil {
		panic(fmt.Sprintf("n=%d has no Stockham plan", n))
	}
	return p.sk
}

func TestPPList(t *testing.T) {
	if os.Getenv("PP_LIST") == "" {
		t.Skip("PP_LIST not set")
	}
	for _, n := range []int{1000, 1080, 1440, 1920, 2160} {
		p := NewPlan(n)
		s := p.sk
		fmt.Printf("n=%d stages=%d f256=%v f1024=%v f2048=%v zn=%v casc=%d fma=%v\n", n, len(s.stages), p.f256, p.f1024, p.f2048, s.zn != nil, s.cascT, s.fma)
		for k, st := range s.stages {
			fmt.Printf("  pass %d: r=%d l1=%d ido=%d split=%d wide=%v fma=%v instreamgap=%dB outstreamgap=%dB twtable=%dB\n",
				k, st.r, st.l1, st.ido, st.split, st.wide, st.fma, st.ido*16, st.ido*st.l1*16, len(st.twX)*16)
		}
	}
}

// TestPPFact times whole-plan alternatives for n (PP_FACT_N) in interleaved
// rounds: the default plan against the factorizations of PP_FACTS
// ("12,12,10;10,12,12"), the alternatives as FMA plans.
func TestPPFact(t *testing.T) {
	ns := os.Getenv("PP_FACT_N")
	if ns == "" {
		t.Skip("PP_FACT_N not set")
	}
	n, _ := strconv.Atoi(ns)
	runtime.LockOSThread()
	plans := []*skPlan{NewPlan(n).sk}
	names := []string{"default"}
	for _, f := range strings.Split(os.Getenv("PP_FACTS"), ";") {
		var fs []int
		for _, x := range strings.Split(f, ",") {
			v, _ := strconv.Atoi(x)
			fs = append(fs, v)
		}
		plans = append(plans, newSKPlanFactorsOpt(n, fs, true))
		names = append(names, f)
	}
	in := make([]complex128, n)
	out := make([]complex128, n)
	rng := rand.New(rand.NewSource(1))
	for i := range in {
		in[i] = complex(rng.Float64()-0.5, rng.Float64()-0.5)
	}
	res := make([][]float64, len(plans))
	for r := 0; r < 9; r++ {
		for i, p := range plans {
			for j := 0; j < 3000; j++ {
				p.transform(out, in, false)
			}
			const it = 60000
			t0 := time.Now()
			for j := 0; j < it; j++ {
				p.transform(out, in, false)
			}
			res[i] = append(res[i], float64(time.Since(t0).Nanoseconds())/it)
		}
	}
	for i := range plans {
		s := append([]float64{}, res[i]...)
		sort.Float64s(s)
		fmt.Printf("PPFACT n=%d %-14s median %.1f ns (min %.1f)\n", n, names[i], s[len(s)/2], s[0])
	}
}

func TestPPLoop(t *testing.T) {
	ns := os.Getenv("PP_N")
	if ns == "" {
		t.Skip("PP_N not set")
	}
	n, _ := strconv.Atoi(ns)
	pass, _ := strconv.Atoi(os.Getenv("PP_PASS")) // -1: whole plan
	iters, _ := strconv.Atoi(os.Getenv("PP_ITERS"))
	runtime.LockOSThread()
	pl := NewPlan(n)
	s := pl.sk
	in := make([]complex128, n)
	out := make([]complex128, n)
	rng := rand.New(rand.NewSource(1))
	for i := range in {
		in[i] = complex(rng.Float64()-0.5, rng.Float64()-0.5)
	}
	run := func() { pl.FFT(out, in) }
	if pass >= 0 {
		st := &s.stages[pass]
		run = func() { st.pass(out, in, false) }
	}
	for i := 0; i < 2000; i++ {
		run()
	}
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		run()
	}
	dt := time.Since(t0)
	fmt.Printf("PPRESULT n=%d pass=%d iters=%d ns/op=%.2f\n", n, pass, iters, float64(dt.Nanoseconds())/float64(iters))
}
