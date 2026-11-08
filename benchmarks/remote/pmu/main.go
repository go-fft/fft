// Command pmu drives the go-fft vs FFTW comparison on a Linux host that has
// neither Go nor Python and a readable PMU (fractal, Zen 4). It is shipped
// next to the harness it drives (benchmarks/remote) and cross-compiled:
//
//	GOWORK=off GOOS=linux GOARCH=amd64 go build -o pmu ./remote/pmu
//
// Subcommands:
//
//	pmu parity -cpu 2 -sizes 1024,65536 -rounds 7 [-fma] [-load 1.5]
//	    Interleaved rounds, one process at a time pinned with taskset,
//	    GOMAXPROCS=1; prints the per-size median and min ns/op of go-fft and
//	    FFTW and their ratio. Refuses to start above the load gate.
//	pmu stat -cpu 2 -n 1024 -a 200000 -b 1000000 -events cycles,instructions,...
//	    perf stat of both implementations at two iteration counts; the
//	    per-execute count is (count(b)-count(a))/(b-a), so planning, warm-up
//	    and process start-up cancel. Events go in groups of <=4 next to
//	    cycles and instructions so nothing is multiplexed.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
)

var (
	benchBin  = flag.String("bench", os.ExpandEnv("$HOME/gofft-bench/runpkg/bench.amd64"), "go test binary")
	fftwBin   = flag.String("fftw", os.ExpandEnv("$HOME/gofft-bench/runpkg/fftw_one"), "fftw_one binary")
	benchName = flag.String("benchname", "BenchmarkComplex_GoFFT", "go benchmark to drive (BenchmarkPow2_GoFFT reaches 512 and 2,048)")
	cpu       = flag.String("cpu", "2", "cpu to pin to (its SMT sibling must be idle)")
	fma       = flag.Bool("fma", false, "GOFFT_FMA=1 (opt-in FMA plans)")
	loadMax   = flag.Float64("load", 1.5, "refuse to run above this 1-minute load")
	sizes     = flag.String("sizes", "1024,65536", "complex sizes")
	rounds    = flag.Int("rounds", 7, "interleaved rounds")
	secs      = flag.Float64("secs", 0.4, "seconds per measurement")
	nOne      = flag.Int("n", 1024, "stat: size")
	iA        = flag.Int("a", 200000, "stat: low iteration count")
	iB        = flag.Int("b", 1000000, "stat: high iteration count")
	events    = flag.String("events", "", "stat: comma-separated extra events")
	only      = flag.String("only", "", "stat: go or fftw (default both)")
	re        = flag.String("re", "Benchmark(Complex|Real|CReal)_GoFFT", "envcmp: benchmark regexp")
	benchB    = flag.String("benchb", "", "envcmp: test binary for configuration B (default -bench)")
	envb      = flag.String("envb", "", "envcmp: comma-separated VAR=val for configuration B")
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: pmu parity|stat [flags]")
		os.Exit(2)
	}
	cmd := os.Args[1]
	flag.CommandLine.Parse(os.Args[2:])
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		fatal(err)
	}
	l, _ := strconv.ParseFloat(strings.Fields(string(b))[0], 64)
	if l > *loadMax {
		fmt.Fprintf(os.Stderr, "load %.2f above gate %.2f: deferring\n", l, *loadMax)
		os.Exit(3)
	}
	switch cmd {
	case "parity":
		parity()
	case "stat":
		stat()
	case "envcmp":
		envcmp()
	default:
		fatal(fmt.Errorf("unknown subcommand %q", cmd))
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func pinned(env []string, args ...string) *exec.Cmd {
	c := exec.Command("taskset", append([]string{"-c", *cpu}, args...)...)
	c.Env = append(os.Environ(), env...)
	return c
}

func goArgs(n, iters int) []string {
	return []string{*benchBin, "-test.run=^$",
		"-test.bench=^" + *benchName + "$/^" + strconv.Itoa(n) + "$",
		"-test.benchtime=" + strconv.Itoa(iters) + "x", "-test.count=1"}
}

func goEnv() []string {
	e := []string{"GOMAXPROCS=1"}
	if *fma {
		e = append(e, "GOFFT_FMA=1")
	}
	return e
}

// runGo returns ns/op from the benchmark line.
func runGo(n int, secsWanted float64, est *float64) float64 {
	iters := 1000
	if *est > 0 {
		iters = int(secsWanted * 1e9 / *est)
	}
	var out bytes.Buffer
	c := pinned(goEnv(), goArgs(n, iters)...)
	c.Stdout = &out
	if err := c.Run(); err != nil {
		fatal(err)
	}
	for _, ln := range strings.Split(out.String(), "\n") {
		f := strings.Fields(ln)
		if len(f) >= 4 && strings.HasPrefix(f[0], *benchName+"/") && f[3] == "ns/op" {
			v, _ := strconv.ParseFloat(f[2], 64)
			*est = v
			return v
		}
	}
	fatal(fmt.Errorf("no benchmark line:\n%s", out.String()))
	return 0
}

func runFFTW(n int, secsWanted float64, est *float64) float64 {
	iters := 1000
	if *est > 0 {
		iters = int(secsWanted * 1e9 / *est)
	}
	var out bytes.Buffer
	c := pinned(nil, *fftwBin, strconv.Itoa(n), strconv.Itoa(iters))
	c.Stdout = &out
	if err := c.Run(); err != nil {
		fatal(err)
	}
	f := strings.Fields(out.String())
	v, _ := strconv.ParseFloat(f[1], 64)
	*est = v
	return v
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func parity() {
	var ns []int
	for _, f := range strings.Split(*sizes, ",") {
		n, _ := strconv.Atoi(f)
		ns = append(ns, n)
	}
	gs := map[int][]float64{}
	fs := map[int][]float64{}
	ge := map[int]*float64{}
	fe := map[int]*float64{}
	for _, n := range ns {
		ge[n], fe[n] = new(float64), new(float64)
		runGo(n, *secs, ge[n]) // calibrate
		runFFTW(n, *secs, fe[n])
	}
	for r := 0; r < *rounds; r++ {
		for _, n := range ns {
			gs[n] = append(gs[n], runGo(n, *secs, ge[n]))
			fs[n] = append(fs[n], runFFTW(n, *secs, fe[n]))
		}
	}
	fmt.Printf("%8s %10s %10s %10s %10s %7s %7s\n", "n", "go med", "go min", "fftw med", "fftw min", "r.med", "r.min")
	for _, n := range ns {
		gm, fm := median(gs[n]), median(fs[n])
		gn, fn := minOf(gs[n]), minOf(fs[n])
		fmt.Printf("%8d %10.1f %10.1f %10.1f %10.1f %7.3f %7.3f\n", n, gm, gn, fm, fn, gm/fm, gn/fn)
	}
}

// envcmp runs the benchmarks matching -re under two environments, A (the
// process environment) and B (plus -envb), alternating rounds, and prints per
// benchmark the median ns/op of each and B/A.
func envcmp() {
	run := func(bin string, extra []string) map[string]float64 {
		args := []string{bin, "-test.run=^$", "-test.bench=" + *re,
			"-test.benchtime=" + strconv.FormatFloat(*secs, 'f', -1, 64) + "s", "-test.count=1"}
		var out bytes.Buffer
		c := pinned(append([]string{"GOMAXPROCS=1"}, extra...), args...)
		c.Stdout = &out
		if err := c.Run(); err != nil {
			fatal(err)
		}
		m := map[string]float64{}
		for _, ln := range strings.Split(out.String(), "\n") {
			f := strings.Fields(ln)
			if len(f) >= 4 && strings.HasPrefix(f[0], "Benchmark") && f[3] == "ns/op" {
				v, _ := strconv.ParseFloat(f[2], 64)
				m[f[0]] = v
			}
		}
		return m
	}
	var envB []string
	if *envb != "" {
		envB = strings.Split(*envb, ",")
	}
	as, bs := map[string][]float64{}, map[string][]float64{}
	var names []string
	for r := 0; r < *rounds; r++ {
		for k, v := range run(*benchBin, nil) {
			if r == 0 {
				names = append(names, k)
			}
			as[k] = append(as[k], v)
		}
		binB := *benchBin
		if *benchB != "" {
			binB = *benchB
		}
		for k, v := range run(binB, envB) {
			bs[k] = append(bs[k], v)
		}
	}
	sort.Strings(names)
	fmt.Printf("%-48s %12s %12s %7s\n", "benchmark", "A med", "B med", "B/A")
	for _, k := range names {
		if len(bs[k]) == 0 {
			continue
		}
		a, b := median(as[k]), median(bs[k])
		fmt.Printf("%-48s %12.1f %12.1f %7.3f\n", k, a, b, b/a)
	}
}

func minOf(v []float64) float64 {
	m := v[0]
	for _, x := range v {
		if x < m {
			m = x
		}
	}
	return m
}

// perfCount runs perf stat over the command and returns event -> count.
func perfCount(evs []string, env []string, args ...string) map[string]float64 {
	pa := []string{"stat", "-x,", "-e", strings.Join(evs, ",")}
	pa = append(pa, "taskset", "-c", *cpu)
	pa = append(pa, args...)
	c := exec.Command("perf", pa...)
	c.Env = append(os.Environ(), env...)
	var errb bytes.Buffer
	c.Stderr = &errb
	if err := c.Run(); err != nil {
		fatal(fmt.Errorf("perf: %v\n%s", err, errb.String()))
	}
	m := map[string]float64{}
	for _, ln := range strings.Split(errb.String(), "\n") {
		f := strings.Split(ln, ",")
		if len(f) < 4 {
			continue
		}
		v, err := strconv.ParseFloat(f[0], 64)
		if err != nil {
			m[f[2]] = -1 // <not counted>/<not supported>
			continue
		}
		m[f[2]] = v
	}
	return m
}

func stat() {
	groups := [][]string{
		{"ex_ret_ops", "ls_not_halted_cyc", "de_no_dispatch_per_slot.backend_stalls", "de_no_dispatch_per_slot.no_ops_from_frontend"},
		{"ls_stlf", "ls_bad_status2.stli_other", "ls_misal_loads.ma64", "ls_misal_loads.ma4k"},
		{"ls_dispatch.ld_dispatch", "ls_dispatch.store_dispatch", "ls_dispatch.ld_st_dispatch", "ls_l1_d_tlb_miss.all"},
		{"ls_any_fills_from_sys.all", "ls_any_fills_from_sys.local_l2", "ls_hw_pf_dc_fills.all", "ls_dmnd_fills_from_sys.local_l2"},
		{"de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall", "de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall", "de_dis_dispatch_token_stalls1.load_queue_rsrc_stall", "de_dis_dispatch_token_stalls1.store_queue_rsrc_stall"},
		{"de_dis_dispatch_token_stalls2.retire_token_stall", "de_src_op_disp.op_cache", "de_src_op_disp.decoder", "ic_tag_hit_miss.instruction_cache_miss"},
		{"de_dis_ops_from_decoder.any_fp_dispatch", "ls_int_taken", "ls_l1_d_tlb_miss.all_l2_miss", "ls_sw_pf_dc_fills.all"},
	}
	if *events != "" {
		groups = [][]string{strings.Split(*events, ",")}
	}
	type impl struct {
		name string
		env  []string
		args func(int) []string
	}
	impls := []impl{
		{"go-fft", goEnv(), func(it int) []string { return goArgs(*nOne, it) }},
		{"fftw", nil, func(it int) []string { return []string{*fftwBin, strconv.Itoa(*nOne), strconv.Itoa(it)} }},
	}
	d := float64(*iB - *iA)
	fmt.Printf("# n=%d cpu=%s fma=%v: per-execute counts, (count(%d)-count(%d))/%d, user mode\n", *nOne, *cpu, *fma, *iB, *iA, *iB-*iA)
	for _, im := range impls {
		if *only != "" && !strings.HasPrefix(im.name, *only) {
			continue
		}
		res := map[string]float64{}
		var order []string
		for _, g := range groups {
			evs := []string{"cycles:u", "instructions:u"}
			for _, e := range g {
				evs = append(evs, e+":u")
			}
			lo := perfCount(evs, im.env, im.args(*iA)...)
			hi := perfCount(evs, im.env, im.args(*iB)...)
			for _, e := range evs {
				if _, dup := res[e]; dup {
					continue
				}
				if lo[e] < 0 || hi[e] < 0 {
					res[e] = -1
				} else {
					res[e] = (hi[e] - lo[e]) / d
				}
				order = append(order, e)
			}
		}
		fmt.Printf("## %s\n", im.name)
		for _, e := range order {
			fmt.Printf("%-62s %12.2f\n", e, res[e])
		}
		if c := res["cycles:u"]; c > 0 {
			fmt.Printf("%-62s %12.3f\n", "IPC", res["instructions:u"]/c)
		}
	}
}
