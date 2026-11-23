// Command ppdrv attributes the time of a Stockham plan to its passes and
// compares it with FFTW, on one pinned core: ns/op from interleaved rounds,
// hardware counters from `perf stat` of two iteration counts subtracted
// (planning cancels). Measurement only (branch exp/passprofile).
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type item struct {
	name  string
	argv  func(iters int) []string
	env   []string
	iters int
	ns    []float64
	ctr   map[string]float64
}

var (
	core   = flag.String("core", "5", "cpu to pin")
	gotest = flag.String("gotest", "", "path of the fft test binary")
	fftwA  = flag.String("fftw512", "", "fftw_pp linked with the AVX-512 build")
	fftwB  = flag.String("fftw2", "", "fftw_pp linked with the AVX2 build")
	wisdir = flag.String("wis", os.Getenv("HOME")+"/gofft-bench/", "wisdom directory")
	rounds = flag.Int("rounds", 7, "timing rounds")
	lens   = flag.String("n", "1000,1080,1440,1920,2160", "lengths")
)

var groups = [][]string{
	{"cycles", "instructions", "L1-dcache-load-misses", "ls_stlf", "ex_ret_brn_misp"},
	{"cycles", "de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall", "de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall",
		"de_dis_dispatch_token_stalls1.load_queue_rsrc_stall", "de_dis_dispatch_token_stalls1.store_queue_rsrc_stall",
		"de_dis_dispatch_token_stalls2.retire_token_stall"},
	{"cycles", "ex_ret_ops", "fp_ret_sse_avx_ops.all", "ls_any_fills_from_sys.local_l2", "ls_any_fills_from_sys.local_ccx"},
}

var resRe = regexp.MustCompile(`ns/op=([0-9.]+)`)

func run(it *item, iters int, perf []string) (string, error) {
	argv := append(append([]string{}, perf...), "taskset", "-c", *core)
	argv = append(argv, it.argv(iters)...)
	c := exec.Command(argv[0], argv[1:]...)
	c.Env = append(os.Environ(), it.env...)
	c.Env = append(c.Env, "PP_ITERS="+strconv.Itoa(iters))
	out, err := c.CombinedOutput()
	return string(out), err
}

func nsOf(out string) float64 {
	m := resRe.FindStringSubmatch(out)
	if m == nil {
		panic("no ns/op in: " + out)
	}
	v, _ := strconv.ParseFloat(m[1], 64)
	return v
}

func counters(it *item, ev []string, iters int) map[string]float64 {
	perf := []string{"perf", "stat", "-x", ",", "-e", strings.Join(ev, ",")}
	parse := func(it2 int) map[string]float64 {
		out, err := run(it, it2, perf)
		if err != nil {
			panic(out)
		}
		m := map[string]float64{}
		for _, l := range strings.Split(out, "\n") {
			f := strings.Split(l, ",")
			if len(f) > 2 {
				if v, e := strconv.ParseFloat(f[0], 64); e == nil {
					m[strings.TrimSuffix(f[2], ":u")] = v
				}
			}
		}
		return m
	}
	lo, hi := parse(iters), parse(3*iters)
	d := map[string]float64{}
	for k := range hi {
		d[k] = (hi[k] - lo[k]) / float64(2*iters)
	}
	return d
}

func median(v []float64) float64 {
	s := append([]float64{}, v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func main() {
	flag.Parse()
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		f := strings.Fields(string(b))
		l, _ := strconv.ParseFloat(f[0], 64)
		fmt.Println("loadavg", strings.TrimSpace(string(b)))
		if l > 8 {
			fmt.Println("DEFER: 1-min load above 8")
			os.Exit(3)
		}
	}
	for _, ns := range strings.Split(*lens, ",") {
		n, _ := strconv.Atoi(ns)
		// pass count and the pass list
		lc := exec.Command(*gotest, "-test.run", "^TestPPList$", "-test.count=1")
		lc.Env = append(os.Environ(), "PP_LIST=1")
		out, err := lc.CombinedOutput()
		if err != nil {
			panic(string(out))
		}
		stages := 0
		for _, l := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(l, fmt.Sprintf("n=%d ", n)) {
				fmt.Sscanf(l[strings.Index(l, "stages="):], "stages=%d", &stages)
			}
		}
		var items []*item
		gen := func(name string, pass int) *item {
			return &item{name: name, env: []string{"PP_N=" + ns, "PP_PASS=" + strconv.Itoa(pass), "PP_LIST="},
				argv: func(int) []string { return []string{*gotest, "-test.run", "^TestPPLoop$", "-test.count=1"} }}
		}
		items = append(items, gen("go.plan", -1))
		for k := 0; k < stages; k++ {
			items = append(items, gen(fmt.Sprintf("go.pass%d", k), k))
		}
		for _, lib := range []struct{ name, bin string }{{"fftw512", *fftwA}, {"fftw2", *fftwB}} {
			if lib.bin == "" {
				continue
			}
			wf := *wisdir + "pp-wis-" + lib.name + "-" + ns
			c := exec.Command("taskset", "-c", *core, lib.bin, ns, "1000", wf, "m")
			c.Env = append(os.Environ(), "FFTW_PRINT_PLAN=1")
			po, err := c.CombinedOutput()
			if err != nil {
				panic(string(po))
			}
			fmt.Printf("n=%d %s plan:\n%s\n", n, lib.name, strings.TrimSpace(strings.ReplaceAll(string(po), "PPRESULT", "(measure-run)")))
			bin := lib.bin
			items = append(items, &item{name: lib.name, argv: func(it int) []string {
				return []string{bin, ns, strconv.Itoa(it), wf, "w"}
			}})
		}
		// calibrate
		for _, it := range items {
			o, err := run(it, 5000, nil)
			if err != nil {
				panic(o)
			}
			it.iters = int(80e6 / nsOf(o))
			if it.iters < 2000 {
				it.iters = 2000
			}
		}
		for r := 0; r < *rounds; r++ {
			for _, it := range items {
				o, err := run(it, it.iters, nil)
				if err != nil {
					panic(o)
				}
				it.ns = append(it.ns, nsOf(o))
			}
		}
		for _, it := range items {
			it.ctr = map[string]float64{}
			for _, g := range groups {
				for k, v := range counters(it, g, it.iters) {
					it.ctr[k] = v
				}
			}
		}
		var sum float64
		var tot float64
		for _, it := range items {
			if it.name == "go.plan" {
				tot = median(it.ns)
			}
			if strings.HasPrefix(it.name, "go.pass") {
				sum += median(it.ns)
			}
		}
		fmt.Printf("\n=== n=%d  sum-of-passes %.1f ns, whole plan %.1f ns\n", n, sum, tot)
		keys := []string{"cycles", "instructions", "ex_ret_ops", "L1-dcache-load-misses", "ls_any_fills_from_sys.local_ccx", "ls_any_fills_from_sys.local_l2",
			"ls_stlf", "ex_ret_brn_misp", "fp_ret_sse_avx_ops.all",
			"de_dis_dispatch_token_stalls1.fp_sch_rsrc_stall", "de_dis_dispatch_token_stalls1.fp_reg_file_rsrc_stall",
			"de_dis_dispatch_token_stalls1.load_queue_rsrc_stall", "de_dis_dispatch_token_stalls1.store_queue_rsrc_stall",
			"de_dis_dispatch_token_stalls2.retire_token_stall"}
		fmt.Printf("%-10s %9s %6s", "item", "ns(med)", "share")
		short := []string{"cyc", "instr", "ops", "L1miss", "L1fill", "fillL2", "stlf", "brmisp", "fpops", "fpSch", "fpReg", "ldQ", "stQ", "retTok"}
		fmt.Printf(" %6s", "IPC")
		for _, s := range short {
			fmt.Printf(" %8s", s)
		}
		fmt.Println()
		for _, it := range items {
			m := median(it.ns)
			fmt.Printf("%-10s %9.1f %5.0f%%", it.name, m, 100*m/tot)
			fmt.Printf(" %6.2f", it.ctr["instructions"]/it.ctr["cycles"])
			for _, k := range keys {
				fmt.Printf(" %8.0f", it.ctr[k])
			}
			fmt.Println()
		}
	}
}
