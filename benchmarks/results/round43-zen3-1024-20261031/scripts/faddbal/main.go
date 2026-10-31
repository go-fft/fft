// Command faddbal is a prototype: it rewrites some register-only VADDPD and
// VSUBPD of chosen TEXT functions of a Go assembly file into FMA forms with
// a multiplier of exactly 1.0 read from memory, which round identically
// (a·1 is exact, so fma(a, 1, ±b) is RN(a ± b)) but issue on the FMA pipes
// instead of the FADD pipes. It exists to measure whether moving additions to
// the multiply pipes speeds up an add-bound kernel before the generator
// (go-asmgen) is changed.
package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var addRe = regexp.MustCompile(`^(\s*)(VADDPD|VSUBPD) (Y\d+), (Y\d+), (Y\d+)\s*$`)

// rewrite converts every every-th eligible add (counted per function, from
// phase) in the named functions, using ones as the memory operand.
func rewrite(src string, funcs map[string]bool, every, phase int, ones string) (string, int) {
	lines := strings.Split(src, "\n")
	var out []string
	cur, count, done := "", 0, 0
	for _, l := range lines {
		if strings.HasPrefix(l, "TEXT ") {
			cur = l[len("TEXT "):]
			if i := strings.Index(cur, "(SB)"); i >= 0 {
				cur = strings.TrimPrefix(cur[:i], "·")
			}
			count = 0
		}
		m := addRe.FindStringSubmatch(l)
		if m == nil || !funcs[cur] {
			out = append(out, l)
			continue
		}
		count++
		if (count-1)%every != phase {
			out = append(out, l)
			continue
		}
		ind, op, b, a, d := m[1], m[2], m[3], m[4], m[5]
		// d = a + b or d = a - b.
		switch {
		case op == "VADDPD" && d == a:
			out = append(out, fmt.Sprintf("%sVFMADD132PD %s, %s, %s", ind, ones, b, d))
		case op == "VADDPD" && d == b:
			out = append(out, fmt.Sprintf("%sVFMADD132PD %s, %s, %s", ind, ones, a, d))
		case op == "VADDPD":
			out = append(out, fmt.Sprintf("%sVMOVAPD %s, %s", ind, a, d))
			out = append(out, fmt.Sprintf("%sVFMADD132PD %s, %s, %s", ind, ones, b, d))
		case d == a: // d = d·1 − b
			out = append(out, fmt.Sprintf("%sVFMSUB132PD %s, %s, %s", ind, ones, b, d))
		case d == b: // d = −(d·1) + a
			out = append(out, fmt.Sprintf("%sVFNMADD132PD %s, %s, %s", ind, ones, a, d))
		default:
			out = append(out, fmt.Sprintf("%sVMOVAPD %s, %s", ind, a, d))
			out = append(out, fmt.Sprintf("%sVFMSUB132PD %s, %s, %s", ind, ones, b, d))
		}
		done++
	}
	return strings.Join(out, "\n"), done
}

// extract returns the named functions of src (each from its TEXT line to the
// next TEXT line), renamed with suffix.
func extract(src string, funcs map[string]bool, suffix string) string {
	var out []string
	keep := false
	for _, l := range strings.Split(src, "\n") {
		if strings.HasPrefix(l, "TEXT ") {
			name := l[len("TEXT "):]
			if i := strings.Index(name, "(SB)"); i >= 0 {
				name = strings.TrimPrefix(name[:i], "·")
			}
			keep = funcs[name]
			if keep {
				l = strings.Replace(l, "·"+name+"(SB)", "·"+name+suffix+"(SB)", 1)
			}
		}
		if keep {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

func main() {
	copyTo := flag.String("copy", "", "write the named functions, renamed with -suffix and rewritten, to this file (appending) instead of rewriting -f")
	suffix := flag.String("suffix", "Bal", "suffix for -copy")
	in := flag.String("f", "", "assembly file, rewritten in place")
	fl := flag.String("funcs", "", "comma-separated TEXT names")
	every := flag.Int("every", 3, "convert one add in every N")
	phase := flag.Int("phase", 0, "which of the N")
	ones := flag.String("ones", "·faddOnes(SB)", "memory operand holding four 1.0")
	flag.Parse()
	b, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	funcs := map[string]bool{}
	for _, f := range strings.Split(*fl, ",") {
		funcs[f] = true
	}
	if *copyTo != "" {
		renamed := map[string]bool{}
		for f := range funcs {
			renamed[f+*suffix] = true
		}
		s, n := rewrite(extract(string(b), funcs, *suffix), renamed, *every, *phase, *ones)
		prev, _ := os.ReadFile(*copyTo)
		if err := os.WriteFile(*copyTo, append(prev, []byte("\n"+s+"\n")...), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Printf("%s: %d adds moved\n", *copyTo, n)
		return
	}
	s, n := rewrite(string(b), funcs, *every, *phase, *ones)
	if err := os.WriteFile(*in, []byte(s), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("%s: %d adds moved\n", *in, n)
}
