// Command asmcount counts the instructions of a Go assembly function between
// two labels, by mnemonic class: the steady loop of a kernel.
//
//	asmcount -f file.s -func name -from gloop -to knext
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
)

// class buckets a mnemonic.
func class(m string, line string) string {
	mem := strings.Contains(line, "(")
	switch {
	case strings.HasPrefix(m, "VMOVUPD") || strings.HasPrefix(m, "VMOVAPD") || strings.HasPrefix(m, "VMOVDDUP"):
		if !mem {
			return "move"
		}
		// store if the memory operand is last
		ops := line[strings.Index(line, m)+len(m):]
		if strings.LastIndex(ops, "(") > strings.LastIndex(ops, ",") {
			return "store"
		}
		return "load"
	case strings.HasPrefix(m, "VINSERTF128"):
		return "insert"
	case strings.HasPrefix(m, "VADD") || strings.HasPrefix(m, "VSUB"):
		return "add"
	case strings.HasPrefix(m, "VMUL"):
		return "mul"
	case strings.HasPrefix(m, "VFM") || strings.HasPrefix(m, "VFNM"):
		return "fma"
	case strings.HasPrefix(m, "VPERM") || strings.HasPrefix(m, "VUNPCK") || strings.HasPrefix(m, "VSHUF") || strings.HasPrefix(m, "VBLEND"):
		return "shuffle"
	case strings.HasPrefix(m, "VXOR"):
		return "xor"
	case strings.HasPrefix(m, "V"):
		return "otherv"
	}
	return "scalar"
}

func main() {
	f := flag.String("f", "", "assembly file")
	fn := flag.String("func", "", "TEXT name")
	from := flag.String("from", "gloop", "first label")
	to := flag.String("to", "knext", "end label")
	flag.Parse()
	b, err := os.ReadFile(*f)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	in, on := false, false
	counts := map[string]int{}
	total := 0
	for _, l := range strings.Split(string(b), "\n") {
		t := strings.TrimSpace(l)
		if strings.HasPrefix(t, "TEXT ") {
			in = strings.Contains(t, "·"+*fn+"(SB)")
			on = false
			continue
		}
		if !in || t == "" {
			continue
		}
		if t == *from+":" {
			on = true
			continue
		}
		if t == *to+":" && on {
			break
		}
		if !on || strings.HasSuffix(t, ":") {
			continue
		}
		m := strings.Fields(t)[0]
		counts[class(m, t)]++
		total++
	}
	if total == 0 {
		fmt.Fprintln(os.Stderr, "no instructions found: wrong function or labels")
		os.Exit(1)
	}
	var keys []string
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("%s %s..%s total %d:", *fn, *from, *to, total)
	for _, k := range keys {
		fmt.Printf(" %s=%d", k, counts[k])
	}
	fmt.Println()
}
