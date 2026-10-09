// znan FILE [REF]: per-row, per-variant medians over rounds of an interleaved
// file (ROW lines of TestZnPlace, ZN lines, or Benchmark lines), and each
// variant's time ÷ REF's time per round (median, min-max). With -place, the
// ROW line's min (best placement) and max are summarised too.
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strconv"
)

func med(x []float64) float64 {
	y := slices.Clone(x)
	slices.Sort(y)
	if len(y) == 0 {
		return 0
	}
	if len(y)%2 == 1 {
		return y[len(y)/2]
	}
	return (y[len(y)/2-1] + y[len(y)/2]) / 2
}

func main() {
	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	ref := ""
	if len(os.Args) > 2 {
		ref = os.Args[2]
	}
	reRound := regexp.MustCompile(`^# round (\d+) variant (\S+)`)
	reRow := regexp.MustCompile(`^ROW (\S+)\s+median\s+([\d.]+)\s+min\s+([\d.]+)\s+max\s+([\d.]+)`)
	reBench := regexp.MustCompile(`^Benchmark\w+/(\S+?)(?:-\d+)?\s+\d+\s+([\d.]+) ns/op`)
	reZN := regexp.MustCompile(`^ZN (\S+)\s+([\d.]+)`)
	type key struct{ row, v, kind string }
	t := map[key]map[int]float64{}
	var rows, vars []string
	rnd, v := 0, ""
	add := func(row, kind string, x float64) {
		k := key{row, v, kind}
		if t[k] == nil {
			t[k] = map[int]float64{}
		}
		t[k][rnd] = x
		if !slices.Contains(rows, row) {
			rows = append(rows, row)
		}
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := sc.Text()
		if m := reRound.FindStringSubmatch(l); m != nil {
			rnd, _ = strconv.Atoi(m[1])
			v = m[2]
			if !slices.Contains(vars, v) {
				vars = append(vars, v)
			}
			continue
		}
		if m := reRow.FindStringSubmatch(l); m != nil {
			a, _ := strconv.ParseFloat(m[2], 64)
			b, _ := strconv.ParseFloat(m[3], 64)
			c, _ := strconv.ParseFloat(m[4], 64)
			add(m[1], "med", a)
			add(m[1], "min", b)
			add(m[1], "max", c)
			continue
		}
		if m := reBench.FindStringSubmatch(l); m != nil {
			a, _ := strconv.ParseFloat(m[2], 64)
			add(m[1], "med", a)
			continue
		}
		if m := reZN.FindStringSubmatch(l); m != nil {
			a, _ := strconv.ParseFloat(m[2], 64)
			add(m[1], "med", a)
		}
	}
	if ref == "" {
		ref = vars[0]
	}
	for _, kind := range []string{"med", "min", "max"} {
		any := false
		for _, row := range rows {
			if t[key{row, ref, kind}] != nil {
				any = true
			}
		}
		if !any {
			continue
		}
		fmt.Printf("== %s (per process); ratio = variant time / %s time, per round\n", kind, ref)
		for _, row := range rows {
			r := t[key{row, ref, kind}]
			if r == nil {
				continue
			}
			fmt.Printf("%-8s", row)
			for _, vv := range vars {
				x := t[key{row, vv, kind}]
				var vals, rs []float64
				for k, val := range x {
					vals = append(vals, val)
					if b, ok := r[k]; ok {
						rs = append(rs, val/b)
					}
				}
				if vv == ref {
					fmt.Printf("  %s %7.1f", vv, med(vals))
				} else if len(rs) > 0 {
					fmt.Printf("  %s %7.1f %.3f (%.3f-%.3f)", vv, med(vals), med(rs), slices.Min(rs), slices.Max(rs))
				}
			}
			fmt.Println()
		}
	}
}
