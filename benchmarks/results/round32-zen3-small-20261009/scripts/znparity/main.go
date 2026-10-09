// znparity MAIN1 MAIN2 BR1 BR2: per REPORT.md row, go-fft's time (mean of a
// variant's two runs) ÷ the median of FFTW's time over all four runs, and
// FFTW's own range (Round 30's parity.py, in Go).
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

type key struct{ sec, n string }

var rowRe = regexp.MustCompile(`^\| ([\dx,]+)[^|]*\| ([\d,]+) \([^)]*\) \| ([\d,]+) \(`)

func rows(dir string) (map[key][2]float64, []key) {
	f, err := os.Open(dir + "/REPORT.md")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	out := map[key][2]float64{}
	var order []key
	sec := ""
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		l := sc.Text()
		if strings.HasPrefix(l, "## ") {
			sec = strings.TrimSpace(strings.SplitN(l[3:], "(", 2)[0])
			continue
		}
		if m := rowRe.FindStringSubmatch(l); m != nil {
			g, _ := strconv.ParseFloat(strings.ReplaceAll(m[2], ",", ""), 64)
			w, _ := strconv.ParseFloat(strings.ReplaceAll(m[3], ",", ""), 64)
			k := key{sec, m[1]}
			out[k] = [2]float64{g, w}
			order = append(order, k)
		}
	}
	return out, order
}

func main() {
	m1, order := rows(os.Args[1])
	m2, _ := rows(os.Args[2])
	b1, _ := rows(os.Args[3])
	b2, _ := rows(os.Args[4])
	fmt.Printf("%-34s %10s %10s %17s %6s %6s\n", "section", "N", "FFTW med", "FFTW range", "main", "branch")
	for _, k := range order {
		rs := []map[key][2]float64{m1, m2, b1, b2}
		ok := true
		var f []float64
		for _, r := range rs {
			v, has := r[k]
			ok = ok && has
			f = append(f, v[1])
		}
		if !ok {
			continue
		}
		s := slices.Clone(f)
		slices.Sort(s)
		fm := (s[1] + s[2]) / 2
		gm := (m1[k][0] + m2[k][0]) / 2
		gb := (b1[k][0] + b2[k][0]) / 2
		sec := k.sec
		if len(sec) > 34 {
			sec = sec[:34]
		}
		fmt.Printf("%-34s %10s %10.0f %8.0f-%-8.0f %6.2f %6.2f\n", sec, k.n, fm, s[0], s[3], gm/fm, gb/fm)
	}
}
