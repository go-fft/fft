package fft

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
	"testing"
)

var gapSpec = flag.String("r43.gaps", "", `"n1,n2:gap1,gap2,..." whole-transform times per scratch gap in bytes (first gap is the reference), with -r30.ab=rounds,ms`)

// TestR43Gap times whole complex transforms with the scratch gap set to each
// listed value, rotated.
func TestR43Gap(t *testing.T) {
	rounds, ms := smallRounds(t)
	if *gapSpec == "" {
		t.Skip("set -r43.gaps")
	}
	defer func(g uintptr) { setGap = g }(setGap)
	ns, gs, _ := strings.Cut(*gapSpec, ":")
	for _, nstr := range strings.Split(ns, ",") {
		n, err := strconv.Atoi(nstr)
		if err != nil {
			t.Fatal(err)
		}
		var gaps []int
		var fns []func()
		for _, g := range strings.Split(gs, ",") {
			v, err := strconv.Atoi(g)
			if err != nil {
				t.Fatal(err)
			}
			gaps = append(gaps, v)
			p := NewPlan(n)
			src := benchComplex(n)
			dst := make([]complex128, n)
			fns = append(fns, func() { setGap = uintptr(v); p.FFT(dst, src) })
		}
		times := smallRotate(fns, rounds, ms)
		for i, g := range gaps {
			fmt.Printf("GAP %d %d %.1f [ref/this %s]\n", n, g, smallMedian(times[i]), smallRatio(times[0], times[i]))
		}
	}
}
