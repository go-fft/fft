//go:build amd64

package fft

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestR30Steps times the untangle and retangle kernels with 1, 2 and 3
// interleaved steps, alone and inside RFFT and IRFFT, all rotated in one
// process.
func TestR30Steps(t *testing.T) {
	rounds, ms := smallRounds(t)
	defer func(v int) { kernels.UntangleSteps = v }(kernels.UntangleSteps)
	for _, s := range strings.Split(*smallReals, ",") {
		n, err := strconv.Atoi(s)
		if err != nil {
			t.Fatal(err)
		}
		p := NewRealPlan(n)
		src := benchReal(n)
		dst := make([]complex128, n/2+1)
		m := n / 2
		Z := make([]complex128, m)
		p.half.execute(Z, asComplex(src[:2*m]), false)
		p.RFFT(dst, src)
		back := make([]float64, n)
		zb := make([]complex128, m)
		steps := []int{1, 3, 22, 23}
		var fns []func()
		for _, st := range steps {
			fns = append(fns,
				func() { kernels.UntangleSteps = st; rfftUntangle(dst, Z, p.tw, m) },
				func() { kernels.UntangleSteps = st; irfftRetangle(zb, dst, p.tw, m, 0.5/float64(m)) },
				func() { kernels.UntangleSteps = st; p.RFFT(dst, src) },
				func() { kernels.UntangleSteps = st; p.IRFFT(back, dst) },
			)
		}
		times := smallRotate(fns, rounds, ms)
		names := []string{"untangle", "retangle", "rfft", "irfft"}
		for i, st := range steps {
			var b strings.Builder
			for j, nm := range names {
				k := 4*i + j
				fmt.Fprintf(&b, " %s %.1f [1/this %s]", nm, smallMedian(times[k]), smallRatio(times[j], times[k]))
			}
			fmt.Printf("STEPS %d s%d%s\n", n, st, b.String())
		}
	}
}

// TestR30StepsMatch runs the untangle and retangle bit-identity tests on
// every step count.
func TestR30StepsMatch(t *testing.T) {
	defer func(v int) { kernels.UntangleSteps = v }(kernels.UntangleSteps)
	for _, st := range []int{1, 2, 3, 22, 23} {
		kernels.UntangleSteps = st
		TestUntangleMatchesScalar(t)
		TestRetangleMatchesScalar(t)
	}
}
