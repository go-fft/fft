//go:build amd64

package fft

// Round 44: a two-pass 32·32 plan for complex 1024 on AMD. TestZ44Frame
// times the FMA 32·32 plan with its twiddled radix-32 pass keeping the values
// between its stages in the output (512-byte streams) or in its frame,
// against the default 8·8·16 and the FMA 8·8·16 plans, whole and pass by
// pass, in rotated rounds in one process.

import (
	"fmt"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

func TestZ44Frame(t *testing.T) {
	defer func() { kernels.R32FrameForce = 0 }()
	rounds, ms := z43Rounds(t)
	n := 1024
	src := benchComplex(n)
	type variant struct {
		name  string
		p     *skPlan
		force int
	}
	vs := []variant{
		{"8x8x16", newSKPlanFactorsOpt(n, []int{8, 8, 16}, false), 0},
		{"8x8x16/fma", newSKPlanFactorsOpt(n, []int{8, 8, 16}, true), 0},
		{"32x32/fma/out", newSKPlanFactorsOpt(n, []int{32, 32}, true), 2},
		{"32x32/fma/frame", newSKPlanFactorsOpt(n, []int{32, 32}, true), 1},
	}
	// The frame variant must give the same bits as the output variant.
	ref, got := make([]complex128, n), make([]complex128, n)
	kernels.R32FrameForce = 2
	vs[2].p.transform(ref, src, false)
	kernels.R32FrameForce = 1
	vs[3].p.transform(got, src, false)
	kernels.R32FrameForce = 0
	if !slices.Equal(ref, got) {
		t.Fatal("frame and output radix-32 kernels differ")
	}
	var fns []func()
	var names []string
	for _, v := range vs {
		p, force := v.p, v.force
		dst := make([]complex128, n)
		fns = append(fns, func() { kernels.R32FrameForce = force; p.transform(dst, src, false) })
		names = append(names, v.name+" whole")
		bp := p.scratch.Get().(*[]complex128)
		scr := (*bp)[:n]
		if takesGap(n) {
			scr = offTheSets(*bp, dst, n)
		}
		s := len(p.stages)
		in := src
		for k := range p.stages {
			out := scr
			if (s-1-k)%2 == 0 {
				out = dst
			}
			st, i0, o0 := &p.stages[k], in, out
			fns = append(fns, func() { kernels.R32FrameForce = force; st.pass(o0, i0, false) })
			names = append(names, fmt.Sprintf("%s p%d(r%d,s%d)", v.name, k, st.r, st.split))
			in = out
		}
	}
	times := comp2Rotate(fns, rounds, ms)
	for q := range fns {
		ratios := make([]float64, rounds)
		for i := range ratios {
			ratios[i] = times[0][i] / times[q][i]
		}
		fmt.Printf("Z44FRAME %-28s %7.1f ns  8x8x16whole/this %.3f\n", names[q], median(times[q]), median(ratios))
	}
}
