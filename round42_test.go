package fft

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// r42R32Factors are factorizations with a radix-32 pass: first, last,
// between, beside every radix with a kernel, with ido from 1 to 64 and l1
// from 1 to 32, so that the split kernel runs with one and several groups
// and blocks, both widths' group sizes, and the final kernel with odd and
// even l1.
var r42R32Factors = [][]int{
	{32}, {32, 32}, {32, 8}, {8, 32}, {32, 4}, {4, 32}, {32, 2}, {2, 32}, {32, 16}, {16, 32},
	{4, 32, 32}, {32, 4, 32}, {32, 32, 4}, {8, 8, 32}, {2, 32, 32}, {32, 32, 2}, {4, 32, 8},
	{3, 32}, {32, 3}, {5, 32}, {32, 5}, {3, 32, 3}, {32, 9}, {9, 32}, {32, 12}, {32, 32, 3},
	{8, 4, 32}, {32, 8, 4}, {4, 8, 32}, {16, 32, 4},
}

// TestR42R32AgainstNaive holds the radix-32 FMA plans and bfly32FMA, their
// Go reference, to the O(n²) DFT: on the kernels where they run, on the Go
// passes elsewhere (to 1,024 points there, which reaches every branch).
func TestR42R32AgainstNaive(t *testing.T) {
	max := 4096
	if !kernels.UseFMA {
		max = 1024
	}
	for _, f := range r42R32Factors {
		n := factorProduct(f)
		if n > max {
			continue
		}
		p := newSKPlanFactorsOpt(n, f, true)
		x := cmplxSignal(n)
		want := r42DFT(x)
		got := make([]complex128, n)
		p.transform(got, x, false)
		var num, den float64
		for i := range got {
			d := cmplx.Abs(got[i] - want[i])
			num += d * d
			den += cmplx.Abs(want[i]) * cmplx.Abs(want[i])
		}
		if num > 1e-26*den {
			t.Errorf("factors %v: relative error² %.3g", f, num/den)
		}
	}
}

// r42DFT is the O(n²) forward DFT with its angles reduced exactly (k·j mod
// n), which naiveDFT's are not.
func r42DFT(x []complex128) []complex128 {
	n := len(x)
	w := make([]complex128, n)
	for k := range w {
		w[k] = cmplx.Rect(1, -2*math.Pi*float64(k)/float64(n))
	}
	out := make([]complex128, n)
	for k := range out {
		var sum complex128
		for j := range x {
			sum += x[j] * w[k*j%n]
		}
		out[k] = sum
	}
	return out
}

// TestR42DefaultRefusesRadix32: radix-32 passes have FMA kernels only, so
// a default plan is never built with one.
func TestR42DefaultRefusesRadix32(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a default plan was built with a radix-32 pass")
		}
	}()
	newSKPlanFactorsOpt(1024, []int{32, 32}, false)
}

// TestR42FMATable: an FMA plan takes fmaTable's factorization and fuses the
// passes its mask sets, radix 32 always; a plan of other factors (a strip's,
// a row plan's) fuses every pass; a length absent from the table takes
// skFactorize's factors, every pass fused.
func TestR42FMATable(t *testing.T) {
	defer func(m map[int]fmaChoice) { fmaTable = m }(fmaTable)
	fmaTable = map[int]fmaChoice{
		1024: {factors: []int{32, 4, 8}, fuse: 0b100},
		4096: {fuse: 0},
	}
	marks := func(p *skPlan) []bool {
		var m []bool
		for _, st := range p.stages {
			m = append(m, st.fma)
		}
		return m
	}
	p := newSKPlanOpt(1024, true)
	if got := r42Factors(p); !slices.Equal(got, []int{32, 4, 8}) {
		t.Fatalf("1024: factors %v", got)
	}
	if got := marks(p); !slices.Equal(got, []bool{true, false, true}) {
		t.Errorf("1024: fused %v, want the radix-32 pass and the last", got)
	}
	if got := marks(newSKPlanFactorsOpt(1024, []int{8, 8, 16}, true)); !slices.Equal(got, []bool{true, true, true}) {
		t.Errorf("1024 as 8·8·16: fused %v, want every pass", got)
	}
	q := newSKPlanOpt(4096, true)
	if !slices.Equal(r42Factors(q), skFactorize(4096)) || slices.Contains(marks(q), true) {
		t.Errorf("4096: factors %v fused %v, want skFactorize's, none fused", r42Factors(q), marks(q))
	}
	r := newSKPlanOpt(2048, true)
	if !slices.Equal(r42Factors(r), skFactorize(2048)) || slices.Contains(marks(r), false) {
		t.Errorf("2048: factors %v fused %v, want skFactorize's, every pass fused", r42Factors(r), marks(r))
	}
	if d := newSKPlanOpt(1024, false); !slices.Equal(r42Factors(d), skFactorize(1024)) || slices.Contains(marks(d), true) {
		t.Error("a default plan of 1024 points took the FMA table")
	}
}

func r42Factors(p *skPlan) []int {
	var f []int
	for _, st := range p.stages {
		f = append(f, st.r)
	}
	return f
}
