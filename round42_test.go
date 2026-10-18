package fft

import (
	"slices"
	"testing"
)

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
