//go:build amd64 && !amd64.v3

package fft

import (
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestR42R32MatchesReference: every radix-32 FMA plan, under every layout the
// machine runs, gives the bits of the same plan run on the Go passes
// (passScalarFMA with bfly32FMA), in place and not, both directions, on the
// generic, signed-zero and infinite signals.
func TestR42R32MatchesReference(t *testing.T) {
	fmaKernelsHere(t)
	defer func(a bool) { kernels.UseStockhamAVX2 = a }(kernels.UseStockhamAVX2)
	for _, l := range r41Layouts() {
		withLayout(l.avx512, l.split, l.sp512, func() {
			split32, inFrame, wide := 0, 0, 0
			for _, f := range r42R32Factors {
				n := factorProduct(f)
				p := newSKPlanFactorsOpt(n, f, true)
				for _, st := range p.stages {
					if st.r == 32 && st.split != 0 {
						split32++
						if st.l1*st.ido >= 64 {
							inFrame++
						}
						if st.split > 4 { // a 512-bit split mode
							wide++
						}
					}
				}
				for s, x := range simdSignals(n) {
					for _, inverse := range []bool{false, true} {
						kernels.UseStockhamAVX2 = false
						ref := make([]complex128, n)
						p.transform(ref, x, inverse)
						kernels.UseStockhamAVX2 = true
						got := make([]complex128, n)
						p.transform(got, x, inverse)
						alias := slices.Clone(x)
						p.transform(alias, alias, inverse)
						for i := range ref {
							if !sameBits(got[i], ref[i]) || !sameBits(alias[i], ref[i]) {
								t.Fatalf("%s factors %v signal %d inverse=%v index %d: %v (in place %v), reference %v",
									l.name, f, s, inverse, i, got[i], alias[i], ref[i])
							}
						}
					}
				}
			}
			if (l.split || l.sp512) && (inFrame == 0 || inFrame == split32) {
				t.Fatalf("%s: %d split radix-32 passes, %d of them in-frame: both kernels must run", l.name, split32, inFrame)
			}
			if l.sp512 && wide == 0 {
				t.Fatalf("%s: no 512-bit split radix-32 pass was built", l.name)
			}
		})
	}
}

// TestR42FrameMasks: a frame plan (kernels.ZnPlan) of an FMA plan that fuses
// only some passes runs each pass on the kernel its stage says, bit for bit
// what the stages' own passes give one after the other, for every mask.
func TestR42FrameMasks(t *testing.T) {
	fmaKernelsHere(t)
	defer func(m map[int]fmaChoice) { fmaTable = m }(fmaTable)
	framed := 0
	for _, n := range []int{64, 128, 192, 256, 384, 512} {
		s := len(newSKPlanOpt(n, true).stages)
		for mask := range uint32(1) << s {
			fmaTable = map[int]fmaChoice{n: {fuse: mask}}
			p := newSKPlanOpt(n, true)
			if p.zn == nil {
				continue
			}
			framed++
			for _, inverse := range []bool{false, true} {
				x := cmplxSignal(n)
				got := make([]complex128, n)
				p.transform(got, x, inverse)
				in := x
				for k := range p.stages {
					out := make([]complex128, n)
					p.stages[k].pass(out, in, inverse)
					in = out
				}
				for i := range got {
					if !sameBits(got[i], in[i]) {
						t.Fatalf("n=%d mask %b inverse=%v index %d: frame %v, passes %v", n, mask, inverse, i, got[i], in[i])
					}
				}
			}
		}
	}
	if framed == 0 {
		t.Skip("no frame plan on this CPU")
	}
}
