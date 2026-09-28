//go:build amd64 && !amd64.v3

package fft

import (
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The scalar oracle is separately rounded below GOAMD64=v3. At v3 the Go
// compiler may fuse its arithmetic; SSE2/AVX2 parity is tested in kernels on
// every amd64 build level instead.
func TestRealPlanRFFTMatchesScalarStages(t *testing.T) {
	rng := rand.New(rand.NewSource(1024))
	for _, n := range []int{4, 8, 16, 64, 256, 512, 1024, 2048, 4096, 16384} {
		p := NewRealPlan(n)
		for trial := range 5 {
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64()
				if trial == 0 {
					x[i] = math.Copysign(0, x[i])
				}
			}
			original := slices.Clone(x)
			z := make([]complex128, n/2)
			for i, j := range p.half.it.revPos {
				z[i] = complex(x[2*j], x[2*j+1])
			}
			for _, st := range p.half.it.stages {
				if st.radix == 2 {
					kernels.Radix2StageScalar(z, n/2, st.span, st.tw)
				} else {
					kernels.Radix4StageScalar(z, n/2, st.span, st.tw[:st.span], st.tw[st.span:2*st.span], st.tw[2*st.span:], false)
				}
			}
			want, got := make([]complex128, n/2+1), make([]complex128, n/2+1)
			rfftUntangle(want, z, p.tw, n/2)
			p.RFFT(got, x)
			for k := range want {
				if math.Float64bits(real(want[k])) != math.Float64bits(real(got[k])) || math.Float64bits(imag(want[k])) != math.Float64bits(imag(got[k])) {
					t.Fatalf("n=%d trial=%d bin=%d: got %v want %v", n, trial, k, got[k], want[k])
				}
			}
			for i := range x {
				if math.Float64bits(x[i]) != math.Float64bits(original[i]) {
					t.Fatalf("n=%d input changed at %d", n, i)
				}
			}
		}
	}
}
