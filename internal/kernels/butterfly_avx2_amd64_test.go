package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Compare both dispatch choices against the existing SSE2 implementation,
// including odd spans and short leaf stages that must remain on SSE2.
func TestAMD64ButterflyPaths(t *testing.T) {
	original := useAVX2
	t.Cleanup(func() { useAVX2 = original })
	rng := rand.New(rand.NewSource(42))
	for _, enabled := range []bool{false, true} {
		if enabled && !supportsAVX2() {
			continue
		}
		useAVX2 = enabled
		for _, span := range []int{1, 2, 3, 4, 6, 8, 32, 128, 256} {
			for _, groups := range []int{1, 2, 7, 64} {
				for _, radix := range []int{2, 4} {
					for _, inverse := range []bool{false, true} {
						name := fmt.Sprintf("avx2=%v/radix=%d/span=%d/groups=%d/inverse=%v", enabled, radix, span, groups, inverse)
						t.Run(name, func(t *testing.T) {
							n := radix * span * groups
							for _, special := range []bool{false, true} {
								a := randComplexSlice(rng, n+2)
								w1, w2, w3 := randComplexSlice(rng, span), randComplexSlice(rng, span), randComplexSlice(rng, span)
								if special {
									values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.Inf(-1), math.Float64frombits(0x7ff8000000001234)}
									for i := 1; i <= n; i++ {
										a[i] = complex(values[i%len(values)], values[(i+3)%len(values)])
									}
								}
								got, want := slices.Clone(a), slices.Clone(a)
								if radix == 2 {
									radix2StageSSE2(&want[1], n, span, &w1[0])
									Radix2Stage(got[1:n+1], n, span, w1)
								} else {
									if inverse {
										radix4StageSSE2Inv(&want[1], n, span, &w1[0], &w2[0], &w3[0])
									} else {
										radix4StageSSE2Fwd(&want[1], n, span, &w1[0], &w2[0], &w3[0])
									}
									Radix4Stage(got[1:n+1], n, span, w1, w2, w3, inverse)
								}
								bitEqualSlice(t, "SSE2 parity with canaries", got, want)
							}
						})
					}
				}
			}
		}
	}
}
