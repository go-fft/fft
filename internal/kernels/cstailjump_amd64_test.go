package kernels

import (
	goflag "flag"
	"fmt"
	"math"
	"slices"
	"testing"
	"time"
)

// Round 35 (folding in Round 37's finding): Untangle and Retangle used to
// call the 512-bit kernel and then, for the pairs left over, the AVX2 one;
// with quads = 0 (every AVX2-only CPU, and every m the 512-bit kernel does
// not take) the first call did nothing but cost one. The 512-bit kernel now
// ends in a tail jump to the AVX2 one. csUntangleTwoCalls and
// csRetangleTwoCalls are the previous wrappers, for the A/B and as the bit
// reference.

func csUntangleTwoCalls(dst, z, tw []complex128, m int) int {
	pairs := (m - 1) / 2 / 2
	if !UseUntangleAVX2 || pairs <= 0 {
		return 0
	}
	_, _, _ = dst[m], z[m-1], tw[2*pairs+1]
	q := clQuads(UseUntangleAVX512, pairs, m, UntangleWideMax)
	clUntangleAVX512(&dst[0], &z[0], &tw[0], &skFwd[0][0], m, q, 2*q) // no pair left: returns
	if rest := pairs - 2*q; rest > 0 {
		smallUntangleAVX2(&dst[4*q], &z[4*q], &tw[4*q], &skFwd[0][0], m-8*q, rest)
	}
	return 2 * pairs
}

func csRetangleTwoCalls(z, x, tw []complex128, m int, h float64) int {
	pairs := (m - 1) / 2 / 2
	if !UseUntangleAVX2 || pairs <= 0 {
		return 0
	}
	_, _, _ = z[m-1], x[m], tw[2*pairs+1]
	hv := [4]float64{h, h, h, h}
	q := clQuads(UseUntangleAVX512, pairs, m, RetangleWideMax)
	clRetangleAVX512(&z[0], &x[0], &tw[0], &skFwd[0][0], &hv, m, q, 2*q)
	if rest := pairs - 2*q; rest > 0 {
		smallRetangleAVX2(&z[4*q], &x[4*q], &tw[4*q], &skFwd[0][0], &hv, m-8*q, rest)
	}
	return 2 * pairs
}

func csTJData(m int) (dst, z, tw []complex128) {
	dst, z, tw = make([]complex128, m+1), make([]complex128, m+1), make([]complex128, m+2)
	for i := range z {
		z[i] = complex(math.Sin(float64(3*i+1)), math.Cos(float64(5*i+2)))
	}
	for i := range tw {
		a := -math.Pi * float64(i) / float64(m)
		tw[i] = complex(math.Cos(a), math.Sin(a))
	}
	return
}

// TestCsTailJumpMatchesTwoCalls: one call with the tail jump gives the bits
// the two calls gave, at both widths, for every m to 600 and a few large
// ones, untangle and retangle, in place and not.
func TestCsTailJumpMatchesTwoCalls(t *testing.T) {
	if !UseUntangleAVX2 {
		t.Skip("no AVX2")
	}
	defer func(v bool) { UseUntangleAVX512 = v }(UseUntangleAVX512)
	modes := []bool{false}
	if UseUntangleAVX512 && intelHasAVX512 {
		modes = append(modes, true)
	}
	ms := []int{1 << 14, 1 << 15, 1 << 16, 1<<16 + 4, 4097}
	for m := 2; m <= 600; m++ {
		ms = append(ms, m)
	}
	for _, wide := range modes {
		UseUntangleAVX512 = wide
		for _, m := range ms {
			a, z, tw := csTJData(m)
			b := slices.Clone(a)
			if Untangle(a, z, tw, m) != csUntangleTwoCalls(b, z, tw, m) || !slices.Equal(a, b) {
				t.Fatalf("untangle m=%d wide=%v differs", m, wide)
			}
			zi, zj := slices.Clone(z), slices.Clone(z)
			if Untangle(zi, zi, tw, m); csUntangleTwoCalls(zj, zj, tw, m) >= 0 && !slices.Equal(zi, zj) {
				t.Fatalf("untangle m=%d wide=%v in place differs", m, wide)
			}
			ra, rb := make([]complex128, m+1), make([]complex128, m+1)
			if Retangle(ra, z, tw, m, 0.25) != csRetangleTwoCalls(rb, z, tw, m, 0.25) || !slices.Equal(ra, rb) {
				t.Fatalf("retangle m=%d wide=%v differs", m, wide)
			}
		}
	}
}

var csTJ = goflag.String("cs.tj", "", "rounds: TestCsTailJumpAB")

// TestCsTailJumpAB times Untangle and Retangle (one call) against the two
// calls, rotated, at both widths, each over the same eight buffer sets; ns
// per call.
func TestCsTailJumpAB(t *testing.T) {
	var rounds int
	if _, err := fmt.Sscanf(*csTJ, "%d", &rounds); err != nil {
		t.Skip("set -cs.tj=rounds")
	}
	defer func(v bool) { UseUntangleAVX512 = v }(UseUntangleAVX512)
	for _, wide := range []bool{false, true} {
		UseUntangleAVX512 = wide && intelHasAVX512
		for _, m := range []int{64, 128, 256, 512, 1000, 1024, 32768} {
			// Eight placements: each variant runs on the same eight buffer
			// sets, so neither owns a lucky one.
			var sets [8][3][]complex128
			var pad [][]complex128
			for i := range sets {
				pad = append(pad, make([]complex128, 1+3*i))
				sets[i][0], sets[i][1], sets[i][2] = csTJData(m)
			}
			_ = pad
			all := func(f func(dst, z, tw []complex128)) func() {
				return func() {
					for i := range sets {
						f(sets[i][0], sets[i][1], sets[i][2])
					}
				}
			}
			fns := []func(){
				all(func(dst, z, tw []complex128) { csUntangleTwoCalls(dst, z, tw, m) }),
				all(func(dst, z, tw []complex128) { Untangle(dst, z, tw, m) }),
				all(func(dst, z, tw []complex128) { csRetangleTwoCalls(dst, z, tw, m, 0.5) }),
				all(func(dst, z, tw []complex128) { Retangle(dst, z, tw, m, 0.5) }),
			}
			iters := max(1, 2500000/m)
			times := make([][]float64, len(fns))
			for r := range rounds {
				for q := range fns {
					v := (q + r) % len(fns)
					t0 := time.Now()
					for range iters {
						fns[v]()
					}
					times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(8*iters))
				}
			}
			med := func(x []float64) float64 { x = slices.Clone(x); slices.Sort(x); return x[len(x)/2] }
			ratio := func(a, b []float64) (m, lo, hi float64) {
				q := make([]float64, len(a))
				for i := range a {
					q[i] = a[i] / b[i]
				}
				return med(q), slices.Min(q), slices.Max(q)
			}
			um, ul, uh := ratio(times[0], times[1])
			rm, rl, rh := ratio(times[2], times[3])
			fmt.Printf("TJR wide=%v m=%5d untangle two/one per-round median %.3f [%.3f %.3f] | retangle %.3f [%.3f %.3f]\n", UseUntangleAVX512, m, um, ul, uh, rm, rl, rh)
			fmt.Printf("TJ wide=%v m=%5d untangle two %8.1f one %8.1f two/one %.3f | retangle two %8.1f one %8.1f two/one %.3f\n",
				UseUntangleAVX512, m, med(times[0]), med(times[1]), med(times[0])/med(times[1]), med(times[2]), med(times[3]), med(times[2])/med(times[3]))
		}
	}
}
