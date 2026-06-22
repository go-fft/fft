package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// The butterfly stage functions are the FFT hot loop, ROUTED ONTO the transform
// path on every arch (kernels.Radix2Stage/Radix4Stage), so their agreement with
// the scalar oracle is a correctness requirement. These tests assert it on
// structured and random inputs across a sweep of (n, span) shapes — including a
// single group and many small groups, so both the inner k loop and the outer
// base loop are exercised — the same shape as the cmul SIMD-vs-scalar test.
//
// Both the dispatched Radix*Stage and the raw radix*StageSIMD are checked. On
// amd64 the routed kernel is hand-written SSE2 assembly and must match the oracle
// BIT-FOR-BIT (its MULPD/ADDPD are separately rounded, like the non-fused
// GOAMD64=v1 oracle). Off amd64 the routed path is an inlinable Go loop the gc
// compiler autovectorizes in the caller, which may FMA-fuse a product with a
// following add and so differ from the noinline oracle by ≤1 ULP — a correct
// rounding difference, checked within a tight tolerance (see
// kernelMatchesOracle and butterfly_bitexact_*).

func randComplexSlice(rng *rand.Rand, n int) []complex128 {
	s := make([]complex128, n)
	for i := range s {
		s[i] = complex(rng.NormFloat64()*1e3, rng.NormFloat64()*1e-2)
	}
	return s
}

func bitEqualSlice(t *testing.T, who string, got, want []complex128) {
	t.Helper()
	for i := range want {
		if math.Float64bits(real(got[i])) != math.Float64bits(real(want[i])) ||
			math.Float64bits(imag(got[i])) != math.Float64bits(imag(want[i])) {
			t.Fatalf("%s index %d: got %v want %v", who, i, got[i], want[i])
		}
	}
}

// kernelMatchesOracle asserts the dispatched/SIMD result matches the scalar
// oracle: bit-for-bit on amd64 (the hand-written SSE2 kernel's no-FMA contract),
// and within a tight relative tolerance elsewhere (the autovectorized loop may
// FMA-fuse a product with a following add, ≤1 ULP — see butterfly_bitexact_*).
func kernelMatchesOracle(t *testing.T, who string, got, want []complex128) {
	t.Helper()
	if butterflyBitExact {
		bitEqualSlice(t, who, got, want)
		return
	}
	const rel = 1e-12
	for i := range want {
		gr, gi := real(got[i]), imag(got[i])
		wr, wi := real(want[i]), imag(want[i])
		if math.Abs(gr-wr) > rel*(1+math.Abs(wr)) || math.Abs(gi-wi) > rel*(1+math.Abs(wi)) {
			t.Fatalf("%s index %d: got %v want %v (beyond rel tol %g)", who, i, got[i], want[i], rel)
		}
	}
}

// TestStageEmptyGuards covers the early-return guards in the Radix2Stage /
// Radix4Stage dispatchers: a zero span or zero length is a no-op (and must not
// touch a[0], which may be a nil/empty slice).
func TestStageEmptyGuards(t *testing.T) {
	var empty []complex128
	tw := []complex128{complex(1, 0)}
	// span == 0 and n == 0 on both dispatchers, with empty backing slices.
	Radix2Stage(empty, 0, 1, tw)
	Radix2Stage(empty, 4, 0, empty)
	Radix4Stage(empty, 0, 1, tw, tw, tw, false)
	Radix4Stage(empty, 4, 0, empty, empty, empty, true)
	// A length-1 buffer with span 0 (a real call shape: the leaf stage before any
	// combination) must also be a no-op and leave the datum unchanged.
	a := []complex128{complex(3, -4)}
	Radix2Stage(a, 1, 0, empty)
	Radix4Stage(a, 1, 0, empty, empty, empty, false)
	if a[0] != complex(3, -4) {
		t.Fatalf("empty-span stage mutated data: %v", a[0])
	}
}

// referenceRadix2 is an independent scalar oracle for a whole radix-2 stage.
func referenceRadix2(a []complex128, n, span int, tw []complex128) []complex128 {
	out := append([]complex128(nil), a...)
	step := 2 * span
	for base := 0; base < n; base += step {
		for k := 0; k < span; k++ {
			i := base + k
			j := i + span
			hr, hi := real(out[j]), imag(out[j])
			wr, wi := real(tw[k]), imag(tw[k])
			t := complex(hr*wr-hi*wi, hr*wi+hi*wr)
			u := out[i]
			out[i] = u + t
			out[j] = u - t
		}
	}
	return out
}

// stageShapes are (n, span) pairs with n a multiple of radix*span, covering a
// single group, many tiny groups, and an odd-ish span.
var radix2Shapes = [][2]int{
	{2, 1}, {4, 1}, {4, 2}, {8, 1}, {8, 2}, {8, 4},
	{16, 4}, {32, 8}, {64, 16}, {256, 64}, {1024, 1}, {2048, 512},
}

func TestRadix2StageMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2B17))
	for _, sh := range radix2Shapes {
		n, span := sh[0], sh[1]
		a := randComplexSlice(rng, n)
		tw := randComplexSlice(rng, span)
		want := referenceRadix2(a, n, span, tw)

		s := append([]complex128(nil), a...)
		Radix2StageScalar(s, n, span, tw)
		bitEqualSlice(t, "scalar", s, want)

		d := append([]complex128(nil), a...)
		Radix2Stage(d, n, span, tw)
		kernelMatchesOracle(t, "dispatch", d, s)

		skipIfNoSIMD(t)
		k := append([]complex128(nil), a...)
		radix2StageSIMD(k, n, span, tw)
		kernelMatchesOracle(t, "simd", k, s)
	}
}

// referenceRadix4 is an independent scalar oracle for a whole radix-4 stage. It
// uses the package's pinned cmul1 for the products so every term is a separate
// rounding boundary — matching both the noinline scalar oracle and the SIMD
// kernel (which compute each product, then each add, as separate roundings).
func referenceRadix4(a []complex128, n, span int, w1, w2, w3 []complex128, inverse bool) []complex128 {
	out := append([]complex128(nil), a...)
	step := 4 * span
	for base := 0; base < n; base += step {
		i0, i1, i2, i3 := base, base+span, base+2*span, base+3*span
		for k := 0; k < span; k++ {
			m1 := cmul1(out[i1+k], w1[k])
			m2 := cmul1(out[i2+k], w2[k])
			m3 := cmul1(out[i3+k], w3[k])
			a0 := out[i0+k]
			t0 := a0 + m2
			t1 := a0 - m2
			t2 := m1 + m3
			d := m1 - m3
			var t3 complex128
			if inverse {
				t3 = complex(-imag(d), real(d))
			} else {
				t3 = complex(imag(d), -real(d))
			}
			out[i0+k] = t0 + t2
			out[i1+k] = t1 + t3
			out[i2+k] = t0 - t2
			out[i3+k] = t1 - t3
		}
	}
	return out
}

var radix4Shapes = [][2]int{
	{4, 1}, {8, 1}, {16, 1}, {16, 4}, {64, 1}, {64, 4}, {64, 16},
	{256, 64}, {1024, 256}, {4096, 1}, {4096, 1024},
}

func TestRadix4StageMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4B17))
	for _, inverse := range []bool{false, true} {
		for _, sh := range radix4Shapes {
			n, span := sh[0], sh[1]
			a := randComplexSlice(rng, n)
			w1 := randComplexSlice(rng, span)
			w2 := randComplexSlice(rng, span)
			w3 := randComplexSlice(rng, span)
			want := referenceRadix4(a, n, span, w1, w2, w3, inverse)

			s := append([]complex128(nil), a...)
			Radix4StageScalar(s, n, span, w1, w2, w3, inverse)
			bitEqualSlice(t, "scalar", s, want)

			d := append([]complex128(nil), a...)
			Radix4Stage(d, n, span, w1, w2, w3, inverse)
			kernelMatchesOracle(t, "dispatch", d, s)

			skipIfNoSIMD(t)
			k := append([]complex128(nil), a...)
			radix4StageSIMD(k, n, span, w1, w2, w3, inverse)
			kernelMatchesOracle(t, "simd", k, s)
		}
	}
}
