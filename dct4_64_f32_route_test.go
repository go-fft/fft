package fft

import (
	"math"
	"testing"
)

// The routing of DCTPlan32's pair and quad forms: a plan with quad set (a copy, so
// every platform exercises it; off amd64 the kernel is the portable stand-in of
// dct4_64_f32_other.go) takes the kernel exactly when no destination overlaps
// another destination or another stream's source, and otherwise runs the single
// calls in order, as written.

const routeN = 64

func routeArena(seed uint64) []float32 {
	a := make([]float32, 9*routeN)
	copy(a, accRealF32(len(a), seed))
	return a
}

type routeLayout struct {
	dst, src [4]int // element offsets in the arena
}

func disjointLayout() routeLayout {
	var l routeLayout
	for i := 0; i < 4; i++ {
		l.dst[i] = routeN * i
		l.src[i] = routeN * (4 + i)
	}
	return l
}

func (l routeLayout) slices(a []float32) (dst, src [4][]float32) {
	for i := 0; i < 4; i++ {
		dst[i] = a[l.dst[i] : l.dst[i]+routeN : l.dst[i]+routeN]
		src[i] = a[l.src[i] : l.src[i]+routeN : l.src[i]+routeN]
	}
	return
}

func sameArena(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: element %d is %v, want %v", what, i, got[i], want[i])
		}
	}
}

// TestDCT32RoutePairQuad covers every overlap the routing refuses and the in-place
// case it accepts, for the pair and the quad.
func TestDCT32RoutePairQuad(t *testing.T) {
	p := NewDCTPlan32(routeN, 4)
	forced := *p
	forced.quad = true
	generic := *p
	generic.quad = false

	type kase struct {
		name string
		l    routeLayout
		kern bool // the kernel route is taken
	}
	var cases []kase
	cases = append(cases, kase{"disjoint", disjointLayout(), true})
	inplace := disjointLayout()
	inplace.src = inplace.dst
	cases = append(cases, kase{"in place", inplace, true})
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i == j {
				continue
			}
			dd := disjointLayout()
			dd.dst[j] = dd.dst[i] + 7
			cases = append(cases, kase{"dst-dst", dd, false})
			ds := disjointLayout()
			ds.src[j] = ds.dst[i] + 5
			cases = append(cases, kase{"dst-src", ds, false})
		}
	}
	for ci, c := range cases {
		a := routeArena(uint64(ci + 1))
		kernel := append([]float32(nil), a...)
		seq := append([]float32(nil), a...)
		got := append([]float32(nil), a...)

		// quad
		dst, src := c.l.slices(got)
		forced.DCTQuad(dst, src, NormBackward)
		sd, ss := c.l.slices(seq)
		for i := 0; i < 4; i++ {
			generic.DCT(sd[i], ss[i], NormBackward)
		}
		kd, ks := c.l.slices(kernel)
		dct4n64F32Quad(&kd, &ks, 1)
		if c.kern {
			sameArena(t, "quad "+c.name, got, kernel)
		} else {
			sameArena(t, "quad "+c.name, got, seq)
		}

		// pair: its own four streams (0 and 1)
		got = append(got[:0], a...)
		kernel = append(kernel[:0], a...)
		seq = append(seq[:0], a...)
		dst, src = c.l.slices(got)
		forced.DCTPair(dst[0], dst[1], src[0], src[1], NormBackward)
		sd, ss = c.l.slices(seq)
		generic.DCT(sd[0], ss[0], NormBackward)
		generic.DCT(sd[1], ss[1], NormBackward)
		kd, ks = c.l.slices(kernel)
		dct4n64F32Pair(kd[0], kd[1], ks[0], ks[1], 1)
		pairOverlap := overlap(dst[0], src[1]) || overlap(dst[1], src[0]) || overlap(dst[0], dst[1])
		if !pairOverlap {
			sameArena(t, "pair "+c.name, got, kernel)
		} else {
			sameArena(t, "pair "+c.name, got, seq)
		}
	}

	// the kernel route differs from the generic plan, so the cases above can tell
	// the two apart.
	a := routeArena(77)
	l := disjointLayout()
	d1, s1 := l.slices(append([]float32(nil), a...))
	d2, s2 := l.slices(append([]float32(nil), a...))
	forced.DCTQuad(d1, s1, NormBackward)
	generic.DCTQuad(d2, s2, NormBackward)
	differ := false
	for i := range d1 {
		for j := range d1[i] {
			differ = differ || d1[i][j] != d2[i][j]
		}
	}
	if !differ {
		t.Error("the kernel route gives the generic plan's bits: the routing tests cannot tell them apart")
	}
}

// TestDCT32RouteScaled: the scaled kernels (ortho and forward norms) are the same
// route as the unscaled ones.
func TestDCT32RouteScaled(t *testing.T) {
	p := NewDCTPlan32(routeN, 4)
	forced := *p
	forced.quad = true
	for _, norm := range []Norm{NormOrtho, NormForward} {
		for _, inverse := range []bool{false, true} {
			a := routeArena(5)
			l := disjointLayout()
			got := append([]float32(nil), a...)
			want := append([]float32(nil), a...)
			dst, src := l.slices(got)
			wd, ws := l.slices(want)
			f := float32(norm.scale(2*routeN, inverse))
			dct4n64F32Quad(&wd, &ws, f)
			if inverse {
				forced.IDCTQuad(dst, src, norm)
			} else {
				forced.DCTQuad(dst, src, norm)
			}
			sameArena(t, "scaled quad", got, want)
			got = append(got[:0], a...)
			want = append(want[:0], a...)
			dst, src = l.slices(got)
			wd, ws = l.slices(want)
			dct4n64F32Pair(wd[0], wd[1], ws[0], ws[1], f)
			if inverse {
				forced.IDCTPair(dst[0], dst[1], src[0], src[1], norm)
			} else {
				forced.DCTPair(dst[0], dst[1], src[0], src[1], norm)
			}
			sameArena(t, "scaled pair", got, want)
		}
	}
}
