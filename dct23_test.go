package fft

import (
	"math"
	"testing"
)

// The fused DCT-II/III (lengths 32, 64, 128, dct23.go) are held to the #121 contract:
// each result within 2·ε·log2 n of a double-double oracle, for every norm and both
// directions, on the plan the package builds, and a mutated table must break it.

var dct23Lens = []int{32, 64, 128}

func dct23Plans(n, typ int) (fast, pooled *DCTPlan) {
	fast = NewDCTPlan(n, typ)
	p := *fast
	p.fast = nil
	return fast, &p
}

func TestDCT23Routed(t *testing.T) {
	for _, n := range []int{31, 33, 63, 65, 127, 129, 256} {
		if NewDCTPlan(n, 2).fast != nil || NewDCTPlan(n, 3).fast != nil {
			t.Errorf("n=%d is routed to the fused kernel", n)
		}
	}
	for _, n := range dct23Lens {
		for _, typ := range []int{1, 4} {
			if NewDCTPlan(n, typ).fast != nil {
				t.Errorf("type %d n=%d is routed to the fused DCT-II/III", typ, n)
			}
		}
		for _, typ := range []int{2, 3} {
			if NewDCTPlan(n, typ).fast == nil {
				t.Errorf("type %d n=%d is not routed to the fused kernel", typ, n)
			}
		}
	}
}

func TestDCT23Oracle(t *testing.T) {
	for _, n := range dct23Lens {
		b64 := 2 * eps64 * math.Log2(float64(n))
		for _, typ := range []int{2, 3} {
			fast, _ := dct23Plans(n, typ)
			worst := 0.0
			for seed := uint64(1); seed <= 4; seed++ {
				x := accRealF64(n, seed)
				for _, norm := range accNorms {
					for _, inverse := range []bool{false, true} {
						want := oR2R(cddFromR(x), true, typ, norm, inverse)
						var got []float64
						if inverse {
							got = fast.IDCT(make([]float64, n), x, norm)
						} else {
							got = fast.DCT(make([]float64, n), x, norm)
						}
						e := errR(got, want)
						worst = math.Max(worst, e)
						if e > b64 {
							t.Errorf("n=%d type %d seed %d %v inverse=%v: error %.3g exceeds %.3g", n, typ, seed, norm, inverse, e, b64)
						}
					}
				}
			}
			t.Logf("n=%d type %d: worst error %.3g (bound %.3g)", n, typ, worst, b64)
		}
	}
}

// TestDCT23Impulses: every unit impulse, both directions, on an absolute bound.
func TestDCT23Impulses(t *testing.T) {
	for _, n := range dct23Lens {
		b64 := 2 * eps64 * math.Log2(float64(n))
		for _, typ := range []int{2, 3} {
			fast, _ := dct23Plans(n, typ)
			for i := 0; i < n; i++ {
				x := make([]float64, n)
				x[i] = 1
				for _, inverse := range []bool{false, true} {
					want := oR2R(cddFromR(x), true, typ, NormBackward, inverse)
					var got []float64
					if inverse {
						got = fast.IDCT(make([]float64, n), x, NormBackward)
					} else {
						got = fast.DCT(make([]float64, n), x, NormBackward)
					}
					if e := errR(got, want); e > b64 {
						t.Fatalf("n=%d type %d impulse %d inverse=%v: error %.3g exceeds %.3g", n, typ, i, inverse, e, b64)
					}
				}
			}
		}
	}
}

// TestDCT23AgainstPooled: the fused kernel and the generic route agree to a few ulps,
// and an in-place call gives the same bits as an out-of-place one.
func TestDCT23AgainstPooled(t *testing.T) {
	for _, n := range dct23Lens {
		for _, typ := range []int{2, 3} {
			fast, pooled := dct23Plans(n, typ)
			x := accRealF64(n, 7)
			for _, inverse := range []bool{false, true} {
				run := func(p *DCTPlan, dst, src []float64) []float64 {
					if inverse {
						return p.IDCT(dst, src, NormOrtho)
					}
					return p.DCT(dst, src, NormOrtho)
				}
				a := run(fast, make([]float64, n), x)
				b := run(pooled, make([]float64, n), x)
				for i := range a {
					if math.Abs(a[i]-b[i]) > 8*eps64*math.Log2(float64(n))*(1+math.Abs(b[i])) {
						t.Fatalf("n=%d type %d inverse=%v: [%d] fused %v pooled %v", n, typ, inverse, i, a[i], b[i])
					}
				}
				in := append([]float64(nil), x...)
				run(fast, in, in)
				for i := range a {
					if in[i] != a[i] {
						t.Fatalf("n=%d type %d inverse=%v: in place [%d] = %v, out of place %v", n, typ, inverse, i, in[i], a[i])
					}
				}
			}
		}
	}
}

// TestDCT23ValidatesBeforeWriting: a short slice or an unknown norm panics before dst
// is touched.
func TestDCT23ValidatesBeforeWriting(t *testing.T) {
	for _, n := range dct23Lens {
		for _, typ := range []int{2, 3} {
			p := NewDCTPlan(n, typ)
			for name, f := range map[string]func(dst, src []float64){
				"short dst":    func(dst, src []float64) { p.DCT(dst[:n-1], src, NormBackward) },
				"short src":    func(dst, src []float64) { p.DCT(dst, src[:n-1], NormBackward) },
				"short dst, I": func(dst, src []float64) { p.IDCT(dst[:n-1], src, NormBackward) },
				"bad norm":     func(dst, src []float64) { p.DCT(dst, src, Norm(99)) },
			} {
				dst, src := make([]float64, n), make([]float64, n)
				for i := range dst {
					dst[i], src[i] = 1234.5, 1
				}
				func() {
					defer func() {
						if recover() == nil {
							t.Errorf("n=%d type %d %s: no panic", n, typ, name)
						}
					}()
					f(dst, src)
				}()
				for i, v := range dst {
					if v != 1234.5 {
						t.Errorf("n=%d type %d %s: dst[%d] written before the panic", n, typ, name, i)
					}
				}
			}
		}
	}
}

// TestDCT23Mutation: a flipped table breaks the oracle bound by orders of magnitude,
// so the oracle tests above can see a wrong kernel.
func TestDCT23Mutation(t *testing.T) {
	n := 64
	b64 := 2 * eps64 * math.Log2(float64(n))
	x := accRealF64(n, 3)
	for _, typ := range []int{2, 3} {
		for _, mut := range []string{"c2", "mt", "c1", "cw"} {
			fast, _ := dct23Plans(n, typ)
			d := newDCT23(n) // a table of its own, flipped at k = 3
			if mut == "cw" {
				d.cw[3] = -d.cw[3]
			} else {
				tab := map[string][]float64{"c2": d.c2f, "mt": d.mtf, "c1": d.c1f}[mut]
				tab[2*3] = -tab[2*3]
			}
			fast.fast = d
			want := oR2R(cddFromR(x), true, typ, NormBackward, false)
			wantI := oR2R(cddFromR(x), true, typ, NormBackward, true)
			e := math.Max(errR(fast.DCT(make([]float64, n), x, NormBackward), want),
				errR(fast.IDCT(make([]float64, n), x, NormBackward), wantI))
			if e <= 100*b64 {
				t.Errorf("type %d mutation %s: error %.3g is not told from the bound %.3g", typ, mut, e, b64)
			}
		}
	}
}
