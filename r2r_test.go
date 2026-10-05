package fft

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"sync"
	"testing"
)

// The oracle below is written from scipy.fft's documented definitions (the
// Notes of scipy.fft.dct and scipy.fft.dst, scipy 1.18), as plain O(n²) sums.
// It shares nothing with the fast kernels but the definitions themselves.

// oracleScale is scipy's normalisation factor for logical size m: the forward
// transform is unscaled under "backward", the inverse divides by m; "forward"
// swaps the two; "ortho" divides both by sqrt(m).
func oracleScale(norm Norm, m int, inverse bool) float64 {
	switch norm {
	case NormOrtho:
		return 1 / math.Sqrt(float64(m))
	case NormBackward:
		if inverse {
			return 1 / float64(m)
		}
	case NormForward:
		if !inverse {
			return 1 / float64(m)
		}
	}
	return 1
}

// oracleR2R computes scipy.fft.{dct,idct,dst,idst}(x, type=typ, norm=norm).
// scipy defines idct/idst of type t as the dct/dst of type 1, 3, 2, 4 (for
// t = 1..4) with the inverse normalisation.
func oracleR2R(x []float64, cosine bool, typ int, norm Norm, inverse bool) []float64 {
	n := len(x)
	if inverse {
		typ = [...]int{0, 1, 3, 2, 4}[typ]
	}
	ortho := norm == NormOrtho
	xs := append([]float64(nil), x...)
	y := make([]float64, n)
	N := float64(n)
	var m int
	if cosine {
		switch typ {
		case 1:
			m = 2 * (n - 1)
			if ortho {
				xs[0] *= math.Sqrt2
				xs[n-1] *= math.Sqrt2
			}
			for k := 0; k < n; k++ {
				s := xs[0] + math.Pow(-1, float64(k))*xs[n-1]
				for j := 1; j <= n-2; j++ {
					s += 2 * xs[j] * math.Cos(math.Pi*float64(k*j)/float64(n-1))
				}
				y[k] = s
			}
			if ortho {
				y[0] /= math.Sqrt2
				y[n-1] /= math.Sqrt2
			}
		case 2:
			m = 2 * n
			for k := 0; k < n; k++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += 2 * xs[j] * math.Cos(math.Pi*float64(k*(2*j+1))/(2*N))
				}
				y[k] = s
			}
			if ortho {
				y[0] /= math.Sqrt2
			}
		case 3:
			m = 2 * n
			if ortho {
				xs[0] *= math.Sqrt2
			}
			for k := 0; k < n; k++ {
				s := xs[0]
				for j := 1; j < n; j++ {
					s += 2 * xs[j] * math.Cos(math.Pi*float64((2*k+1)*j)/(2*N))
				}
				y[k] = s
			}
		case 4:
			m = 2 * n
			for k := 0; k < n; k++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += 2 * xs[j] * math.Cos(math.Pi*float64((2*k+1)*(2*j+1))/(4*N))
				}
				y[k] = s
			}
		}
	} else {
		switch typ {
		case 1:
			m = 2 * (n + 1)
			for k := 0; k < n; k++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += 2 * xs[j] * math.Sin(math.Pi*float64((k+1)*(j+1))/(N+1))
				}
				y[k] = s
			}
		case 2:
			m = 2 * n
			for k := 0; k < n; k++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += 2 * xs[j] * math.Sin(math.Pi*float64((k+1)*(2*j+1))/(2*N))
				}
				y[k] = s
			}
			if ortho {
				y[n-1] /= math.Sqrt2
			}
		case 3:
			m = 2 * n
			if ortho {
				xs[n-1] *= math.Sqrt2
			}
			for k := 0; k < n; k++ {
				s := math.Pow(-1, float64(k)) * xs[n-1]
				for j := 0; j <= n-2; j++ {
					s += 2 * xs[j] * math.Sin(math.Pi*float64((2*k+1)*(j+1))/(2*N))
				}
				y[k] = s
			}
		case 4:
			m = 2 * n
			for k := 0; k < n; k++ {
				s := 0.0
				for j := 0; j < n; j++ {
					s += 2 * xs[j] * math.Sin(math.Pi*float64((2*k+1)*(2*j+1))/(4*N))
				}
				y[k] = s
			}
		}
	}
	f := oracleScale(norm, m, inverse)
	for i := range y {
		y[i] *= f
	}
	return y
}

func r2rSignal(n, seed int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(0.37*float64(i)+float64(seed)) + 0.5*math.Cos(1.3*float64(i*i%7)) - 0.1*float64(seed%3)
	}
	return x
}

func maxAbs(x []float64) float64 {
	m := 0.0
	for _, v := range x {
		m = math.Max(m, math.Abs(v))
	}
	return m
}

// closeTo compares at an absolute tolerance of rtol·n·max(1, ‖want‖∞): the
// error of an FFT-based transform grows with the size of the largest output.
func closeTo(t *testing.T, what string, got, want []float64, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", what, len(got), len(want))
	}
	tol := rtol * float64(max(1, len(want))) * math.Max(1, maxAbs(want))
	for i := range want {
		if d := math.Abs(got[i] - want[i]); !(d <= tol) {
			t.Fatalf("%s: index %d: got %.17g want %.17g (|d|=%.3g > %.3g)", what, i, got[i], want[i], d, tol)
		}
	}
}

// r2rLengths covers every small length (each parity, primes, powers of two)
// and larger odd/even/prime/smooth/Bluestein lengths.
var r2rLengths = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 32, 33, 64, 97, 100, 127, 128, 210, 257, 1000, 1009}

// r2rCall runs one of the four 1-D transforms through a fresh plan, writing
// in place (dst aliasing src) to exercise aliasing, and through the
// package-level function; both must agree exactly.
func r2rCall(t *testing.T, x []float64, cosine bool, typ int, norm Norm, inverse bool) []float64 {
	t.Helper()
	n := len(x)
	buf := append([]float64(nil), x...)
	var viaPlan, viaFunc []float64
	if cosine {
		p := NewDCTPlan(n, typ)
		if p.Len() != n || p.Type() != typ {
			t.Fatalf("DCTPlan reports n=%d type=%d", p.Len(), p.Type())
		}
		if inverse {
			viaPlan, viaFunc = p.IDCT(buf, buf, norm), IDCT(x, typ, norm)
		} else {
			viaPlan, viaFunc = p.DCT(buf, buf, norm), DCT(x, typ, norm)
		}
	} else {
		p := NewDSTPlan(n, typ)
		if p.Len() != n || p.Type() != typ {
			t.Fatalf("DSTPlan reports n=%d type=%d", p.Len(), p.Type())
		}
		if inverse {
			viaPlan, viaFunc = p.IDST(buf, buf, norm), IDST(x, typ, norm)
		} else {
			viaPlan, viaFunc = p.DST(buf, buf, norm), DST(x, typ, norm)
		}
	}
	for i := range viaPlan {
		if viaPlan[i] != viaFunc[i] {
			t.Fatalf("plan and function disagree at %d: %v vs %v", i, viaPlan[i], viaFunc[i])
		}
	}
	return viaFunc
}

func r2rLabel(cosine bool, typ int, norm Norm, inverse bool, n int) string {
	name := "dst"
	if cosine {
		name = "dct"
	}
	if inverse {
		name = "i" + name
	}
	return name + strconv.Itoa(typ) + "/" + norm.String() + "/n=" + strconv.Itoa(n)
}

// TestR2RMatchesOracle: every family × type × norm × direction × length
// against the O(n²) definitions.
func TestR2RMatchesOracle(t *testing.T) {
	for _, cosine := range []bool{true, false} {
		for typ := 1; typ <= 4; typ++ {
			for _, norm := range allNorms {
				for _, inverse := range []bool{false, true} {
					for _, n := range r2rLengths {
						if n < minR2RLen(cosine, typ) {
							continue
						}
						x := r2rSignal(n, typ+n)
						got := r2rCall(t, x, cosine, typ, norm, inverse)
						want := oracleR2R(x, cosine, typ, norm, inverse)
						closeTo(t, r2rLabel(cosine, typ, norm, inverse, n), got, want, 1e-13)
					}
				}
			}
		}
	}
}

// TestR2RRoundTrip: the inverse undoes the forward transform for every norm,
// type and length, in both orders.
func TestR2RRoundTrip(t *testing.T) {
	for _, cosine := range []bool{true, false} {
		for typ := 1; typ <= 4; typ++ {
			for _, norm := range allNorms {
				for _, n := range r2rLengths {
					if n < minR2RLen(cosine, typ) {
						continue
					}
					x := r2rSignal(n, 3*typ+n)
					var back, back2 []float64
					if cosine {
						back = IDCT(DCT(x, typ, norm), typ, norm)
						back2 = DCT(IDCT(x, typ, norm), typ, norm)
					} else {
						back = IDST(DST(x, typ, norm), typ, norm)
						back2 = DST(IDST(x, typ, norm), typ, norm)
					}
					closeTo(t, "round trip "+r2rLabel(cosine, typ, norm, false, n), back, x, 1e-14)
					closeTo(t, "reverse round trip "+r2rLabel(cosine, typ, norm, false, n), back2, x, 1e-14)
				}
			}
		}
	}
}

// TestR2RIdentities checks the scipy identities without the oracle: the
// unnormalised DCT-III inverts the DCT-II up to 2N (DST likewise), DCT-I is
// its own inverse up to 2(N-1) and DST-I up to 2(N+1), DCT-IV and DST-IV up
// to 2N; and under NormOrtho every type is an orthonormal matrix
// (O·Oᵀ = I), which also pins scipy's orthogonalisation.
func TestR2RIdentities(t *testing.T) {
	for _, n := range []int{2, 3, 7, 8, 16, 25, 30} {
		x := r2rSignal(n, n)
		twoN := float64(2 * n)
		scaled := func(v []float64, f float64) []float64 {
			out := make([]float64, len(v))
			for i := range v {
				out[i] = v[i] * f
			}
			return out
		}
		for _, cosine := range []bool{true, false} {
			tr := DST
			if cosine {
				tr = DCT
			}
			closeTo(t, "III(II(x)) = 2N x", tr(tr(x, 2, NormBackward), 3, NormBackward), scaled(x, twoN), 1e-14)
			closeTo(t, "II(III(x)) = 2N x", tr(tr(x, 3, NormBackward), 2, NormBackward), scaled(x, twoN), 1e-14)
			closeTo(t, "IV(IV(x)) = 2N x", tr(tr(x, 4, NormBackward), 4, NormBackward), scaled(x, twoN), 1e-14)
			m := float64(2 * (n + 1))
			if cosine {
				m = float64(2 * (n - 1))
			}
			closeTo(t, "I(I(x)) = M x", tr(tr(x, 1, NormBackward), 1, NormBackward), scaled(x, m), 1e-14)

			for typ := 1; typ <= 4; typ++ {
				// Column j of O is the transform of the unit vector e_j.
				cols := make([][]float64, n)
				for j := range cols {
					e := make([]float64, n)
					e[j] = 1
					cols[j] = tr(e, typ, NormOrtho)
				}
				for a := 0; a < n; a++ {
					for b := 0; b < n; b++ {
						dot := 0.0
						for j := 0; j < n; j++ {
							dot += cols[j][a] * cols[j][b]
						}
						want := 0.0
						if a == b {
							want = 1
						}
						if math.Abs(dot-want) > 1e-13 {
							t.Fatalf("cosine=%v type %d n=%d: (O·Oᵀ)[%d][%d] = %v", cosine, typ, n, a, b, dot)
						}
					}
				}
			}
		}
	}
}

// scipyR2R is testdata/scipy_r2r.json, written by testdata/gen_scipy_r2r.py
// from real scipy.
type scipyR2R struct {
	Scipy string `json:"scipy"`
	Cases []struct {
		Fn    string    `json:"fn"`
		Type  int       `json:"type"`
		Norm  string    `json:"norm"`
		Shape []int     `json:"shape"`
		X     []float64 `json:"x"`
		Y     []float64 `json:"y"`
	} `json:"cases"`
	ND []struct {
		Fn    string    `json:"fn"`
		Type  int       `json:"type"`
		Norm  string    `json:"norm"`
		Shape []int     `json:"shape"`
		X     []float64 `json:"x"`
		Y     []float64 `json:"y"`
	} `json:"nd"`
}

func parseNorm(t *testing.T, s string) Norm {
	for _, m := range allNorms {
		if m.String() == s {
			return m
		}
	}
	t.Fatalf("unknown norm %q", s)
	return 0
}

// TestR2RAgainstScipy compares with values computed by scipy itself (the
// version is recorded in the file): every 1-D function, type and norm at
// lengths 1..31, and the N-D forms on a 3×4×5 array.
func TestR2RAgainstScipy(t *testing.T) {
	raw, err := os.ReadFile("testdata/scipy_r2r.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref scipyR2R
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if len(ref.Cases) != 4*4*3*8-2*3 || len(ref.ND) != 4*4*3 {
		t.Fatalf("reference has %d 1-D and %d N-D cases", len(ref.Cases), len(ref.ND))
	}
	fns := map[string]func([]float64, int, Norm) []float64{"dct": DCT, "idct": IDCT, "dst": DST, "idst": IDST}
	for _, c := range ref.Cases {
		got := fns[c.Fn](c.X, c.Type, parseNorm(t, c.Norm))
		closeTo(t, "scipy "+ref.Scipy+" "+c.Fn+strconv.Itoa(c.Type)+" "+c.Norm+" n="+strconv.Itoa(len(c.X)), got, c.Y, 1e-14)
	}
	nfns := map[string]func([]float64, []int, int, Norm) []float64{"dctn": DCTN, "idctn": IDCTN, "dstn": DSTN, "idstn": IDSTN}
	for _, c := range ref.ND {
		got := nfns[c.Fn](c.X, c.Shape, c.Type, parseNorm(t, c.Norm))
		closeTo(t, "scipy "+c.Fn+strconv.Itoa(c.Type)+" "+c.Norm, got, c.Y, 1e-14)
	}
}

// TestR2RNDMatchesAxes: the N-D transform equals the 1-D one applied along
// each axis by hand, round-trips, leaves its input alone, and treats an empty
// shape as a scalar left unchanged.
func TestR2RNDMatchesAxes(t *testing.T) {
	shape := []int{4, 3, 5}
	x := r2rSignal(60, 1)
	orig := append([]float64(nil), x...)
	for _, cosine := range []bool{true, false} {
		for typ := 2; typ <= 4; typ++ {
			for _, norm := range allNorms {
				want := append([]float64(nil), x...)
				strides := []int{15, 5, 1}
				for ax, n := range shape {
					for base := 0; base < 60; base++ {
						// base is a line start iff its coordinate on ax is 0.
						if (base/strides[ax])%n != 0 {
							continue
						}
						line := make([]float64, n)
						for i := range line {
							line[i] = want[base+i*strides[ax]]
						}
						if cosine {
							line = DCT(line, typ, norm)
						} else {
							line = DST(line, typ, norm)
						}
						for i := range line {
							want[base+i*strides[ax]] = line[i]
						}
					}
				}
				fwd, inv := DSTN, IDSTN
				if cosine {
					fwd, inv = DCTN, IDCTN
				}
				got := fwd(x, shape, typ, norm)
				closeTo(t, "N-D vs per-axis", got, want, 1e-14)
				closeTo(t, "N-D round trip", inv(got, shape, typ, norm), x, 1e-14)
			}
		}
	}
	for i := range x {
		if x[i] != orig[i] {
			t.Fatal("N-D transform modified its input")
		}
	}
	if got := DCTN([]float64{2.5}, nil, 2, NormOrtho); len(got) != 1 || got[0] != 2.5 {
		t.Fatalf("empty shape: %v", got)
	}
}

// TestR2RLongerSlices: a longer dst or src is accepted; only Len() values are
// written and the returned slice has length Len().
func TestR2RLongerSlices(t *testing.T) {
	const n = 9
	src := r2rSignal(n+2, 4)
	for typ := 1; typ <= 4; typ++ {
		dst := make([]float64, n+3)
		dst[n], dst[n+1], dst[n+2] = 7, 8, 9
		got := NewDCTPlan(n, typ).DCT(dst, src, NormOrtho)
		closeTo(t, "DCT longer slices", got, DCT(src[:n], typ, NormOrtho), 0)
		got = NewDSTPlan(n, typ).IDST(dst, src, NormForward)
		closeTo(t, "IDST longer slices", got, IDST(src[:n], typ, NormForward), 0)
		if dst[n] != 7 || dst[n+1] != 8 || dst[n+2] != 9 {
			t.Fatalf("type %d wrote past Len(): %v", typ, dst[n:])
		}
	}
}

// TestR2RConcurrent: one plan shared by many goroutines (run under -race in
// CI) gives every caller the right answer.
func TestR2RConcurrent(t *testing.T) {
	const n = 64
	plans := []*DCTPlan{NewDCTPlan(n, 1), NewDCTPlan(n, 2), NewDCTPlan(n, 4), NewDCTPlan(n+1, 4)}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for _, p := range plans {
				x := r2rSignal(p.Len(), g)
				want := oracleR2R(x, true, p.Type(), NormOrtho, false)
				for r := 0; r < 20; r++ {
					got := p.DCT(make([]float64, p.Len()), x, NormOrtho)
					for i := range got {
						if math.Abs(got[i]-want[i]) > 1e-12 {
							t.Errorf("goroutine %d: type %d index %d: %v want %v", g, p.Type(), i, got[i], want[i])
							return
						}
					}
				}
			}
		}(g)
	}
	wg.Wait()
}

// TestR2RPanics: bad arguments fail with the package's message, before
// anything is allocated.
func TestR2RPanics(t *testing.T) {
	x := make([]float64, 8)
	mustPanicWith(t, "DCT type 0", "fft: unknown DCT type 0", func() { DCT(x, 0, NormBackward) })
	mustPanicWith(t, "DST type 5", "fft: unknown DST type 5", func() { DST(x, 5, NormBackward) })
	mustPanicWith(t, "IDCT type -1", "fft: unknown DCT type", func() { IDCT(x, -1, NormBackward) })
	mustPanicWith(t, "IDST type 9", "fft: unknown DST type", func() { IDST(x, 9, NormBackward) })
	mustPanicWith(t, "DCT-I n=1", "fft: DCT-I length 1 is below its minimum 2", func() { DCT(x[:1], 1, NormBackward) })
	mustPanicWith(t, "IDCT-I n=1", "fft: DCT-I length 1", func() { IDCT(x[:1], 1, NormOrtho) })
	mustPanicWith(t, "DCT-II empty", "fft: DCT-II length 0", func() { DCT(nil, 2, NormBackward) })
	mustPanicWith(t, "DST-I empty", "fft: DST-I length 0", func() { DST(nil, 1, NormBackward) })
	mustPanicWith(t, "DST-IV negative", "fft: DST-IV length -3", func() { NewDSTPlan(-3, 4) })
	mustPanicWith(t, "NewDCTPlan type 7", "fft: unknown DCT type 7", func() { NewDCTPlan(1<<40, 7) })
	mustPanicWith(t, "NewDCTPlan n=1 type 1", "fft: DCT-I length 1", func() { NewDCTPlan(1, 1) })
	mustPanicWith(t, "DCT bad norm", "fft: unknown Norm", func() { DCT(x, 2, Norm(3)) })
	mustPanicWith(t, "DST bad norm", "fft: unknown Norm", func() { DST(x, 2, Norm(-1)) })
	dp, sp := NewDCTPlan(8, 3), NewDSTPlan(8, 3)
	mustPanicWith(t, "DCTPlan bad norm", "fft: unknown Norm", func() { dp.DCT(x, x, Norm(4)) })
	mustPanicWith(t, "DSTPlan bad norm", "fft: unknown Norm", func() { sp.IDST(x, x, Norm(4)) })
	mustPanicWith(t, "DCTPlan short dst", "fft: DCTPlan slice shorter", func() { dp.DCT(x[:7], x, NormBackward) })
	mustPanicWith(t, "DCTPlan short src", "fft: DCTPlan slice shorter", func() { dp.IDCT(x, x[:7], NormBackward) })
	mustPanicWith(t, "DSTPlan short dst", "fft: DSTPlan slice shorter", func() { sp.DST(x[:1], x, NormBackward) })
	mustPanicWith(t, "DSTPlan short src", "fft: DSTPlan slice shorter", func() { sp.IDST(x, nil, NormBackward) })
	mustPanicWith(t, "DCTN shape mismatch", "fft: shape product does not match", func() { DCTN(x, []int{3, 3}, 2, NormBackward) })
	mustPanicWith(t, "DCTN huge shape", "fft: shape product overflows int", func() { DCTN(nil, []int{1 << 32, 1 << 32}, 2, NormBackward) })
	mustPanicWith(t, "DCTN axis too short for DCT-I", "fft: DCT-I length 1", func() { DCTN(x, []int{8, 1}, 1, NormBackward) })
	mustPanicWith(t, "IDCTN bad type", "fft: unknown DCT type 0", func() { IDCTN(x, []int{8}, 0, NormBackward) })
	mustPanicWith(t, "DSTN bad norm", "fft: unknown Norm", func() { DSTN(x, []int{8}, 1, Norm(9)) })
	mustPanicWith(t, "IDSTN zero axis", "fft: shape lengths must be positive", func() { IDSTN(nil, []int{0}, 1, NormBackward) })
}

// TestR2RScaleReal covers the scaling helpers' branches directly.
func TestR2RScaleReal(t *testing.T) {
	x := []float64{1, 2, 3}
	scaleReal(x, 1)
	scaleReal(x, 2)
	scaleAlternating(x, 0.5)
	if x[0] != 1 || x[1] != -2 || x[2] != 3 {
		t.Fatalf("scaleReal/scaleAlternating: %v", x)
	}
}
