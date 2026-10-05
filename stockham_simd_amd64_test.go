//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestStockhamPassMatchesScalar runs every smooth length up to 2100, and a few
// larger ones, through the Stockham engine twice — AVX2 pass kernels on, then
// off — and requires the two results to be bit-identical, forward and inverse,
// in place and not. The range covers radix 2/3/4/5/8 passes at ido == 2, at
// odd ido (the 128-bit tail) and at long ido, with many and few blocks. It is
// restricted to GOAMD64 < v3, where the scalar oracle does not fuse multiply-
// adds; the kernels never do.
func TestStockhamPassMatchesScalar(t *testing.T) {
	if !kernels.UseStockhamAVX2 {
		t.Skip("no AVX2 on this CPU: the scalar passes run and there is nothing to compare")
	}
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	modes := []struct {
		name         string
		avx2, avx512 bool
	}{{"AVX2", true, false}}
	if kernels.UseStockhamAVX512 {
		modes = append(modes, struct {
			name         string
			avx2, avx512 bool
		}{"AVX-512", true, true})
	} else {
		t.Log("no AVX-512 on this CPU: only the AVX2 kernels are compared")
	}
	sizes := []int{4096, 8192, 16384, 20160, 45000, 65536, 65536 * 3}
	for n := 2; n <= 2100; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		if !factorsAreSmall(n) {
			continue
		}
		p := newSKPlan(n)
		for s, x := range simdSignals(n) {
			for _, inverse := range []bool{false, true} {
				kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = false, false
				scalar := make([]complex128, n)
				p.transform(scalar, x, inverse)
				for _, m := range modes {
					kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = m.avx2, m.avx512
					simd := make([]complex128, n)
					p.transform(simd, x, inverse)
					alias := append([]complex128(nil), x...)
					p.transform(alias, alias, inverse)
					for i := range scalar {
						if !sameBits(simd[i], scalar[i]) || !sameBits(alias[i], scalar[i]) {
							t.Fatalf("%s n=%d signal %d inverse=%v index %d: %v (in place %v) vs scalar %v (factors %v)",
								m.name, n, s, inverse, i, simd[i], alias[i], scalar[i], skFactorize(n))
						}
					}
				}
			}
		}
	}
}

// TestUntangleMatchesScalar holds the AVX2 real-FFT untangle to the Go loop,
// bit for bit, at every half-length m up to 700 and a few large ones, on the
// same generic, signed-zero and infinite signals as the pass kernels.
func TestUntangleMatchesScalar(t *testing.T) {
	if !kernels.UseUntangleAVX2 {
		t.Skip("no AVX2 on this CPU: the Go loop runs and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseUntangleAVX2 = v }(kernels.UseUntangleAVX2)
	sizes := []int{2048, 4097, 32768}
	for m := 2; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	for _, m := range sizes {
		tw := NewRealPlan(2 * m).tw
		for s, z := range simdSignals(m) {
			kernels.UseUntangleAVX2 = true
			simd := make([]complex128, m+1)
			rfftUntangle(simd, z, tw, m)
			kernels.UseUntangleAVX2 = false
			scalar := make([]complex128, m+1)
			rfftUntangle(scalar, z, tw, m)
			for k := range scalar {
				if !sameBits(simd[k], scalar[k]) {
					t.Fatalf("m=%d signal %d bin %d: AVX2 %v vs Go %v", m, s, k, simd[k], scalar[k])
				}
			}
		}
	}
}

// TestRetangleMatchesScalar holds the AVX2 inverse untangle to the Go loop,
// bit for bit, the same way.
func TestRetangleMatchesScalar(t *testing.T) {
	if !kernels.UseUntangleAVX2 {
		t.Skip("no AVX2 on this CPU: the Go loop runs and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseUntangleAVX2 = v }(kernels.UseUntangleAVX2)
	sizes := []int{2048, 4097, 32768}
	for m := 2; m <= 700; m++ {
		sizes = append(sizes, m)
	}
	for _, m := range sizes {
		tw := NewRealPlan(2 * m).tw
		for s, x := range simdSignals(m + 1) {
			for _, h := range []float64{0.5, 0.5 / float64(m)} {
				kernels.UseUntangleAVX2 = true
				simd := make([]complex128, m)
				irfftRetangle(simd, x, tw, m, h)
				kernels.UseUntangleAVX2 = false
				scalar := make([]complex128, m)
				irfftRetangle(scalar, x, tw, m, h)
				for k := range scalar {
					if !sameBits(simd[k], scalar[k]) {
						t.Fatalf("m=%d signal %d h=%v bin %d: AVX2 %v vs Go %v", m, s, h, k, simd[k], scalar[k])
					}
				}
			}
		}
	}
}

// TestStockhamBatchMatchesScalar holds the AVX2 batched pass kernels to the Go
// batched pass, bit for bit, pass by pass: every batched radix at several
// l1 and ido (1, the final pass, through odd and even), batches of 1 to 8
// lines (the odd one runs the 128-bit tail), and point strides wider than the
// batch on either side, forward and inverse, on the generic, signed-zero and
// infinite signals. Outputs start as a sentinel, so a kernel that wrote
// between the rows of the strip would fail too.
func TestStockhamBatchMatchesScalar(t *testing.T) {
	if !kernels.UseStockhamBatchAVX2 {
		t.Skip("no AVX2 on this CPU: the Go batched pass runs and there is nothing to compare")
	}
	defer func(v bool) { kernels.UseStockhamBatchAVX2 = v }(kernels.UseStockhamBatchAVX2)
	sentinel := complex(-7.25, 3.5)
	for _, r := range []int{2, 3, 4, 5, 8} {
		for _, l1 := range []int{1, 2, 3, 5} {
			for _, ido := range []int{1, 2, 3, 4, 5, 7, 8, 13} {
				var factors []int
				stage := 0
				if l1 > 1 {
					factors, stage = append(factors, l1), 1
				}
				factors = append(factors, r)
				if ido > 1 {
					factors = append(factors, ido)
				}
				n := r * ido * l1
				st := newSKPlanFactors(n, factors).stages[stage]
				if st.r != r || st.l1 != l1 || st.ido != ido {
					t.Fatalf("stage %+v, want r=%d l1=%d ido=%d", st, r, l1, ido)
				}
				fwd, conj := st.batchTwiddles()
				for _, w := range []int{1, 2, 3, 4, 5, 8} {
					for _, gap := range [][2]int{{0, 0}, {3, 0}, {0, 1}, {2, 5}} {
						sIn, sOut := w+gap[0], w+gap[1]
						size := (n-1)*sIn + w
						for s, x := range simdSignals(size) {
							for _, inverse := range []bool{false, true} {
								tw := fwd
								if inverse {
									tw = conj
								}
								var out [2][]complex128
								for k, on := range []bool{false, true} {
									kernels.UseStockhamBatchAVX2 = on
									out[k] = make([]complex128, (n-1)*sOut+w)
									for i := range out[k] {
										out[k][i] = sentinel
									}
									st.passBatch(out[k], x, tw, w, sIn, sOut, inverse)
								}
								for i := range out[0] {
									if !sameBits(out[1][i], out[0][i]) {
										t.Fatalf("r=%d l1=%d ido=%d w=%d sIn=%d sOut=%d signal %d inverse=%v index %d: AVX2 %v vs Go %v",
											r, l1, ido, w, sIn, sOut, s, inverse, i, out[1][i], out[0][i])
									}
								}
							}
						}
					}
				}
			}
		}
	}
	// A radix without a batched kernel is refused, and the Go pass runs.
	kernels.UseStockhamBatchAVX2 = true
	if kernels.StockhamBatchPass(7, 1, 1, make([]complex128, 7), make([]complex128, 7), nil, 1, 1, 1, false) {
		t.Fatal("radix 7 has no batched kernel")
	}
}
