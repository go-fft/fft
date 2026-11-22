//go:build amd64

package fft

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/rand"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The generated kernels' output bits are pinned. The generator picks, among
// schedules, the one with the fewest spills; which multiplication of an add of two
// products is fused must not depend on that choice, or a different schedule would
// silently change the rounding of every result (it did, in the first version of
// the schedule search). The digests below were computed on the kernels of the
// commit before the search was added, so a schedule change that moves a bit
// fails here, on an AVX-512 host (kernels.UseStockhamAVX512), before it ships.
//
// If a deliberate change to the arithmetic (not the schedule) moves the bits, copy
// the digests the failure prints and say why in the PR.
var dct23Digests = map[string]string{
	"Dct2x32": "672e2036b715b2350a168bce8cedaabf3a03580740ca42906921c533c8302190",
	"Dct3x32": "bd6f79e2896b41da32d3445ec35db066375895446752e79438cdd7bca9b7e99b",
	"Dct2x64": "666f9dc9b8984e23cb83b3dfe84d27f308a6dc56da5cbda2effb129e9ecf3929",
	"Dct3x64": "613f4f8d59c2159dc57b2b68f2884fb1bf7032ec3a92143aea71f7347c3b0c71",
}

func dct23Digest(name string, n int, run func(dst, src []float64, a0 float64)) string {
	rng := rand.New(rand.NewSource(int64(n) * 7919))
	h := sha256.New()
	var buf [8]byte
	special := []float64{0, math.Copysign(0, -1), 1, -1, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.NaN()}
	for it := 0; it < 2000; it++ {
		src := make([]float64, n)
		for i := range src {
			switch it % 4 {
			case 0:
				src[i] = rng.Float64()*2 - 1
			case 1:
				src[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(80)-40))
			case 2:
				src[i] = math.Float64frombits(rng.Uint64())
			default:
				src[i] = special[rng.Intn(len(special))]
			}
		}
		a0 := []float64{1, math.Sqrt2, 0.5, -3}[it%4]
		dst := make([]float64, n)
		run(dst, src, a0)
		for _, v := range dst {
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
			h.Write(buf[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestDCT23KernelBitsPinned(t *testing.T) {
	if !kernels.UseStockhamAVX512 {
		t.Skip("no AVX-512 here: the kernels do not run")
	}
	got := map[string]string{
		"Dct2x32": dct23Digest("Dct2x32", 32, func(d, s []float64, _ float64) { kernels.Dct2x32(&d[0], &s[0]) }),
		"Dct3x32": dct23Digest("Dct3x32", 32, func(d, s []float64, a float64) { kernels.Dct3x32(&d[0], &s[0], a) }),
		"Dct2x64": dct23Digest("Dct2x64", 64, func(d, s []float64, _ float64) { kernels.Dct2x64(&d[0], &s[0]) }),
		"Dct3x64": dct23Digest("Dct3x64", 64, func(d, s []float64, a float64) { kernels.Dct3x64(&d[0], &s[0], a) }),
	}
	for name, g := range got {
		if g != dct23Digests[name] {
			t.Errorf("%s: output bits changed: digest %s, pinned %s", name, g, dct23Digests[name])
		}
	}
}
