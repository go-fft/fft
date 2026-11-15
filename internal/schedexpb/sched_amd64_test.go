package schedexp

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

func twPtr() *complex128 { return (*complex128)(unsafe.Pointer(&kernels.Fused256Table(false)[0])) }

func enable(t testing.TB) {
	if !kernels.UseStockhamAVX512 {
		t.Skip("no AVX-512")
	}
	old := kernels.Fused256On
	kernels.Fused256On = true
	if tt, ok := t.(*testing.T); ok {
		tt.Cleanup(func() { kernels.Fused256On = old })
	}
}

func inputs() [][]complex128 {
	r := rand.New(rand.NewSource(1))
	var in [][]complex128
	for k := 0; k < 30; k++ {
		x := make([]complex128, 256)
		for i := range x {
			x[i] = complex(2*r.Float64()-1, 2*r.Float64()-1)
			if k%5 == 4 {
				x[i] *= complex(math.Exp2(float64(r.Intn(40)-20)), math.Exp2(float64(r.Intn(40)-20)))
			}
		}
		in = append(in, x)
	}
	for j := 0; j < 256; j++ {
		for _, u := range []complex128{1, 1i} {
			x := make([]complex128, 256)
			x[j] = u
			in = append(in, x)
		}
	}
	z := make([]complex128, 256)
	z[3] = complex(math.Copysign(0, -1), 1)
	in = append(in, z)
	return in
}

func bitsEq(a, b []complex128) bool {
	for i := range a {
		if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) || math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
			return false
		}
	}
	return true
}

// TestVariantsBitEqual: every variant gives the shipped kernel's bits, on random
// data, data over 40 binades, every unit impulse (1 and i) and a negative zero.
func TestVariantsBitEqual(t *testing.T) {
	enable(t)
	tw := twPtr()
	ins := inputs()
	want := make([][]complex128, len(ins))
	for k, x := range ins {
		want[k] = make([]complex128, 256)
		if !kernels.Fused256((*[256]complex128)(want[k]), (*[256]complex128)(x), false) {
			t.Fatal("the shipped kernel refused")
		}
	}
	bad := 0
	for v, f := range variantFns {
		got := make([]complex128, 256)
		for k, x := range ins {
			f(&got[0], &x[0], tw)
			if !bitsEq(got, want[k]) {
				t.Errorf("variant %d (%s): differs from the shipped kernel on input %d", v, variantDescr[v], k)
				bad++
				break
			}
		}
	}
	t.Logf("%d variants checked against %d inputs, %d differ", len(variantFns), len(ins), bad)
}

const batch = 20000

func timeFn(f func()) float64 {
	t0 := time.Now()
	for i := 0; i < batch; i++ {
		f()
	}
	return float64(time.Since(t0).Nanoseconds()) / batch
}

func fixtures() (src, dst []complex128, tw *complex128) {
	src = make([]complex128, 256)
	dst = make([]complex128, 256)
	for i := range src {
		src[i] = complex(float64((i*7+1)%13)*0.1, float64((i*3+2)%11)*0.1)
	}
	return src, dst, twPtr()
}

// TestSchedRound times the shipped kernel and every variant once (SCHED_REV=1
// walks them backwards) and prints "name ns" lines for the driver. With
// SCHED_ONLY=k (-1: the shipped kernel) it loops that one for SCHED_SECS seconds
// (for perf stat).
func TestSchedRound(t *testing.T) {
	if os.Getenv("SCHED_ROUND") == "" && os.Getenv("SCHED_ONLY") == "" {
		t.Skip("driver only")
	}
	enable(t)
	src, dst, tw := fixtures()
	s, d := (*[256]complex128)(src), (*[256]complex128)(dst)
	base := func() { kernels.Fused256(d, s, false) }
	fn := func(i int) func() { return func() { variantFns[i](&dst[0], &src[0], tw) } }
	if only := os.Getenv("SCHED_ONLY"); only != "" {
		k, _ := strconv.Atoi(only)
		secs, _ := strconv.Atoi(os.Getenv("SCHED_SECS"))
		f := base
		if k >= 0 {
			f = fn(k)
		}
		end := time.Now().Add(time.Duration(secs) * time.Second)
		for time.Now().Before(end) {
			for i := 0; i < 10000; i++ {
				f()
			}
		}
		return
	}
	order := make([]int, 0, len(variantFns)+1)
	for i := -1; i < len(variantFns); i++ {
		order = append(order, i)
	}
	if os.Getenv("SCHED_REV") != "" {
		for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
			order[i], order[j] = order[j], order[i]
		}
	}
	for _, i := range order {
		f := base
		if i >= 0 {
			f = fn(i)
		}
		timeFn(f) // warm
		fmt.Printf("%d %.3f\n", i, timeFn(f))
	}
}

func TestDescr(t *testing.T) {
	if os.Getenv("SCHED_DESCR") == "" {
		t.Skip()
	}
	for i, d := range variantDescr {
		fmt.Printf("%d %s\n", i, d)
	}
}
