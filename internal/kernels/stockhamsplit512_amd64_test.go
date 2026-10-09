package kernels

import (
	"reflect"
	"slices"
	"strings"
	"testing"
)

// TestIntelSplit512Default: the 512-bit split layout runs with AVX2 and
// AVX-512 on an Intel CPU only.
func TestIntelSplit512Default(t *testing.T) {
	for _, c := range []struct{ avx2, avx512, intel, want bool }{
		{true, true, true, true}, {true, true, false, false}, {true, false, true, false},
		{false, true, true, false}, {false, false, false, false},
	} {
		if got := intelSplit512Default(c.avx2, c.avx512, c.intel); got != c.want {
			t.Errorf("intelSplit512Default(avx2=%v, avx512=%v, intel=%v) = %v", c.avx2, c.avx512, c.intel, got)
		}
	}
}

// TestIntelSplitModes512: every maximal run of radix-4/8 passes whose ido is
// a multiple of eight gets the 512-bit modes, in a power of two of at least
// intelSplit512MinN points; everything else stays interleaved.
func TestIntelSplitModes512(t *testing.T) {
	const b = intelSplit512Base
	for _, c := range []struct {
		r, ido []int
		want   []uint8
	}{
		{[]int{4, 8, 8}, []int{64, 8, 1}, []uint8{b + splitOut, b + splitIn, 0}},
		{[]int{4, 4, 4, 4}, []int{64, 16, 4, 1}, []uint8{b + splitOut, b + splitIn, 0, 0}},
		{[]int{4, 4, 4, 4, 4}, []int{256, 64, 16, 4, 1}, []uint8{b + splitOut, b + splitBoth, b + splitIn, 0, 0}},
		{[]int{8, 8, 16}, []int{128, 16, 1}, []uint8{b + splitOut, b + splitIn, 0}},
		{[]int{16, 4, 4}, []int{16, 4, 1}, []uint8{0, 0, 0}},
		{[]int{16, 4, 4, 4}, []int{64, 16, 4, 1}, []uint8{0, b + splitNone, 0, 0}},
		{[]int{2, 4, 4, 8}, []int{128, 32, 8, 1}, []uint8{0, b + splitOut, b + splitIn, 0}},
		{[]int{4, 16, 4}, []int{64, 4, 1}, []uint8{b + splitNone, 0, 0}},
		{[]int{4, 4, 8}, []int{32, 8, 1}, []uint8{0, 0, 0}},           // 128 < intelSplit512MinN
		{[]int{4, 3, 4, 8}, []int{96, 32, 8, 1}, []uint8{0, 0, 0, 0}}, // not a power of two
		{nil, nil, []uint8{}},
	} {
		if got := intelSplitModes512(c.r, c.ido); !slices.Equal(got, c.want) {
			t.Errorf("intelSplitModes512(%v, %v) = %v, want %v", c.r, c.ido, got, c.want)
		}
	}
	defer func(a, h, s bool) { UseStockhamSplit512, intelHasAVX512, UseStockhamSplit = a, h, s }(UseStockhamSplit512, intelHasAVX512, UseStockhamSplit)
	r, ido := []int{4, 8, 8}, []int{64, 8, 1}
	UseStockhamSplit512, intelHasAVX512, UseStockhamSplit = true, true, true
	if got := StockhamSplitModes(r, ido, []int{1, 4, 32}); got[0] != b+splitOut {
		t.Errorf("StockhamSplitModes, 512 on: %v", got)
	}
	intelHasAVX512 = false
	if got := StockhamSplitModes(r, ido, []int{1, 4, 32}); got[0] != splitOut {
		t.Errorf("StockhamSplitModes, 512 on without AVX-512: %v, want the 256-bit modes", got)
	}
}

// TestIntelSplitTwiddles512: per eight-point group and twiddle j, the real
// parts in the order 0, 4, 1, 5, 2, 6, 3, 7, then the imaginary parts,
// conjugated in the conjugate table; StockhamSplitTwiddles gives it when the
// 512-bit layout is on.
func TestIntelSplitTwiddles512(t *testing.T) {
	const n = 256
	root := make([]complex128, n)
	for k := range root {
		root[k] = complex(float64(k), -float64(k)-0.5)
	}
	r, ido, l1 := 4, 16, 4
	defer func(a, h bool) { UseStockhamSplit512, intelHasAVX512 = a, h }(UseStockhamSplit512, intelHasAVX512)
	UseStockhamSplit512, intelHasAVX512 = true, true
	fwd, conj := StockhamSplitTwiddles(r, ido, l1, root)
	if len(fwd) != (r-1)*ido || len(conj) != len(fwd) {
		t.Fatalf("lengths %d, %d", len(fwd), len(conj))
	}
	at := 0
	for i0 := 0; i0 < ido; i0 += 8 {
		for j := 1; j < r; j++ {
			w := func(d int) complex128 { return root[(j*l1*(i0+d))%n] }
			order := []int{0, 4, 1, 5, 2, 6, 3, 7}
			for q := 0; q < 4; q++ {
				a, b := w(order[2*q]), w(order[2*q+1])
				re, im := complex(real(a), real(b)), complex(imag(a), imag(b))
				if fwd[at+q] != re || conj[at+q] != re || fwd[at+4+q] != im || conj[at+4+q] != -im {
					t.Fatalf("i0=%d j=%d q=%d: %v %v %v %v", i0, j, q, fwd[at+q], fwd[at+4+q], conj[at+q], conj[at+4+q])
				}
			}
			at += 8
		}
	}
}

// TestIntelSplitModeOK: the modes, radices, ido and l1 each width's kernels
// accept, and the 512-bit modes refused without AVX-512.
func TestIntelSplitModeOK(t *testing.T) {
	defer func(h bool) { intelHasAVX512 = h }(intelHasAVX512)
	const b = intelSplit512Base
	for _, has := range []bool{true, false} {
		intelHasAVX512 = has
		for _, c := range []struct {
			mode       uint8
			r, ido, l1 int
			want       bool
		}{
			{splitOut, 4, 4, 1, true}, {splitNone, 8, 12, 3, true}, {splitBoth, 4, 6, 1, false},
			{splitIn, 16, 4, 1, false}, {0, 4, 4, 1, false}, {splitBoth, 4, 4, 0, false},
			{b + splitOut, 4, 8, 1, has}, {b + splitNone, 8, 24, 5, has}, {b + splitBoth, 4, 4, 1, false},
			{b + splitIn, 4, 12, 1, false}, {b + splitBoth, 2, 8, 1, false}, {b + splitBoth, 16, 8, 1, false},
			{b + splitBoth, 4, 0, 1, false}, {b + splitBoth, 4, -8, 1, false}, {b + splitBoth, 8, 8, 0, false},
			{b + splitNone + 1, 4, 8, 1, false}, {255, 4, 8, 1, false},
		} {
			if got := intelSplitModeOK(c.mode, c.r, c.ido, c.l1); got != c.want {
				t.Errorf("AVX-512 %v: intelSplitModeOK(%d, %d, %d, %d) = %v", has, c.mode, c.r, c.ido, c.l1, got)
			}
		}
	}
}

// TestIntelSplitKernel: every mode, radix and direction selects its own
// kernel.
func TestIntelSplitKernel(t *testing.T) {
	seen := map[uintptr]string{}
	for _, r := range []int{4, 8} {
		for mode := uint8(1); mode <= intelSplit512Base+splitNone; mode++ {
			for _, inverse := range []bool{false, true} {
				fn := intelSplitKernel(mode, r, inverse)
				want := skSplitFwd[r][mode%5]
				switch {
				case mode > splitNone && inverse:
					want = intelSplit512Inv[r][mode-intelSplit512Base]
				case mode > splitNone:
					want = intelSplit512Fwd[r][mode-intelSplit512Base]
				case inverse:
					want = skSplitInv[r][mode]
				}
				p := reflect.ValueOf(fn).Pointer()
				if fn == nil || p != reflect.ValueOf(want).Pointer() {
					t.Fatalf("intelSplitKernel(%d, %d, %v): wrong kernel", mode, r, inverse)
				}
				name := strings.Join([]string{string(rune('0' + r)), string(rune('0' + mode)), map[bool]string{false: "f", true: "i"}[inverse]}, "/")
				if prev, dup := seen[p]; dup {
					t.Fatalf("%s and %s share a kernel", prev, name)
				}
				seen[p] = name
			}
		}
	}
}

// TestIntelSplit512PassLayoutRefuses: a 512-bit mode the pass cannot run, or
// any 512-bit mode without AVX-512, and slices shorter than the pass, are
// refused before any kernel runs, on every machine.
func TestIntelSplit512PassLayoutRefuses(t *testing.T) {
	defer func(h bool) { intelHasAVX512 = h }(intelHasAVX512)
	buf := make([]complex128, 4096)
	expect := func(name, prefix string, f func()) {
		t.Helper()
		defer func() {
			msg := ""
			switch v := recover().(type) {
			case string:
				msg = v
			case error:
				msg = v.Error()
			}
			if !strings.Contains(msg, prefix) {
				t.Errorf("%s: recovered %q, want a %q panic", name, msg, prefix)
			}
		}()
		f()
	}
	const refuse = "kernels: StockhamPassLayout:"
	const b = intelSplit512Base
	intelHasAVX512 = true
	for _, c := range []struct {
		mode       uint8
		r, ido, l1 int
	}{
		{b + splitBoth, 4, 4, 1}, {b + splitBoth, 4, 12, 1}, {b + splitOut, 16, 8, 1}, {b + splitIn, 2, 8, 1},
		{b + splitNone, 8, 8, 0}, {b + splitBoth, 4, -8, 1}, {b + splitNone + 1, 4, 8, 1},
	} {
		expect("mode", refuse, func() {
			StockhamPassLayout(c.mode, c.r, c.ido, c.l1, buf, buf, buf, false, true)
		})
	}
	short := buf[:127]
	expect("cc", "index out of range", func() { StockhamPassLayout(b+splitBoth, 4, 8, 4, short, buf, buf, false, true) })
	expect("ch", "index out of range", func() { StockhamPassLayout(b+splitBoth, 4, 8, 4, buf, short, buf, false, true) })
	expect("tw", "index out of range", func() { StockhamPassLayout(b+splitBoth, 8, 8, 1, buf, buf, buf[:55], false, true) })
	intelHasAVX512 = false
	expect("no AVX-512", refuse, func() { StockhamPassLayout(b+splitBoth, 4, 8, 1, buf, buf, buf, false, true) })
}

// TestClCompSplit256: where the 512-bit layout is on, a power of two takes
// its modes, a composite of at most clCompSplitMax points the 256-bit ones,
// and a longer composite, or any composite with ClCompSplit256 off, none; a
// mode's twiddles are its width's.
func TestClCompSplit256(t *testing.T) {
	defer func(a, h, c bool) { UseStockhamSplit512, intelHasAVX512, ClCompSplit256 = a, h, c }(UseStockhamSplit512, intelHasAVX512, ClCompSplit256)
	UseStockhamSplit512, intelHasAVX512, ClCompSplit256 = true, true, true
	plan := func(n int, f ...int) (r, ido, l1 []int) {
		l := 1
		for _, x := range f {
			r, ido, l1 = append(r, x), append(ido, n/(l*x)), append(l1, l)
			l *= x
		}
		return
	}
	none := func(m []uint8) bool { return !slices.ContainsFunc(m, func(x uint8) bool { return x != 0 }) }
	// 1920 = 3·5·4·8·4: a run of radix-4/8 passes after the odd ones.
	r, ido, l1 := plan(1920, 3, 5, 4, 8, 4)
	want := []uint8{0, 0, splitOut, splitIn, 0}
	if got := StockhamSplitModes(r, ido, l1); !slices.Equal(got, want) {
		t.Errorf("1920: modes %v, want %v", got, want)
	}
	r, ido, l1 = plan(1024, 4, 4, 8, 8)
	if got, want := StockhamSplitModes(r, ido, l1), intelSplitModes512(r, ido); !slices.Equal(got, want) || got[0] <= splitNone {
		t.Errorf("1024: modes %v, want the 512-bit %v", got, want)
	}
	r, ido, l1 = plan(clCompSplitMax*3/2, 3, 4, 4, 8, 8, 8, 4)
	if got := StockhamSplitModes(r, ido, l1); !none(got) {
		t.Errorf("%d: modes %v, want none", clCompSplitMax*3/2, got)
	}
	ClCompSplit256 = false
	r, ido, l1 = plan(1920, 3, 5, 4, 8, 4)
	if got := StockhamSplitModes(r, ido, l1); !none(got) {
		t.Errorf("1920 with ClCompSplit256 off: modes %v, want none", got)
	}
	if got := StockhamSplitModes(nil, nil, nil); len(got) != 0 {
		t.Errorf("no passes: %v", got)
	}
	const n = 96
	root := make([]complex128, n)
	for k := range root {
		root[k] = complex(float64(k), -float64(k)-0.5)
	}
	for _, mode := range []uint8{splitOut, splitBoth, splitIn, splitNone} {
		fwd, _ := StockhamSplitTwiddlesFor(mode, 4, 8, 3, root)
		if w, _ := clSplitTwiddles256(4, 8, 3, root); !slices.Equal(fwd, w) {
			t.Errorf("mode %d: twiddles not the 256-bit layout's", mode)
		}
		fwd, _ = StockhamSplitTwiddlesFor(mode+intelSplit512Base, 4, 8, 3, root)
		if w, _ := intelSplitTwiddles512(4, 8, 3, root); !slices.Equal(fwd, w) {
			t.Errorf("mode %d: twiddles not the 512-bit layout's", mode+intelSplit512Base)
		}
	}
}
