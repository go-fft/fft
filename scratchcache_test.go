package fft

import (
	"sync"
	"testing"
)

// TestScratchCacheSlotThenPool: the first buffer a call returns goes to the slot,
// a second goes to the pool, the next call takes the slot, then the pool, then
// builds a new one.
func TestScratchCacheSlotThenPool(t *testing.T) {
	built := 0
	var c scratchCache[[]int]
	c.init(func() *[]int { built++; s := make([]int, 4); return &s })

	a := c.get() // slot and pool empty: built
	b := c.get() // built again
	if built != 2 || a == b {
		t.Fatalf("built %d buffers, want 2 distinct", built)
	}
	c.put(a) // slot
	if c.slot.Load() != a {
		t.Fatal("the first buffer returned did not go to the slot")
	}
	c.put(b) // slot is taken: pool
	if c.slot.Load() != a {
		t.Fatal("the slot was overwritten")
	}
	if got := c.get(); got != a {
		t.Fatal("get did not take the slot first")
	}
	if c.slot.Load() != nil {
		t.Fatal("get left the buffer in the slot")
	}
	// the pool may keep or drop b (sync.Pool is allowed to drop): either way get
	// returns a usable buffer, building one only if the pool dropped it
	if got := c.get(); got == nil || len(*got) != 4 {
		t.Fatal("get returned no usable buffer")
	}
	if built > 3 {
		t.Fatalf("built %d buffers, want at most 3", built)
	}
}

// TestCheckR2RRefusesHugeLengths: the allocation guard still names the transform
// now that its name is built only when the guard fires.
func TestCheckR2RRefusesHugeLengths(t *testing.T) {
	const huge = int(^uint(0) >> 1)
	mustPanicWith(t, "DCT-IV", "fft: DCT-IV: length ", func() { checkR2R(true, 4, huge, NormBackward) })
	mustPanicWith(t, "DST-I", "fft: DST-I: length ", func() { checkR2R(false, 1, huge, NormBackward) })
	checkR2R(true, 4, 1<<20, NormBackward) // a length below the bound passes
}

// TestPlansAreSafeForConcurrentUse: many goroutines on one plan, each with its
// own slices, give the single-threaded result, on both MDCT routes, both
// precisions and the DCT-IV. Run under -race in CI: the one-slot scratch cache must
// never hand the same buffer to two calls.
func TestPlansAreSafeForConcurrentUse(t *testing.T) {
	const goroutines, rounds = 8, 200
	for _, n := range []int{8, 12, 128} { // N/2 even, odd, even
		p, p32 := NewMDCTPlan(n), NewMDCTPlan32(n)
		x := accRealF64(n, 21)
		c := accRealF64(n/2, 22)
		x32, c32 := accRealF32(n, 21), accRealF32(n/2, 22)
		wantF := p.MDCT(make([]float64, n/2), x, 1)
		wantI := p.IMDCT(make([]float64, n), c, 1)
		wantF32 := p32.MDCT(make([]float32, n/2), x32, 1)
		wantI32 := p32.IMDCT(make([]float32, n), c32, 1)
		d := NewDCTPlan(n/2, 4)
		wantD := d.DCT(make([]float64, n/2), c, NormBackward)
		var wg sync.WaitGroup
		errs := make(chan string, goroutines)
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				f, i := make([]float64, n/2), make([]float64, n)
				f32, i32 := make([]float32, n/2), make([]float32, n)
				dd := make([]float64, n/2)
				for r := 0; r < rounds; r++ {
					p.MDCT(f, x, 1)
					p.IMDCT(i, c, 1)
					p32.MDCT(f32, x32, 1)
					p32.IMDCT(i32, c32, 1)
					d.DCT(dd, c, NormBackward)
					if !equalF(f, wantF) || !equalF(i, wantI) || !equalF(dd, wantD) {
						errs <- "float64 result differs under concurrency"
						return
					}
					for k := range f32 {
						if f32[k] != wantF32[k] {
							errs <- "float32 MDCT differs under concurrency"
							return
						}
					}
					for k := range i32 {
						if i32[k] != wantI32[k] {
							errs <- "float32 IMDCT differs under concurrency"
							return
						}
					}
				}
			}()
		}
		wg.Wait()
		close(errs)
		for e := range errs {
			t.Errorf("N=%d: %s", n, e)
		}
	}
}
