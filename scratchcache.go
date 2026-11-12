package fft

import (
	"sync"
	"sync/atomic"
)

// scratchCache hands a call its working memory: the buffer the last call left in
// the one-slot cache when it is there, a sync.Pool otherwise. A call takes the
// slot with one atomic swap and gives it back with one compare-and-swap, 3 ns
// against 7.5 ns for a sync.Pool Get and Put on an Apple M4 (go-fft/fft#132, a
// tenth of a 100 ns transform); concurrent callers find the slot empty and fall
// back to the pool, so a plan stays safe for concurrent use and keeps its zero
// allocations in steady state.
type scratchCache[T any] struct {
	slot atomic.Pointer[T]
	pool sync.Pool
}

// init sets the constructor of a buffer, called when both the slot and the pool
// are empty.
func (c *scratchCache[T]) init(newT func() *T) {
	c.pool.New = func() any { return newT() }
}

func (c *scratchCache[T]) get() *T {
	if s := c.slot.Swap(nil); s != nil {
		return s
	}
	return c.pool.Get().(*T)
}

func (c *scratchCache[T]) put(s *T) {
	if !c.slot.CompareAndSwap(nil, s) {
		c.pool.Put(s)
	}
}
