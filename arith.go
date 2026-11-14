package fft

// arith names the arithmetic a float64 plan was built for, and so the cache
// its plans live in. Today the default and the exact arithmetic are the same
// code and the same bits, but they are different promises: Options.Exact is
// the reference arithmetic whatever the default becomes (#121), and the
// default may later fuse multiply-adds, reorder its kernels or both. A plan
// of one arithmetic must never be returned for another, at any level: an
// Options.Exact transform of a Rader or Bluestein length builds its
// convolution on Exact plans, an N-D plan its axes on Exact plans, a real plan
// its half-length complex plan on an Exact plan. So each arithmetic has its
// own caches (planCache, planCacheFMA, planCacheExact, and the same for every
// plan kind) and the arithmetic travels with the plan (the ar fields).
//
// Only the float64 complex, real, 2-D real and N-D plans have more than one
// arithmetic. The float32 plans, and the DCT, DST and MDCT kernels, take no
// Options and have one: their caches are not partitioned.
type arith uint8

const (
	arithDefault arith = iota // what the package computes when nothing is asked
	arithFMA                  // Options.FMA: fused twiddle products where measured faster
	arithExact                // Options.Exact: the reference arithmetic
)

// arithFromFMA is the arithmetic of the bool-flag API (the FMA plans or the default).
func arithFromFMA(fma bool) arith {
	if fma {
		return arithFMA
	}
	return arithDefault
}

// fused reports whether plans of a fuse their twiddle products: only the FMA
// arithmetic does.
func (a arith) fused() bool { return a == arithFMA }

// arith resolves o to the arithmetic a transform called with it runs: Exact
// wins (Exact and FMA together are refused before this is asked), then FMA
// where the CPU has it, else the default.
func (o Options) arith() arith {
	switch {
	case o.Exact:
		return arithExact
	case o.fma():
		return arithFMA
	}
	return arithDefault
}
