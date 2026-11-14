package fft

// arith names the arithmetic a float64 plan was built for, and so the cache
// its plans live in. The default arithmetic fuses multiply-adds where the CPU
// has FMA3 (amd64); Options.Exact is the reference arithmetic whatever the
// default becomes (#121); Options.NoFMA is the default without the fusing; and
// Options.FMA asks for the fused plans explicitly (the default where they run).
// A plan of one arithmetic must never be returned for another, at any level: an
// Options.Exact transform of a Rader or Bluestein length builds its
// convolution on Exact plans, an N-D plan its axes on Exact plans, a real plan
// its half-length complex plan on an Exact plan. So each arithmetic has its
// own caches (planCache, planCacheFMA, planCacheExact, planCachePlain, and the
// same for every plan kind) and the arithmetic travels with the plan (the ar
// fields).
//
// Only the float64 complex, real, 2-D real and N-D plans have more than one
// arithmetic. The float32 plans, and the DCT, DST and MDCT kernels, take no
// Options and have one: their caches are not partitioned.
type arith uint8

const (
	arithDefault arith = iota // what the package computes when nothing is asked: fused where FMA3 runs
	arithFMA                  // Options.FMA: fused twiddle products where measured faster
	arithExact                // Options.Exact: the reference arithmetic, v0.28.16's bits
	arithPlain                // Options.NoFMA: the default arithmetic, not fused
)

// arithFromFMA is the arithmetic of the bool-flag API (the FMA plans or the default).
func arithFromFMA(fma bool) arith {
	if fma {
		return arithFMA
	}
	return arithDefault
}

// defaultFuses is whether the default arithmetic fuses where the CPU has FMA3.
// It is on; a variable only so that the tests that compare an FMA plan with the
// unfused default (forceFMA) can switch the default's fusing off, and the
// tests of the default itself can switch it back on where no CPU has FMA3.
var defaultFuses = true

// fused reports whether plans of a fuse their twiddle products: the FMA
// arithmetic does, and so does the default where the CPU has FMA3
// (fmaAvailable: amd64 with the FMA kernels); Exact and NoFMA never do.
func (a arith) fused() bool {
	switch a {
	case arithFMA:
		return true
	case arithDefault:
		return fmaAvailable && defaultFuses
	}
	return false
}

// arith resolves o to the arithmetic a transform called with it runs: Exact
// wins (Exact with FMA or NoFMA is refused or redundant, see checkExact), then
// NoFMA, then FMA where the CPU has it, else the default.
func (o Options) arith() arith {
	switch {
	case o.Exact:
		return arithExact
	case o.NoFMA:
		return arithPlain
	case o.fma():
		return arithFMA
	}
	return arithDefault
}
