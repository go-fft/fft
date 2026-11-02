package fft

import (
	"math"
	"strconv"
)

// checkAlloc panics with the package's message, before anything is allocated,
// when n elements of elemBytes bytes each cannot be one allocation because
// their byte count overflows int: the point where make would otherwise fail
// with the runtime's "makeslice: len out of range" (or, where the size
// arithmetic itself wraps, index a wrapped length). The bound is exact for the
// allocation it names — math.MaxInt/elemBytes elements, 2^59 of 16 bytes on a
// 64-bit target and 2^27 on a 32-bit one — and it is the only length cap: below
// it a length allocates in proportion to itself, as documented in SECURITY.md,
// and the runtime refuses what the address space cannot hold.
func checkAlloc(what string, n, elemBytes int) {
	if n > math.MaxInt/elemBytes {
		panic("fft: " + what + ": length " + strconv.Itoa(n) + " is too large: more than " +
			strconv.Itoa(math.MaxInt/elemBytes) + " elements of " + strconv.Itoa(elemBytes) + " bytes")
	}
}
