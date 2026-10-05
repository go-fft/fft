package fft

import "math"

// pow2StockhamMaxDefault routes every power of two to the Stockham engine on
// ppc64le. Its passes run scalar there, like the iterative pow2 kernel's
// butterflies; on 2026-09-29 the two tied below 65536. With the scratch buffer
// placed off dst's L1 sets (see offTheSets), POWER9 (cfarm29, idle) measured,
// pow2 kernel time ÷ Stockham time (2026-10-04):
//
//	n        4096  8192  16384  32768  65536  2^17  2^18  2^19  2^20
//	POWER9   1.20  1.24  1.34   1.30   1.37   1.15  1.22  1.44  1.81
func pow2StockhamMaxDefault() int { return math.MaxInt }

// r8MaxPow2Default and pow2OneRadix8Max factor a power of two the way arm64
// does: radix-4 passes, opening with one radix-8 pass for an odd exponent up
// to 8192, a lone radix-2 pass above. Timed on POWER9 against radix 8 as far
// as it goes (rule A) and radix 4 with a radix-2 pass (rule B) at every 2^e
// from 256 to 2^20 (2026-10-04), time over the best of the three, geometric
// mean: this rule 1.003 (worst 1.03), A up to 4096 then B — the rule ppc64le
// had — 1.056 (worst 1.25), A everywhere 1.234.
func r8MaxPow2Default() int { return 0 }

const pow2OneRadix8Max = 8192

// oddRadicesFirstDefault keeps the pocketfft order here (powers of two first,
// odd primes last): the odd-first order was measured on amd64 only (see
// route_amd64.go).
func oddRadicesFirstDefault() bool { return false }

// parMinChunkDefault is the 8192 elements chosen in 2026-09 (BENCHMARKS.md,
// Round 3), kept here: the change of Round 17 was measured on amd64 only.
func parMinChunkDefault() int { return 1 << 13 }
