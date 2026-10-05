package fft

import (
	"math"
	"strconv"
)

// NextFastLen returns the smallest length >= target that this package
// transforms fast, the counterpart of scipy.fft.next_fast_len(target, real)
// (https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html):
// zero-pad a signal to it before transforming (FFTWith/RFFTWith with
// Options{N: NextFastLen(len(x), ...)}).
//
// What is fast depends on the library, so the rule differs from scipy's:
//
//   - complex (real == false): the 7-smooth lengths, 2^a·3^b·5^c·7^d. Every
//     such length runs entirely on the Stockham engine's specialized radix
//     2/3/4/5/7/8 passes. A factor 11 or 13 also stays on the mixed-radix path
//     but through its general O(p²) radix pass, and a larger prime factor goes
//     to Rader or Bluestein. Measured on Apple M4 Max (2026-10-05, ns per
//     point): 1001 = 7·11·13 cost 20 and 1100 = 2²·5²·11 cost 11, against 2.5
//     for 1008 and 1120; 2197 = 13³ cost 33 and 4067 = 7²·83 (Bluestein) 15,
//     against 2.7 for 4096. scipy, whose pocketfft has specialized radix-11
//     passes, accepts 11-smooth lengths here.
//   - real (real == true): the EVEN 7-smooth lengths. An even length packs into
//     a half-length complex transform; an odd one runs the full complex
//     transform: as a real transform 1125 = 3²·5³ cost 4.7 ns per point against
//     1.6 for 1134 = 2·3⁴·7.
//     scipy's real rule is 5-smooth, odd lengths included.
//
// Targets 0 and 1 are returned unchanged (such transforms are trivial). A
// negative target panics, as does a target above the largest qualifying
// length that fits in an int.
func NextFastLen(target int, real bool) int {
	if target < 0 {
		panic("fft: NextFastLen: negative target " + strconv.Itoa(target))
	}
	if target <= 1 {
		return target
	}
	if real {
		// The smallest even 7-smooth m >= target is twice the smallest
		// 7-smooth number >= ceil(target/2).
		h, ok := nextSmooth7(target/2 + target%2) // ceil without overflow
		if !ok || h > math.MaxInt/2 {
			panic("fft: NextFastLen: no fast length >= " + strconv.Itoa(target) + " fits in an int")
		}
		return 2 * h
	}
	m, ok := nextSmooth7(target)
	if !ok {
		panic("fft: NextFastLen: no fast length >= " + strconv.Itoa(target) + " fits in an int")
	}
	return m
}

// nextSmooth7 returns the smallest 2^a·3^b·5^c·7^d >= t (t >= 1), and false
// when none fits in an int. It enumerates every 3^b·5^c·7^d up to t and
// completes each with the least power of two that reaches t, so it costs
// O(log³ t) whatever the gap between consecutive smooth numbers — a linear
// scan would not finish near the top of the int range.
func nextSmooth7(t int) (int, bool) {
	best, found := 0, false
	for p7 := 1; ; p7 *= 7 {
		for p5 := p7; ; p5 *= 5 {
			for p3 := p5; ; p3 *= 3 {
				m := p3
				for m < t && m <= math.MaxInt/2 {
					m *= 2
				}
				if m >= t && (!found || m < best) {
					best, found = m, true
				}
				if p3 >= t || p3 > math.MaxInt/3 {
					break
				}
			}
			if p5 >= t || p5 > math.MaxInt/5 {
				break
			}
		}
		if p7 >= t || p7 > math.MaxInt/7 {
			break
		}
	}
	return best, found
}
