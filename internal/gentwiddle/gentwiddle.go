// Package gentwiddle is the correctly rounded twiddle arithmetic the code
// generators share (internal/gendct4, internal/genfused256): cos and sin of
// 2π·num/den at 400 bits, rounded once to float64.
package gentwiddle

import "math/big"

// bigPi is π to 400 bits (Machin: 16·atan(1/5) - 4·atan(1/239)).
func bigPi() *big.Float {
	const prec = 400
	atanInv := func(k int64) *big.Float {
		x := new(big.Float).SetPrec(prec).Quo(big.NewFloat(1).SetPrec(prec), new(big.Float).SetPrec(prec).SetInt64(k))
		x2 := new(big.Float).SetPrec(prec).Mul(x, x)
		sum := new(big.Float).SetPrec(prec)
		term := new(big.Float).SetPrec(prec).Set(x)
		for i := int64(0); i < 400; i++ {
			t := new(big.Float).SetPrec(prec).Quo(term, new(big.Float).SetPrec(prec).SetInt64(2*i+1))
			if i%2 == 0 {
				sum.Add(sum, t)
			} else {
				sum.Sub(sum, t)
			}
			term.Mul(term, x2)
		}
		return sum
	}
	a := new(big.Float).SetPrec(prec).Mul(big.NewFloat(16).SetPrec(prec), atanInv(5))
	b := new(big.Float).SetPrec(prec).Mul(big.NewFloat(4).SetPrec(prec), atanInv(239))
	return a.Sub(a, b)
}

// CosSin is cos and sin of 2π·num/den, correctly rounded to float64: the
// Taylor series at 400 bits, so the only rounding is the last. (math.Sincos of a
// float64 2πk/n carries the rounding of the angle into the constant.)
func CosSin(num, den int) (c, s float64) {
	cs, sn := bigCosSin(num, den)
	c, _ = cs.Float64()
	s, _ = sn.Float64()
	return c, s
}

// CosSin32 is cos and sin of 2π·num/den, correctly rounded to float32 from the
// 400-bit value in one rounding: never through float64, which would round twice
// and can land one float32 ulp away from the correctly rounded value.
func CosSin32(num, den int) (c, s float32) {
	cs, sn := bigCosSin(num, den)
	c, _ = cs.Float32()
	s, _ = sn.Float32()
	return c, s
}

// bigCosSin is cos and sin of 2π·num/den at 400 bits.
func bigCosSin(num, den int) (cs, sn *big.Float) {
	const prec = 400
	x := new(big.Float).SetPrec(prec).Mul(big.NewFloat(2).SetPrec(prec), bigPi())
	x.Mul(x, new(big.Float).SetPrec(prec).SetInt64(int64(num)))
	x.Quo(x, new(big.Float).SetPrec(prec).SetInt64(int64(den)))
	x2 := new(big.Float).SetPrec(prec).Mul(x, x)
	cs = new(big.Float).SetPrec(prec)
	sn = new(big.Float).SetPrec(prec)
	tc := new(big.Float).SetPrec(prec).SetInt64(1) // x^(2k)/(2k)!
	ts := new(big.Float).SetPrec(prec).Set(x)      // x^(2k+1)/(2k+1)!
	for k := int64(0); k < 120; k++ {
		if k%2 == 0 {
			cs.Add(cs, tc)
			sn.Add(sn, ts)
		} else {
			cs.Sub(cs, tc)
			sn.Sub(sn, ts)
		}
		tc.Mul(tc, x2)
		tc.Quo(tc, new(big.Float).SetPrec(prec).SetInt64((2*k+1)*(2*k+2)))
		ts.Mul(ts, x2)
		ts.Quo(ts, new(big.Float).SetPrec(prec).SetInt64((2*k+2)*(2*k+3)))
	}
	return cs, sn
}
