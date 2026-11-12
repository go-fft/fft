package guardtest

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/go-fft/fft"
)

// TestImpulseSpectrumOnEveryEngine: the DFT of a unit impulse at j is the single
// complex exponential exp(-2πi·jk/n), so one transform checks every twiddle the
// engine used, against a cosine whose argument is reduced in int64. The lengths
// reach the sizes where a product of two indices exceeds 2^31 (46,341² and up),
// which on a 32-bit int overflowed in Rader's generator walk and modular power,
// in Bluestein's chirp, and in the mixed-radix twiddle indices. On a 64-bit int
// none of them is reachable, so this test is only red on the 386 lane.
func TestImpulseSpectrumOnEveryEngine(t *testing.T) {
	lengths := []struct {
		n      int
		engine string
	}{
		{65536, "power of two"},
		{59049, "3^10"},
		{78125, "5^7"},
		{117649, "7^6"},
		{100000, "2^5 5^5"},
		{50177, "Rader (prime)"},
		{65537, "Rader (prime, 2^16+1)"},
		{70001, "prime"},
		{65538, "Bluestein"},
		{46349, "prime above 46,341"},
	}
	for _, c := range lengths {
		n := c.n
		for _, j := range []int{1, n / 3, n - 2} {
			x := make([]complex128, n)
			x[j] = 1
			got := fft.NewPlan(n).FFT(make([]complex128, n), x)
			x32 := make([]complex64, n)
			x32[j] = 1
			got32 := fft.NewPlan32(n).FFT(make([]complex64, n), x32)
			var worst, worst32 float64
			for k := 0; k < n; k++ {
				a := 2 * math.Pi * float64(int64(j)*int64(k)%int64(n)) / float64(n)
				want := cmplx.Rect(1, -a)
				worst = math.Max(worst, cmplx.Abs(got[k]-want))
				worst32 = math.Max(worst32, cmplx.Abs(complex128(got32[k])-want))
			}
			if worst > 1e-9 {
				t.Errorf("n=%d (%s) impulse at %d: float64 error %g", n, c.engine, j, worst)
			}
			if worst32 > 1e-3 {
				t.Errorf("n=%d (%s) impulse at %d: float32 error %g", n, c.engine, j, worst32)
			}
		}
	}
}
