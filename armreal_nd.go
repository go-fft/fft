package fft

import "github.com/go-fft/fft/internal/kernels"

// armrealRotates reports whether p runs as rotating passes (armrealRotate):
// every axis longer than 1 is transformed, by a Stockham plan without the blocked schedule,
// at least two of them, and the plan is small enough to run on one
// goroutine.
func (p *PlanN) armrealRotates() bool {
	if !armrealRotateND {
		return false
	}
	axes := 0
	for ax, n := range p.shape {
		pl := p.axes[ax]
		if n == 1 {
			continue
		}
		if pl == nil || pl.sk == nil || pl.sk.cascT > 0 || parallelizeAxis(p.size/n, n) {
			return false
		}
		axes++
	}
	return axes >= 2
}

// armrealRotate writes the N-D transform of src into dst as one run of
// Stockham passes, last axis first. A Stockham pass treats its l1 blocks
// alike, so the lines of the last axis, contiguous in src, are l1·(other
// elements) blocks of one pass sequence: each line gets exactly the passes of
// its own 1-D plan, and the sequence ends with output f of line c at
// f·(other elements) + c. That moves the transformed axis to the front and
// makes the axis before it contiguous, which the next sequence transforms the
// same way; after every axis the array is back in its natural order. The
// passes alternate between dst and scr (p.size elements each).
func (p *PlanN) armrealRotate(dst, src, scr []complex128, inverse bool) {
	t := len(p.rot)
	in := src
	if t%2 == 1 && &dst[0] == &src[0] {
		copy(scr, src)
		in = scr
	}
	for k, rp := range p.rot {
		out := scr
		if (t-1-k)%2 == 0 {
			out = dst
		}
		rp.st.armrealPassL(out, in, rp.l1, inverse)
		in = out
	}
}

// armrealPass is one pass of the rotating sequence: a stage of an axis plan
// and the number of blocks it runs over.
type armrealPass struct {
	st *skStage
	l1 int
}

// armrealPasses lists the rotating passes of p, last axis first.
func (p *PlanN) armrealPasses() []armrealPass {
	var rot []armrealPass
	for ax := len(p.shape) - 1; ax >= 0; ax-- {
		if pl := p.axes[ax]; pl != nil {
			for k := range pl.sk.stages {
				st := &pl.sk.stages[k]
				rot = append(rot, armrealPass{st, st.l1 * (p.size / p.shape[ax])})
			}
		}
	}
	return rot
}

// armrealPassL is pass with l1 blocks instead of st.l1.
func (st *skStage) armrealPassL(ch, cc []complex128, l1 int, inverse bool) {
	twX := st.twX
	if inverse {
		twX = st.twXc
	}
	if kernels.StockhamPassLayout(st.split, st.r, st.ido, l1, cc, ch, twX, inverse, st.wide) {
		return
	}
	st.passScalarL(ch, cc, l1, inverse)
}
