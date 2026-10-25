package fft

import "github.com/go-fft/fft/internal/kernels"

// znTwoPassMax and znThreePassMax are the longest two- and three-pass
// Stockham transforms that run on a kernels.ZnPlan (Round 32): the buffers
// between their passes are in the kernel's own stack frame, not a sync.Pool
// round trip; 0 turns them off. They are per-architecture (route_*.go).
var znTwoPassMax, znThreePassMax = znTwoPassMaxDefault(), znThreePassMaxDefault()

// znBinder builds a frame plan from its passes' radices, layouts and first
// two twiddle tables (kernels.ZnPlanForWide's signature).
type znBinder func(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128, wide bool) *kernels.ZnPlan

// znPlanFor returns the kernels.ZnPlan that runs a plan of these stages, or
// nil: more than znTwoPassMax or znThreePassMax points for its pass count,
// or a plan the frame kernels cannot run. An FMA plan (fma) binds through
// znFMABind, per architecture (on amd64: the FMA twins of the kernels of its
// stages that fuse).
func znPlanFor(n int, st []skStage, fma bool) *kernels.ZnPlan {
	var bind znBinder = kernels.ZnPlanForWide
	if fma {
		bind = znFMABind(st)
	}
	switch {
	case len(st) == 2 && n <= znTwoPassMax:
		return bind([]int{st[0].r, st[1].r}, []uint8{st[0].split, st[1].split}, st[0].twX, st[0].twXc, nil, nil, st[0].wide)
	case len(st) == 3 && n <= znThreePassMax:
		return bind([]int{st[0].r, st[1].r, st[2].r}, []uint8{st[0].split, st[1].split, st[2].split},
			st[0].twX, st[0].twXc, st[1].twX, st[1].twXc, st[0].wide)
	}
	return nil
}
