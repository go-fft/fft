package fft

import "github.com/go-fft/fft/internal/kernels"

// znTwoPassMax and znThreePassMax are the longest two- and three-pass
// Stockham transforms that run on a kernels.ZnPlan (Round 32): the buffers
// between their passes are in the kernel's own stack frame, not a sync.Pool
// round trip; 0 turns them off. They are per-architecture (route_*.go).
var znTwoPassMax, znThreePassMax = znTwoPassMaxDefault(), znThreePassMaxDefault()

// znPlanFor returns the kernels.ZnPlan that runs a plan of these stages, or
// nil: more than znTwoPassMax or znThreePassMax points for its pass count,
// or a plan the frame kernels cannot run.
func znPlanFor(n int, st []skStage) *kernels.ZnPlan {
	switch {
	case len(st) == 2 && n <= znTwoPassMax:
		return kernels.ZnPlanFor([]int{st[0].r, st[1].r}, []uint8{st[0].split, st[1].split}, st[0].twX, st[0].twXc, nil, nil)
	case len(st) == 3 && n <= znThreePassMax:
		return kernels.ZnPlanFor([]int{st[0].r, st[1].r, st[2].r}, []uint8{st[0].split, st[1].split, st[2].split},
			st[0].twX, st[0].twXc, st[1].twX, st[1].twXc)
	}
	return nil
}
