//go:build !amd64

package kernels

// ZnTwoPassMax and ZnThreePassMax are 0 here: there is no frame kernel
// (Round 32 built them for amd64 only).
const (
	ZnTwoPassMax   = 0
	ZnThreePassMax = 0
)

// ZnPlan is amd64's frame-kernel transform; here it never runs.
type ZnPlan struct{}

// ZnPlanFor returns nil: the fft package runs the passes on its scratch
// buffer.
func ZnPlanFor(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128) *ZnPlan { return nil }

// Run reports false: there is no kernel to run.
func (z *ZnPlan) Run(cc, ch []complex128, inverse bool) bool { return false }
