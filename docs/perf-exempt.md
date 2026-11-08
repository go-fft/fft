# Public functions without a performance case

Read by `refbench` in the CI mode (`cd bench && go test -run TestPerf .`): every
public function of the module has a case in `bench/specs_test.go` or is named
here with its reason. The method and the table are in [perf.md](perf.md).

## Plan constructors: plan cost is excluded on both sides

A plan row builds its plan before the timed loop, and scipy.fft's plan cache is
warm after the calibration call, so neither side times planning. The constructors
therefore have no row of their own; scipy.fft has no constructor to compare with
(pocketfft builds a plan inside the first call for a length).

- `NewPlan`: plan construction, excluded on both sides.
- `NewPlanWith`: as `NewPlan`.
- `NewPlan32`: as `NewPlan`.
- `NewPlanN`: as `NewPlan`.
- `NewPlanNWith`: as `NewPlan`.
- `NewPlanN32`: as `NewPlan`.
- `NewRealPlan`: as `NewPlan`.
- `NewRealPlanWith`: as `NewPlan`.
- `NewRealPlan32`: as `NewPlan`.
- `NewRealPlan2`: as `NewPlan`.
- `NewRealPlan2With`: as `NewPlan`.
- `NewRealPlan2_32`: as `NewPlan`.
- `NewDCTPlan`: as `NewPlan`.
- `NewDCTPlan32`: as `NewPlan`.
- `NewDSTPlan`: as `NewPlan`.
- `NewDSTPlan32`: as `NewPlan`.

## Accessors

- `Plan.Len`: returns a stored field.
- `Plan32.Len`: as `Plan.Len`.
- `PlanN.Len`: as `Plan.Len`.
- `PlanN.Shape`: copies a short slice of lengths; no reference.
- `PlanN32.Len`: as `Plan.Len`.
- `PlanN32.Shape`: as `PlanN.Shape`.
- `RealPlan.Len`: as `Plan.Len`.
- `RealPlan32.Len`: as `Plan.Len`.
- `RealPlan2.SpectrumLen`: as `Plan.Len`.
- `RealPlan2_32.SpectrumLen`: as `Plan.Len`.
- `DCTPlan.Len`: as `Plan.Len`.
- `DCTPlan.Type`: as `Plan.Len`.
- `DCTPlan32.Len`: as `Plan.Len`.
- `DCTPlan32.Type`: as `Plan.Len`.
- `DSTPlan.Len`: as `Plan.Len`.
- `DSTPlan.Type`: as `Plan.Len`.
- `DSTPlan32.Len`: as `Plan.Len`.
- `DSTPlan32.Type`: as `Plan.Len`.
