package fft

// armrealRotateND runs a complex128 PlanN small enough for one goroutine as
// rotating passes (armrealRotate) instead of rows and column strips.
var armrealRotateND = true
