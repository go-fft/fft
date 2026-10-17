package fft

// fmaSplitDefault: every amd64 split kernel has an FMA twin (fmasplit*.s).
func fmaSplitDefault() bool { return true }
