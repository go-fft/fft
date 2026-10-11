package fft_test

// The measurement behind BENCHMARKS.md's Round 36 between binaries: Round
// 32's placement harness (round32_place_test.go) with 2-D rows, exported API
// only, so the same file builds against the release it is compared with.

import (
	"flag"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-fft/fft"
)

var (
	zmPlace      = flag.String("zm.place", "", "rounds,ms,placements: run TestZmPlace")
	zmPlaceSizes = flag.String("zm.sizes", "C10007,D64,D128,C6144,C10240,C12288,C1024,C1000,R4096", "TestZmPlace's rows: C<n> complex, R<n> RFFT, D<n> n×n FFT2 on a PlanN")
)

// TestZmPlace is TestZnPlace with D rows: each row's plan built on several
// heap placements, every (row, placement) timed in rotated rounds.
func TestZmPlace(t *testing.T) {
	if *zmPlace == "" {
		t.Skip("set -zm.place=rounds,ms,placements")
	}
	var rounds, ms, places int
	if _, err := fmt.Sscanf(*zmPlace, "%d,%d,%d", &rounds, &ms, &places); err != nil {
		t.Fatal(err)
	}
	type cell struct {
		row string
		fn  func()
	}
	var cells []cell
	rows := strings.Split(*zmPlaceSizes, ",")
	for j := range places {
		for _, row := range rows {
			znPad(j)
			n, err := strconv.Atoi(row[1:])
			if err != nil {
				t.Fatal(err)
			}
			var fn func()
			switch row[0] {
			case 'C', 'D':
				m := n
				if row[0] == 'D' {
					m = n * n
				}
				src := make([]complex128, m)
				for i := range src {
					src[i] = complex(float64((i*7+1)%13)*0.1, float64(i*((i*3+2)%11))*0.1)
				}
				dst := make([]complex128, m)
				if row[0] == 'C' {
					p := fft.NewPlan(n)
					fn = func() { p.FFT(dst, src) }
				} else {
					p := fft.NewPlanN(n, n)
					fn = func() { p.FFT(dst, src) }
				}
			case 'R':
				p := fft.NewRealPlan(n)
				src := make([]float64, n)
				for i := range src {
					src[i] = float64((i*7+1)%13) * 0.1
				}
				dst := make([]complex128, n/2+1)
				fn = func() { p.RFFT(dst, src) }
			default:
				t.Fatalf("row %q", row)
			}
			fn()
			cells = append(cells, cell{row, fn})
		}
	}
	iters := map[string]int{}
	for _, c := range cells {
		if _, ok := iters[c.row]; ok {
			continue
		}
		k := 1
		for {
			t0 := time.Now()
			for range k {
				c.fn()
			}
			if time.Since(t0) > time.Duration(ms)*time.Millisecond/4 {
				k *= 4
				break
			}
			k *= 2
		}
		iters[c.row] = k
	}
	times := make([][]float64, len(cells))
	for round := range rounds {
		for q := range cells {
			v := (q + round*7) % len(cells)
			k := iters[cells[v].row]
			t0 := time.Now()
			for range k {
				cells[v].fn()
			}
			times[v] = append(times[v], float64(time.Since(t0).Nanoseconds())/float64(k))
		}
	}
	per := map[string][]float64{}
	for i, c := range cells {
		per[c.row] = append(per[c.row], znMedian(times[i]))
	}
	for _, row := range rows {
		x := per[row]
		fmt.Printf("ROW %-7s median %10.1f  min %10.1f  max %10.1f  (%d placements)\n", row, znMedian(x), slices.Min(x), slices.Max(x), len(x))
	}
}
