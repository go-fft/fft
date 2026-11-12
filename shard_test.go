package fft

import (
	"flag"
	"fmt"
	"hash/fnv"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// Deterministic sharding of this package's tests, for the CI lanes that are too
// slow in one piece (the qemu architectures, the Windows race lane).
//
//	GOFFT_SHARD=i/N go test .
//
// runs the tests whose name hashes (FNV-1a, 32 bits) to i modulo N, with i from 0
// to N-1; the N shards together run every top-level test, Fuzz seed corpus and
// Example exactly once (TestShardsPartitionEveryTest). With the variable unset
// every test runs, so a local run does not change. A malformed value, or a -run
// given at the same time, is refused with exit status 2 rather than ignored: a
// lane that quietly ran everything, or nothing, would be a green tick for the
// wrong reason.
//
// The names come from the compiled test binary (the unexported tests, fuzzTargets
// and examples fields of *testing.M, read by reflection), so build constraints are
// already applied and nothing is parsed from source. If a Go release renames
// those fields, TestMain refuses to run instead of guessing.

// shardOf is the shard, 0 <= shardOf < n, a test name belongs to.
func shardOf(name string, n int) int {
	h := fnv.New32a()
	h.Write([]byte(name))
	return int(h.Sum32() % uint32(n))
}

// parseShard reads "i/N".
func parseShard(s string) (i, n int, err error) {
	a, b, ok := strings.Cut(s, "/")
	if ok {
		i, err = strconv.Atoi(a)
		if err == nil {
			n, err = strconv.Atoi(b)
		}
	}
	if !ok || err != nil || n < 1 || i < 0 || i >= n {
		return 0, 0, fmt.Errorf("GOFFT_SHARD=%q: want i/N with 0 <= i < N", s)
	}
	return i, n, nil
}

// shardRegex is the -run expression selecting the names of shard i of n. A shard
// with no name gets an expression no test name matches.
func shardRegex(names []string, i, n int) string {
	var sel []string
	for _, name := range names {
		if shardOf(name, n) == i {
			sel = append(sel, regexp.QuoteMeta(name))
		}
	}
	if len(sel) == 0 {
		return "^$"
	}
	return "^(" + strings.Join(sel, "|") + ")$"
}

// testNames lists the Test, Fuzz and Example functions compiled into the binary.
func testNames(m *testing.M) ([]string, error) {
	v := reflect.ValueOf(m).Elem()
	var names []string
	for _, field := range []string{"tests", "fuzzTargets", "examples"} {
		f := v.FieldByName(field)
		if !f.IsValid() || f.Kind() != reflect.Slice {
			return nil, fmt.Errorf("testing.M has no %s slice: this Go release changed its layout", field)
		}
		for j := 0; j < f.Len(); j++ {
			n := f.Index(j).FieldByName("Name")
			if !n.IsValid() || n.Kind() != reflect.String {
				return nil, fmt.Errorf("testing.M.%s elements have no Name field", field)
			}
			names = append(names, n.String())
		}
	}
	sort.Strings(names)
	return names, nil
}

var testMainNames []string // set by TestMain, read by the partition test

func TestMain(m *testing.M) {
	flag.Parse()
	names, err := testNames(m)
	if err != nil {
		fmt.Fprintln(os.Stderr, "shard:", err)
		os.Exit(2)
	}
	testMainNames = names
	if spec := os.Getenv("GOFFT_SHARD"); spec != "" {
		i, n, err := parseShard(spec)
		if err == nil && flag.Lookup("test.run").Value.String() != "" {
			err = fmt.Errorf("GOFFT_SHARD cannot be combined with -run")
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "shard:", err)
			os.Exit(2)
		}
		if err := flag.Set("test.run", shardRegex(names, i, n)); err != nil {
			fmt.Fprintln(os.Stderr, "shard:", err)
			os.Exit(2)
		}
		fmt.Printf("shard %d/%d: %d of %d tests\n", i, n, countMatching(names, i, n), len(names))
	}
	os.Exit(m.Run())
}

func countMatching(names []string, i, n int) int {
	c := 0
	for _, name := range names {
		if shardOf(name, n) == i {
			c++
		}
	}
	return c
}

// TestShardsPartitionEveryTest: for every N the CI uses (and 1 to 8), each test
// is selected by exactly one of the N shard expressions, and the shard sizes add up
// to the total: none lost, none run twice.
func TestShardsPartitionEveryTest(t *testing.T) {
	names := testMainNames
	if len(names) < 100 {
		t.Fatalf("only %d test names found: the listing is broken", len(names))
	}
	for n := 1; n <= 8; n++ {
		total := 0
		hits := make(map[string]int, len(names))
		for i := 0; i < n; i++ {
			re := regexp.MustCompile(shardRegex(names, i, n))
			size := 0
			for _, name := range names {
				if re.MatchString(name) {
					hits[name]++
					size++
				}
			}
			total += size
			if size != countMatching(names, i, n) {
				t.Errorf("N=%d shard %d: the expression selects %d tests, shardOf %d", n, i, size, countMatching(names, i, n))
			}
		}
		if total != len(names) {
			t.Errorf("N=%d: the shards select %d tests in all, there are %d", n, total, len(names))
		}
		for _, name := range names {
			if hits[name] != 1 {
				t.Errorf("N=%d: %s is selected by %d shards, want 1", n, name, hits[name])
			}
		}
	}
}

// TestShardRegexIsExactAndQuoted: a name that is a prefix of another, and a name
// with regexp metacharacters, are selected alone.
func TestShardRegexIsExactAndQuoted(t *testing.T) {
	names := []string{"TestA", "TestAB", "Example.x", "TestC(1)"}
	for i := 0; i < 2; i++ {
		re := regexp.MustCompile(shardRegex(names, i, 2))
		for _, name := range names {
			if got, want := re.MatchString(name), shardOf(name, 2) == i; got != want {
				t.Errorf("shard %d of 2, %q: matches %v, want %v", i, name, got, want)
			}
		}
		if re.MatchString("Test") || re.MatchString("TestABC") || re.MatchString("ExampleXx") {
			t.Errorf("shard %d of 2 matches a name that is not listed", i)
		}
	}
	if got := shardRegex(nil, 0, 3); got != "^$" || regexp.MustCompile(got).MatchString("TestA") {
		t.Errorf("an empty shard must match nothing, got %q", got)
	}
}

func TestParseShard(t *testing.T) {
	for _, s := range []string{"0/1", "2/3", "7/8"} {
		i, n, err := parseShard(s)
		if err != nil || fmt.Sprintf("%d/%d", i, n) != s {
			t.Errorf("parseShard(%q) = %d, %d, %v", s, i, n, err)
		}
	}
	for _, s := range []string{"", "1", "3/3", "-1/3", "a/3", "1/b", "1/0", "1/2/3", "/"} {
		if _, _, err := parseShard(s); err == nil {
			t.Errorf("parseShard(%q) succeeded", s)
		}
	}
}
