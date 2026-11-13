package fft

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestRoutingStateIsolated is the rule behind isolatePlans (#139), enforced:
// a test that assigns package-level state (a routing table, a kernel switch,
// kernels.Use...) and then calls a function that caches its plan leaves a plan
// built on the overridden route in a cache, and whichever test runs next, in
// any order, runs on it. So every function in a _test.go file that assigns
// package state must call isolatePlans (or forceFMA, which clears the caches
// itself), or be a helper whose every caller does. It fails, naming the
// function, when one does not: init functions and TestMain cannot call
// isolatePlans, so they cannot assign such state at all.
//
// Package state is every package-level variable the non-test files declare,
// except the plan caches and their locks, plus the kernels.Use... switches.
// stateWritten names the package state an assignment target writes, or "": a
// package variable (a name another file declares, so Obj is nil), an element
// of one (passWeight[8] = 9.9, splitTable[n] = ...), a field of one, or any
// kernels.X variable and element of it.
func stateWritten(e ast.Expr, pkgVars map[string]bool) string {
	switch e := e.(type) {
	case *ast.Ident:
		if e.Obj == nil && pkgVars[e.Name] {
			return e.Name
		}
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok && id.Name == "kernels" {
			return "kernels." + e.Sel.Name
		}
		if inner := stateWritten(e.X, pkgVars); inner != "" {
			return inner + "." + e.Sel.Name
		}
	case *ast.IndexExpr:
		if inner := stateWritten(e.X, pkgVars); inner != "" {
			return inner + "[...]"
		}
	case *ast.StarExpr:
		return stateWritten(e.X, pkgVars)
	case *ast.ParenExpr:
		return stateWritten(e.X, pkgVars)
	}
	return ""
}

func TestRoutingStateIsolated(t *testing.T) {
	fset := token.NewFileSet()
	pkgVars := map[string]bool{}
	nonTest, _ := filepath.Glob("*.go")
	var testFiles []string
	for _, f := range nonTest {
		if strings.HasSuffix(f, "_test.go") {
			testFiles = append(testFiles, f)
			continue
		}
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			continue // a file of another build configuration that does not parse alone
		}
		for _, d := range af.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.VAR {
				continue
			}
			for _, s := range gd.Specs {
				for _, n := range s.(*ast.ValueSpec).Names {
					if !strings.HasSuffix(n.Name, "Cache") && !strings.HasSuffix(n.Name, "CacheFMA") && !strings.HasSuffix(n.Name, "Mu") && n.Name != "_" {
						pkgVars[n.Name] = true
					}
				}
			}
		}
	}
	type fn struct {
		file, name string
		assigns    []string
		calls      map[string]bool
		isolated   bool
	}
	funcs := map[string]*fn{}
	for _, f := range testFiles {
		af, err := parser.ParseFile(fset, f, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range af.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Body == nil || fd.Recv != nil {
				continue
			}
			x := &fn{file: f, name: fd.Name.Name, calls: map[string]bool{}}
			ast.Inspect(fd.Body, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.IncDecStmt:
					if name := stateWritten(n.X, pkgVars); name != "" {
						x.assigns = append(x.assigns, name)
					}
				case *ast.AssignStmt:
					for _, l := range n.Lhs {
						if name := stateWritten(l, pkgVars); name != "" {
							x.assigns = append(x.assigns, name)
						}
					}
				case *ast.CallExpr:
					if id, ok := n.Fun.(*ast.Ident); ok {
						x.calls[id.Name] = true
						if id.Name == "isolatePlans" || id.Name == "forceFMA" {
							x.isolated = true
						}
					}
				}
				return true
			})
			funcs[x.name] = x
		}
	}
	// The helpers of the isolation itself.
	for _, n := range []string{"forceFMA", "isolatePlans", "forgetAllPlans", "forgetFMAPlans"} {
		if f := funcs[n]; f != nil {
			f.isolated = true
		}
	}
	isTest := func(name string) bool {
		for _, p := range []string{"Test", "Benchmark", "Fuzz", "Example"} {
			if strings.HasPrefix(name, p) {
				return true
			}
		}
		return name == "init" || name == "main"
	}
	// A helper is covered when it has callers and every one is covered.
	var covered func(name string, seen map[string]bool) bool
	covered = func(name string, seen map[string]bool) bool {
		f := funcs[name]
		if f == nil {
			return false
		}
		if f.isolated {
			return true
		}
		if isTest(name) || seen[name] {
			return false
		}
		seen[name] = true
		callers := 0
		for _, g := range funcs {
			if g.calls[name] {
				callers++
				if !covered(g.name, seen) {
					return false
				}
			}
		}
		return callers > 0
	}
	var bad []string
	for _, f := range funcs {
		if len(f.assigns) > 0 && !covered(f.name, map[string]bool{}) {
			seen := map[string]bool{}
			var u []string
			for _, a := range f.assigns {
				if !seen[a] {
					seen[a] = true
					u = append(u, a)
				}
			}
			bad = append(bad, f.file+": "+f.name+" assigns "+strings.Join(u, ", "))
		}
	}
	sort.Strings(bad)
	for _, b := range bad {
		t.Errorf("%s, and neither it nor every caller calls isolatePlans (see plancaches_test.go)", b)
	}
	if len(pkgVars) < 20 || len(funcs) < 100 {
		t.Fatalf("the audit saw %d package variables and %d test functions: it is not reading the package", len(pkgVars), len(funcs))
	}
}
