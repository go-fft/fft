/* fftw_pp <n> <iters> <wisdomfile> <m|w>: out-of-place 1-D complex forward.
 * m: plan with FFTW_MEASURE and write wisdom; w: import wisdom, plan with
 * FFTW_WISDOM_ONLY (no search, so perf-stat counters of two runs subtract).
 * FFTW_PRINT_PLAN=1 prints fftw_fprint_plan to stderr. Measurement shim only. */
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 5) return 2;
    long n = atol(argv[1]), iters = atol(argv[2]);
    const char *wf = argv[3];
    int measure = argv[4][0] == 'm';
    fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p;
    if (measure) {
        p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
        fftw_export_wisdom_to_filename(wf);
    } else {
        if (!fftw_import_wisdom_from_filename(wf)) return 3;
        p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE | FFTW_WISDOM_ONLY);
        if (!p) return 4;
    }
    if (getenv("FFTW_PRINT_PLAN")) {
        fftw_fprint_plan(p, stderr);
        fprintf(stderr, "\n");
    }
    for (long i = 0; i < n; i++) {
        in[i][0] = ((i * 7 + 1) % 13) * 0.1;
        in[i][1] = ((i * 3 + 2) % 11) * 0.1;
    }
    for (int i = 0; i < 2000; i++) fftw_execute(p);
    double t0 = now_s();
    for (long i = 0; i < iters; i++) fftw_execute(p);
    double dt = now_s() - t0;
    printf("PPRESULT n=%ld pass=-1 iters=%ld ns/op=%.2f\n", n, iters, dt / iters * 1e9);
    return 0;
}
