/* Native-FFTW reference timer for the go-fft parity report.
 *
 * This is the authoritative FFTW gold-standard number: it links the native
 * Homebrew FFTW (arm64, NEON-tuned) directly from C — not the generic pyfftw
 * pip wheel, whose bundled FFTW plans a poor 2-D transform on Apple Silicon.
 * Single-threaded, plan built once with FFTW_MEASURE and reused; the steady
 * state transform is timed (plan/setup cost reported separately).
 *
 * Inputs are bit-identical to the Go/Python harnesses:
 *     complex:  ((i*7+1)%13)*0.1 + i*((i*3+2)%11)*0.1
 *     real:     ((i*7+1)%13)*0.1
 *
 * Build (see run.sh):
 *     cc -O3 -I$(brew --prefix fftw)/include fftw_bench.c \
 *        -L$(brew --prefix fftw)/lib -lfftw3 -lm -o fftw_bench
 *
 * Output: JSON on stdout, human table on stderr.
 */
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* best-of ns/op, each batch auto-scaled to ~target seconds */
static double bench_ns(void (*run)(void *), void *ctx, double target) {
    run(ctx); run(ctx);
    long iters = 1;
    double dt;
    for (;;) {
        double t0 = now_s();
        for (long i = 0; i < iters; i++) run(ctx);
        dt = now_s() - t0;
        if (dt >= target || iters > (1L << 26)) break;
        iters *= 2;
    }
    double best = 1e30;
    for (int b = 0; b < 6; b++) {
        double t0 = now_s();
        for (long i = 0; i < iters; i++) run(ctx);
        double d = (now_s() - t0) / iters;
        if (d < best) best = d;
    }
    return best * 1e9;
}

static double gflops(long n, double ns, int real) {
    if (ns <= 0 || n <= 1) return 0;
    double f = 5.0 * n * log2((double)n);
    if (real) f *= 0.5;
    return f / (ns * 1e-9) / 1e9;
}

typedef struct { fftw_plan p; } ctx_t;
static void run_plan(void *c) { fftw_execute(((ctx_t *)c)->p); }

static const long CSIZES[] = {256, 1024, 4096, 65536, 1048576,
                              1000, 1080, 1920, 1009, 1296, 10007};
static const long RSIZES[] = {256, 1024, 4096, 65536, 1048576, 1000, 1080, 1920};
static const long SH[][2] = {{64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}};

int main(void) {
    printf("{\"fftw_version\":\"%s\",\"complex\":[", fftw_version);
    fprintf(stderr, "# FFTW %s (native, 1 thread, FFTW_MEASURE)\n", fftw_version);

    /* ---- complex 1-D ---- */
    fprintf(stderr, "\n## Complex FFT (ns/op | GFLOP/s | plan ns)\n");
    for (size_t s = 0; s < sizeof(CSIZES) / sizeof(*CSIZES); s++) {
        long n = CSIZES[s];
        fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n);
        double p0 = now_s();
        fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
        double plan_ns = (now_s() - p0) * 1e9;
        for (long i = 0; i < n; i++)
            in[i] = ((i * 7 + 1) % 13) * 0.1 + I * (((i * 3 + 2) % 11) * 0.1);
        ctx_t c = {p};
        double ns = bench_ns(run_plan, &c, 0.2);
        printf("%s{\"n\":%ld,\"fftw_ns\":%.1f,\"fftw_gflops\":%.3f,\"fftw_plan_ns\":%.1f}",
               s ? "," : "", n, ns, gflops(n, ns, 0), plan_ns);
        fprintf(stderr, "%9ld  %12.0f  %7.2f  %12.0f\n", n, ns, gflops(n, ns, 0), plan_ns);
        fftw_destroy_plan(p); fftw_free(in); fftw_free(out);
    }

    /* ---- real 1-D (r2c) ---- */
    printf("],\"real\":[");
    fprintf(stderr, "\n## Real RFFT (ns/op | GFLOP/s | plan ns)\n");
    for (size_t s = 0; s < sizeof(RSIZES) / sizeof(*RSIZES); s++) {
        long n = RSIZES[s];
        double *in = fftw_malloc(sizeof(double) * n);
        fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1));
        double p0 = now_s();
        fftw_plan p = fftw_plan_dft_r2c_1d(n, in, out, FFTW_MEASURE);
        double plan_ns = (now_s() - p0) * 1e9;
        for (long i = 0; i < n; i++) in[i] = ((i * 7 + 1) % 13) * 0.1;
        ctx_t c = {p};
        double ns = bench_ns(run_plan, &c, 0.2);
        printf("%s{\"n\":%ld,\"fftw_ns\":%.1f,\"fftw_gflops\":%.3f,\"fftw_plan_ns\":%.1f}",
               s ? "," : "", n, ns, gflops(n, ns, 1), plan_ns);
        fprintf(stderr, "%9ld  %12.0f  %7.2f  %12.0f\n", n, ns, gflops(n, ns, 1), plan_ns);
        fftw_destroy_plan(p); fftw_free(in); fftw_free(out);
    }

    /* ---- 2-D complex ---- */
    printf("],\"fft2\":[");
    fprintf(stderr, "\n## 2-D FFT2 (ns/op | GFLOP/s | plan ns)\n");
    for (size_t s = 0; s < sizeof(SH) / sizeof(*SH); s++) {
        long r = SH[s][0], cc = SH[s][1], n = r * cc;
        fftw_complex *in = fftw_malloc(sizeof(fftw_complex) * n);
        fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * n);
        double p0 = now_s();
        fftw_plan p = fftw_plan_dft_2d(r, cc, in, out, FFTW_FORWARD, FFTW_MEASURE);
        double plan_ns = (now_s() - p0) * 1e9;
        for (long i = 0; i < n; i++)
            in[i] = ((i * 7 + 1) % 13) * 0.1 + I * (((i * 3 + 2) % 11) * 0.1);
        ctx_t c = {p};
        double ns = bench_ns(run_plan, &c, 0.2);
        printf("%s{\"shape\":\"%ldx%ld\",\"n\":%ld,\"fftw_ns\":%.1f,\"fftw_gflops\":%.3f,\"fftw_plan_ns\":%.1f}",
               s ? "," : "", r, cc, n, ns, gflops(n, ns, 0), plan_ns);
        fprintf(stderr, "%5ldx%-5ld  %12.0f  %7.2f  %12.0f\n", r, cc, ns, gflops(n, ns, 0), plan_ns);
        fftw_destroy_plan(p); fftw_free(in); fftw_free(out);
    }
    printf("]}\n");
    return 0;
}
