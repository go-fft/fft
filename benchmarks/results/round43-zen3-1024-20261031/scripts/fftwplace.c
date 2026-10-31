/* fftwplace: FFTW's n-point complex forward FFT (one FFTW_MEASURE plan, its
   plan printed) executed with the input at several byte offsets (mod 4 KB)
   from the output, in rotated rounds; prints the median and best ns of each. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

#define NOFF 6
#define ROUNDS 21

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 1024;
  int offs[NOFF] = {0, 576, 1088, 2048, 2112, 3136};
  char *arena;
  if (posix_memalign((void **)&arena, 4096, (size_t)n * 64 + 8192)) return 1;
  fftw_complex *out = (fftw_complex *)arena;
  fftw_complex *in[NOFF];
  for (int o = 0; o < NOFF; o++) {
    in[o] = (fftw_complex *)(arena + (size_t)n * 32 + offs[o]);
  }
  fftw_plan p = fftw_plan_dft_1d(n, in[0], out, FFTW_FORWARD, FFTW_MEASURE);
  fftw_print_plan(p);
  printf("\n");
  double t[NOFF][ROUNDS];
  int it = 20000;
  for (int r = 0; r < ROUNDS; r++) {
    for (int q = 0; q < NOFF; q++) {
      int o = (q + r) % NOFF;
      for (int i = 0; i < n; i++) { in[o][i][0] = (i * 7 + 1) % 13 * 0.1; in[o][i][1] = (i % 11) * 0.1; }
      double t0 = now();
      for (int k = 0; k < it; k++) fftw_execute_dft(p, in[o], out);
      t[o][r] = (now() - t0) / it * 1e9;
    }
  }
  for (int o = 0; o < NOFF; o++) {
    qsort(t[o], ROUNDS, sizeof(double), cmp);
    printf("FFTWPLACE %d in@%d median %.1f best %.1f\n", n, offs[o], t[o][ROUNDS / 2], t[o][0]);
  }
  return 0;
}
