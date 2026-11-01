# Coverage against SciPy, NumPy and MATLAB

Inventory of what `go-fft/fft` (v0.28.8) implements against `scipy.fft`
v1.18.1, `numpy.fft` v2.5.3 and the MATLAB FFT function family, with the
FFTW capabilities listed by name. Status: **done** (same semantics and the
options that matter), **partial** (what is missing is named), **missing**.
"Dep" is what a missing item needs beyond this package.

Sources. SciPy: `scipy/fft/__init__.py` at tag v1.18.1 (its `__all__`, 41
names, is followed line by line). NumPy: `numpy/fft/_pocketfft.py` and
`_helper.py` `__all__` at tag v2.5.3 (18 names). Go side: README, `go doc`,
`docs/plan-fft.md` ("Not planned" list) and the sources on main. MATLAB: the
MathWorks function lists are rendered client-side and could not be read, so
the MATLAB and FFTW columns come from knowledge of the documented function
families and must be re-checked against the MathWorks and fftw.org manuals
before a slice is started.

Conventions of the Go API that apply to every row: data are flat row-major
slices plus an explicit `shape`; numpy's `n`, `norm`, `axes` are `Options`
fields of the `...With` variants; single precision has a `32` twin of each
transform; `Plan` types are the allocation-free, in-place-capable form.

## Totals

| | done | partial | missing | rows |
|---|---|---|---|---|
| `scipy.fft` v1.18.1 | 10 | 17 | 14 | 41 |
| `numpy.fft` v2.5.3 | 10 | 8 | 0 | 18 |
| MATLAB (FFT family, Signal Processing Toolbox spectral functions) | 4 | 9 | 12 | 25 |

## scipy.fft and numpy.fft

NumPy column: the name if `numpy.fft` has it, `-` if not.

| # | SciPy | NumPy | Go | Status | Missing / dep |
|---|---|---|---|---|---|
| 1 | `fft` | `fft` | `FFT`, `FFTWith`, `FFT32`, `Plan` | done | |
| 2 | `ifft` | `ifft` | `IFFT`, `IFFTWith`, `IFFT32`, `Plan` | done | |
| 3 | `fft2` | `fft2` | `FFT2`, `FFT2With`, `FFT2_32` | partial | `s` (per-axis pad/crop), listed in plan-fft "Not planned"; none |
| 4 | `ifft2` | `ifft2` | `IFFT2`, `IFFT2With`, `IFFT2_32` | partial | `s` |
| 5 | `fftn` | `fftn` | `FFTN`, `FFTNWith`, `FFTN32`, `PlanN` | partial | `s`; an axis listed twice is refused (numpy transforms it twice) |
| 6 | `ifftn` | `ifftn` | `IFFTN`, `IFFTNWith`, `IFFTN32` | partial | `s` |
| 7 | `rfft` | `rfft` | `RFFT`, `RFFTWith`, `RFFT32`, `RealPlan` | done | |
| 8 | `irfft` | `irfft` | `IRFFT`, `IRFFTWith`, `IRFFT32` | done | |
| 9 | `rfft2` | `rfft2` | `RFFT2`, `RFFT2With`, `RealPlan2` | partial | `s` |
| 10 | `irfft2` | `irfft2` | `IRFFT2`, `IRFFT2With` | partial | `s` (Go takes the real output shape instead) |
| 11 | `rfftn` | `rfftn` | `RFFTN`, `RFFTNWith`, `RFFTN32` | partial | `s` |
| 12 | `irfftn` | `irfftn` | `IRFFTN`, `IRFFTNWith` | partial | `s` (output shape instead) |
| 13 | `hfft` | `hfft` | `HFFT`, `HFFTWith`, `HFFT32` | done | |
| 14 | `ihfft` | `ihfft` | `IHFFT`, `IHFFTWith`, `IHFFT32` | done | |
| 15 | `hfft2` | - | | missing | none (conjugate + `IRFFTN` composition) |
| 16 | `ihfft2` | - | | missing | none |
| 17 | `hfftn` | - | | missing | none |
| 18 | `ihfftn` | - | | missing | none |
| 19 | `dct` | - | `DCT`, `DCT32`, `DCTPlan` | partial | types I-IV done; `n`, `axis` (batch), `orthogonalize=False` (plan-fft "Not planned"), `workers` |
| 20 | `idct` | - | `IDCT`, `IDCT32` | partial | same as `dct` |
| 21 | `dst` | - | `DST`, `DST32`, `DSTPlan` | partial | same as `dct` |
| 22 | `idst` | - | `IDST`, `IDST32` | partial | same as `dct` |
| 23 | `dctn` | - | `DCTN`, `DCTN32` | partial | `s`, `axes`, per-axis `type`, `orthogonalize=False` |
| 24 | `idctn` | - | `IDCTN`, `IDCTN32` | partial | same as `dctn` |
| 25 | `dstn` | - | `DSTN`, `DSTN32` | partial | same as `dctn` |
| 26 | `idstn` | - | `IDSTN`, `IDSTN32` | partial | same as `dctn` |
| 27 | `fht` | - | | missing | `loggamma` of a complex argument (go-numerics/special or go-bignum), plus `FFT` (have) |
| 28 | `ifht` | - | | missing | same as `fht` |
| 29 | `fhtoffset` | - | | missing | same as `fht` |
| 30 | `fftshift` | `fftshift` | `FFTShift`, `FFTShiftN` | done | |
| 31 | `ifftshift` | `ifftshift` | `IFFTShift`, `IFFTShiftN` | done | |
| 32 | `fftfreq` | `fftfreq` | `FFTFreq` | done | numpy's `device` argument has no meaning here |
| 33 | `rfftfreq` | `rfftfreq` | `RFFTFreq` | done | |
| 34 | `next_fast_len` | - | `NextFastLen` | partial | returns the next 7-smooth (even for real) length by design; scipy's table is 11-smooth |
| 35 | `prev_fast_len` | - | | missing | none |
| 36 | `set_workers` | - | | missing | none (large transforms use goroutines internally, with no user control) |
| 37 | `get_workers` | - | | missing | none |
| 38 | `set_backend` | - | | missing | not applicable (uarray backend protocol); decide to leave out |
| 39 | `skip_backend` | - | | missing | not applicable |
| 40 | `set_global_backend` | - | | missing | not applicable |
| 41 | `register_backend` | - | | missing | not applicable |

### Keyword arguments and dtypes (both libraries)

| Argument / type | Go | Status | Missing / dep |
|---|---|---|---|
| `n` | `Options.N` (1-D) | done | zero means default, not an error; zero-length returns empty (numpy raises) |
| `norm` | `Options.Norm` (backward, ortho, forward) | done | |
| `axis` (1-D transforms along any axis) | `FFTNWith` with `Axes: {-1}` (batched) | partial | no `axis` on the 1-D `FFT`, `RFFT`, `DCT`, `DST` functions; strided views need go-ndarray |
| `axes` (N-D) | `Options.Axes` | done | |
| `s` (N-D shape to pad or crop) | | missing | none |
| `overwrite_x` | `Plan.FFT(dst, src)` may alias | done | |
| `workers` | | missing | none |
| `plan` (scipy) | explicit `Plan` types, internal per-length cache | done | |
| complex128, float64 | all transforms | done | |
| complex64, float32 | `32` twins of every transform except windows/PSD | done | |
| float16, integer, bool input | caller converts | missing | go-ndarray dtype promotion |
| `longdouble` / `clongdouble` | | missing | double-double or go-bignum kernel; numpy and scipy use 80-bit extended on x86 |
| non-contiguous and negative-stride input | | missing | go-ndarray views |
| `out=` (numpy 2) | plans write into `dst` | done | |

## MATLAB (FFT family and spectral functions)

| # | MATLAB | Go | Status | Missing / dep |
|---|---|---|---|---|
| 1 | `fft` (`n`, `dim`) | `FFT`, `FFTWith` | done | |
| 2 | `ifft` (`n`, `dim`, `'symmetric'`) | `IFFT`, `IFFTWith` | done | `'symmetric'` is `IRFFT` of the half spectrum |
| 3 | `fft2` (`m`, `n`) | `FFT2` | partial | pad/crop sizes |
| 4 | `ifft2` | `IFFT2` | partial | pad/crop sizes |
| 5 | `fftn` (`sz`) | `FFTN` | partial | pad/crop sizes |
| 6 | `ifftn` | `IFFTN` | partial | pad/crop sizes |
| 7 | `fftshift` | `FFTShift`, `FFTShiftN` | done | |
| 8 | `ifftshift` | `IFFTShift`, `IFFTShiftN` | done | |
| 9 | `nextpow2` | | missing | none (trivial) |
| 10 | `fftw` (planner, wisdom) | internal plan cache | missing | not applicable; plans are explicit |
| 11 | `dct` (Signal Processing Toolbox; type II only, `n`) | `DCT` | partial | `n` |
| 12 | `idct` | `IDCT` | partial | `n` |
| 13 | `hann`, `hamming`, `blackman`, `blackmanharris`, `bartlett` | `Hann`, `Hamming`, `Blackman`, `BlackmanHarris`, `Bartlett` | partial | `'symmetric'` / `'periodic'` flag; other windows (`kaiser`, `chebwin`, `tukeywin`, `gausswin`, `flattopwin`, `nuttallwin`): `kaiser` needs Bessel I0 (go-special) |
| 14 | `periodogram` | `PSD` | partial | two-sided, centered, `nfft`, confidence bounds |
| 15 | `pwelch` | | missing | none (average of `Spectrogram` frames) |
| 16 | `spectrogram` | `Spectrogram` | partial | `nfft`, two-sided, complex output, time/frequency vectors |
| 17 | `stft`, `istft` | | missing | none (frames + `RFFT`; inverse needs overlap-add with window check) |
| 18 | `cpsd`, `mscohere`, `tfestimate` | | missing | none |
| 19 | `conv`, `deconv`, `xcorr`, `fftfilt` | | missing | none (FFT convolution; direct small-size path) |
| 20 | `hilbert` | | missing | none |
| 21 | `czt` | | missing | none (Bluestein core exists internally) |
| 22 | `goertzel` | | missing | none |
| 23 | `dftmtx` | | missing | none |
| 24 | `cceps`, `rceps`, `icceps` | | missing | phase unwrapping |
| 25 | `pspectrum` | | missing | `pwelch` plus spectrogram options |

## FFTW capabilities (names only, not counted)

| FFTW capability | Go | Status |
|---|---|---|
| `dft` any size (Cooley-Tukey, Rader, Bluestein) | yes | done |
| `r2c` / `c2r`, `dft_r2c_2d`, `dft_r2c_Nd` | `RFFT*`, `IRFFT*` | done |
| `r2r` REDFT00/10/01/11 (DCT-I..IV) | `DCT` types I-IV | done |
| `r2r` RODFT00/10/01/11 (DST-I..IV) | `DST` types I-IV | done |
| `r2r` DHT (discrete Hartley) | | missing |
| `r2r` R2HC / HC2R (halfcomplex layout) | complex half spectrum instead | missing |
| `r2r` N-D with per-axis kind | `DCTN`/`DSTN` one type for all axes | partial |
| in-place transforms | `Plan` dst aliasing src | done |
| single precision (`fftwf`) | `32` twins | done |
| long double (`fftwl`), quad (`fftwq`) | | missing |
| advanced / guru interface (`howmany`, strides, `dist`) | batching through `FFTNWith` Axes only | partial |
| split-format real/imaginary arrays | | missing |
| multithreading (`fftw_plan_with_nthreads`) | internal goroutines, not settable | partial |
| MPI distributed transforms | | missing |
| planner flags (`ESTIMATE`...`EXHAUSTIVE`), wisdom | not applicable | missing |
| `fftw_alignment_of`, SIMD-aligned alloc | not applicable | missing |

## Proposed slices

1. **Hermitian N-D and `s`** (rows 15-18, 3-6, 9-12, `s`): `hfft2`, `ihfft2`,
   `hfftn`, `ihfftn`, and pad/crop `s` on every N-D transform. No dependency.
2. **Helpers**: `prev_fast_len`, `nextpow2`, `set_workers` / `get_workers`
   (an `Options.Workers` field), `axis` on the 1-D functions. No dependency.
3. **DCT/DST completeness** (rows 19-26): `n`, batching, per-axis types,
   `orthogonalize=False`. No dependency.
4. **Fast Hankel** (rows 27-29): needs `loggamma` of a complex argument.
5. **Spectral analysis** (MATLAB 13-25): `pwelch`, `stft`/`istft`, `cpsd`,
   `mscohere`, convolution, `hilbert`, `czt`, `goertzel`, windows with flags.
   Probably a separate `go-signal` package on top of this one; `kaiser` needs
   Bessel I0 (go-special).
6. **Extended precision** (`longdouble`): double-double kernels or go-bignum;
   large and probably last.
7. **FFTW r2r leftovers**: DHT, halfcomplex layout, guru strides; strides
   want go-ndarray views.
