# swot-piflux

Computes sub-filter-scale kinetic energy flux from sea surface height
(SSH), two ways: coarse-graining (Pi_L) and the Fourier transfer
spectrum (Pi(k)). Built so the result is **directly comparable against
SWOT** (Surface Water and Ocean Topography) observations -- this is the
same computation this project uses on actual SWOT passes, extracted so it
can be run on SSH from somewhere else.

Typical use: you have SSH from another source (model output, another
altimetry product, a SWOT-simulator run, ...) and want a Pi_L/Pi(k)
estimate computed the same way, so the two are apples-to-apples. Two grid
shapes are handled -- regular lat/lon model output (HYCOM, MOM6, ROMS,
LLC4320, ...) and SWOT-swath-shaped data (track-oriented, with a nadir
gap) -- each with the geostrophic derivative method that actually suits
it (plain centered difference for the regular grid, Tranchant
fitting-kernel for the swath).

Two example scripts are included -- pick the one that matches your data:

| Your SSH is... | Use | Geostrophic derivative | CG/Fourier |
|---|---|---|---|
| On a regular, lat/lon-aligned model grid (HYCOM/MOM6/ROMS/LLC4320 native output) | `regular_grid_pipeline.py` | simple stencil-3 centered difference | 1D, 2D (isotropic), and 1D Abel-inverted to isotropic-equivalent |
| Shaped like a real SWOT pass (num_lines x num_pixels, track-oriented, has a nadir gap -- e.g. HYCOM run through the official SWOT simulator/OSSE pipeline) | `swot_swath_pipeline.py` | Tranchant fitting-kernel derivative | 1D, plus 1D Abel-inverted to isotropic-equivalent |

The regular-grid script computes 1D (along-track strips) and 2D (genuinely
isotropic) versions of both CG and Fourier, plus a 5th curve: the 1D
Fourier transfer density Abel-inverted into an isotropic-2D-equivalent
estimate, for direct comparison against the genuine 2D curve. The
SWOT-swath script can't compute a genuine 2D estimate (a single swath side
is only ~27 pixels / ~54 km wide cross-track, too narrow for that to mean
anything), so the Abel-inverted curve is the closest stand-in available.

## Quick start

```bash
pip install -r requirements.txt
python examples/regular_grid_pipeline.py   # regular grid
python examples/swot_swath_pipeline.py     # SWOT-swath grid
```

Both run on synthetic data out of the box (no external files needed) and
save a `..._output.png` next to themselves. Open either script and search
for `REPLACE` to see exactly where to swap in your own SSH data.

## What's in here

```
swot_func/
    coarse_graining.py       core CG: compute_sfs_energy_flux
    spectralflux_utils.py    core Fourier: compute_transfer_and_flux_for_strip
                              (1D) and compute_transfer_and_flux_2d (2D),
                              plus isotropic ring-summing / k-binning and
                              Tukey windowing
    geostrophy.py            shared SSH -> geostrophic velocity engine
                              (stencil and Tranchant-fit methods), used by
                              both coarse_graining.py and spectralflux_utils.py
    grid_utils.py            compute_dx_dy: grid spacing from lat/lon
    Tranchant/                fitting-kernel geostrophic/derivative method
                              (diagnosis.py, misc.py) -- third-party code,
                              see Attribution below. A hard import of
                              spectralflux_utils.py and geostrophy.py,
                              always required.
examples/
    regular_grid_pipeline.py   regular model grid, SSH in, Pi_L/Pi(k) out
    swot_swath_pipeline.py     SWOT-swath grid, SSH in, Pi_L/Pi(k) out
requirements.txt
LICENSE
```

This is a subset of a larger project's `swot_func/` package. Everything
here is either directly reachable from the two example
scripts' own calls, or (in `geostrophy.py`'s case) a hard import required
for the package to load at all -- SWOT-specific I/O and file-format
helpers, plotting utilities, Jacobian/vorticity diagnostics, and a handful
of unused functions in the files above (SWOT save-to-netcdf helpers,
Blackman-window filtering, an alternate `compute_for_pass`-style driver,
etc.) were left out. `Tranchant/` is copied verbatim, unpruned, since it's
someone else's code.

## Both pipelines take SSH in, not u/v

Neither script has a code path that accepts u, v directly -- geostrophic
velocity is always derived from SSH internally, so that Pi_L/Pi(k) are
always computed from the same SSH-derived velocity rather than a possibly
inconsistent externally-supplied one. If your model output happens to
include u, v directly and you want to bypass the geostrophic derivation,
you can call `compute_sfs_energy_flux` and
`compute_transfer_and_flux_for_strip` directly with your own velocity
instead.

**Why two different derivatives.** A plain centered difference
(`regular_grid_pipeline.py::geostrophic_velocity_from_ssh`, equivalent to
`swot_func/geostrophy.py::geostrophic_from_ssh(method='stencil')`) is fine
on a clean, densely-sampled, regular model grid. It is NOT robust to a real
SWOT swath's noise and gaps: it needs every stencil point valid, and
produces one noisy finite difference per point. `swot_swath_pipeline.py`
instead uses `swot_func/Tranchant/diagnosis.py::compute_ocean_diagnostics_from_eta`
-- a 2D polynomial fit over a local circular kernel, tolerant of some
missing points within the kernel, effectively low-pass-filtering
small-scale noise as part of the fit, and (just as importantly) rotating
the fitted derivatives from track-oriented (along-track/cross-track)
coordinates into the geographic (east/north) frame using the local track
angle -- a SWOT swath's two axes are not aligned with east/north, and that
misalignment isn't even constant across one pass. One Tranchant fit gives
both the velocity AND SSH's second derivatives directly, from which the
nonlinear advection term (N1, N2) is derived analytically rather than by
re-differentiating velocity a second, noisier time -- see
`swot_swath_pipeline.py::process_one_side` for the exact formulas.

## SWOT-swath specifics: nadir gap, left/right split, gap-fill

A real SWOT pass has two swaths (~10-60 km either side of nadir) separated
by a ~20 km gap with no data at all -- structurally NaN in every cycle, not
random dropout. `swot_swath_pipeline.py`'s synthetic data mimics this with a
fixed always-NaN pixel band (empirically matched against a real downloaded
SWOT pass: pixels ~30-37 of 69 were >90% NaN in every one of 31 cycles
checked). The two sides are processed SEPARATELY (`process_one_side` called
once per side, via `LEFT_PIX`/`RIGHT_PIX`) and combined afterward. Feeding
one array with an internal gap straight through and hoping NaN-propagation
sorts it out has not been validated here -- don't do that.

On top of the structural gap, real SWOT pixels also have ordinary scattered
dropout (clouds, rain, quality flags -- ~15% typical). `gap_fill_scattered_nan`
linearly interpolates over this before the Tranchant fit -- **this step is
required**: verified that skipping it leaves ~15% scattered NaN feeding
directly into `compute_sfs_energy_flux`'s top-hat filter, which chokes on
interior NaN and comes back entirely NaN. The structural nadir gap itself
is never filled (it's genuinely unobserved) -- it's excluded by the
left/right pixel slicing instead, before gap-filling ever runs.

**A scale-range gotcha if you adapt this further**: the 1D top-hat CG filter
runs along the ALONG-TRACK axis (the long one -- thousands of lines), not
cross-track (the narrow one -- tens of pixels). `Lmax_m` must scale with the
along-track extent; using the cross-track pixel count here (copy-pasted
from the regular-grid script without adjusting) silently produced only 3
usable scale points instead of ~25.

## Averaging over time

A single snapshot's Fourier Pi(k) estimate is intrinsically noisy -- the
nonlinear transfer term amplifies small-scale noise (two derivatives plus a
product), so individual wavenumber bins are noisy even from a clean input.
Both examples average over `N_SNAPSHOTS` synthetic realizations to
demonstrate this; do the same with your own time series (loop over time
steps, call `process_one_snapshot` for each, average the returned curves
across time) -- more time steps averaged, smoother the result.

## Sign convention

Pi > 0 means forward/downscale cascade (energy moving from large scales to
small scales); Pi < 0 means inverse/upscale cascade. This is the Aluie/Eyink
convention, and both `compute_sfs_energy_flux` and
`compute_transfer_and_flux_for_strip` follow it -- verified via synthetic
forward-Euler convergence tests (dE(k)/dt measured directly vs. the
function's own T(k), dt -> 0).

## Dependencies

`numpy`, `scipy`, `xarray`, `matplotlib`, `opencv-python` (used by
`coarse_graining.py`'s 2D top-hat filter), `haversine` (used by
`grid_utils.py::compute_dx_dy`), `pandas` (used by `Tranchant/misc.py`).
All pinned loosely in `requirements.txt` -- tested against no specific
versions, so pin exact versions yourself if you need
reproducibility.

## Attribution

`swot_func/Tranchant/` (`diagnosis.py`, `misc.py`) is the fitting-kernel
geostrophic/derivative method from
[**treden/SwotDiag**](https://github.com/treden/SwotDiag) by
Yves-Tristan Tranchant et al., released under the MIT License (see
`swot_func/Tranchant/LICENSE`, copied verbatim from the original). The
method is described in:

> Tranchant, Y.-T., Legresy, B., Foppert, A., Pena-Molino, B., and
> Phillips, H. (2025). SWOT reveals fine-scale balanced motions driving
> near-surface currents and dispersion in the Antarctic Circumpolar
> Current. *Earth and Space Science*, 12, e2025EA004248.
> https://doi.org/10.1029/2025EA004248

The copy here is extended (second-derivative support, used to compute the
nonlinear advection term N1/N2 analytically -- see `compute_for_pass`-style
usage in `spectralflux_utils.py`) beyond what the original repository
provides.

## License

This project (everything outside `swot_func/Tranchant/`) is released under
the MIT License -- see `LICENSE`. `swot_func/Tranchant/` carries its own
MIT License from the original author -- see `swot_func/Tranchant/LICENSE`.
