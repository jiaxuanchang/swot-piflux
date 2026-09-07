"""
Example: spectral energy flux (Pi_L / Pi(k)) via coarse-graining
(CG) and via the Fourier transfer-spectrum method, for SSH shaped
like an actual SWOT pass -- (num_lines, num_pixels), track-oriented (not
lat/lon-aligned), with a real nadir gap. This is the SWOT-swath sibling of
scripts/examples/regular_grid_pipeline.py -- see that file's docstring
for the regular-model-grid case (HYCOM/MOM6/ROMS/LLC4320 on their native
grid), which uses a plain stencil-3 centered difference instead.

WHO THIS IS FOR
    Anyone whose SSH is already gridded to look like a real SWOT pass --
    e.g. HYCOM run through the official SWOT simulator/OSSE pipeline
    (https://swot-simulator.readthedocs.io), which has a plugin for regional
    HYCOM output specifically. That output is swath-shaped (num_lines x
    num_pixels), not a regular lat/lon grid, and has a real nadir gap plus
    per-pixel measurement noise/dropout, by design -- the whole point of an
    OSSE is to look like what the real instrument would actually see.

1D METHOD ONLY, NOT JUST A 1D OUTPUT ARRAY (both CG and Fourier)
    Every Pi(scale)/Pi(k) curve is a 1D array of course -- that's true of the
    2D/isotropic method too (see regular_grid_pipeline.py). What's different
    here is that the filtering/FFT step itself never sees 2D spatial
    structure: it only ever operates along the along-track axis, treating
    each cross-track pixel as an independent 1D signal, then averages the
    per-pixel results together afterward. Concretely:
      - CG: filter_1d_spatial_tophat is a 1D boxcar averaged along-track
        only (swot_func/coarse_graining.py). The 2D version instead
        convolves with a genuine 2D circular disk kernel
        (filter_2d_spatial_tophat) -- isotropic, sees structure at every
        orientation, not just along-track.
      - Fourier: compute_transfer_and_flux_for_strip does a 1D FFT
        (np.fft.rfft) on one cross-track pixel's along-track signal at a
        time; results are averaged over pixels afterward. The 2D version
        (compute_transfer_and_flux_2d) does a genuine 2D FFT over the whole
        patch, then bins by the isotropic radial wavenumber
        |k| = sqrt(kx^2+ky^2) (spectralflux_utils.py::ring_sum_and_flux).
    These are NOT the same calculation with the same answer reshaped: a 1D
    along-track boxcar/FFT and a 2D isotropic disk/ring-sum treat
    differently-oriented structures differently, so they generally give
    different Pi(L)/Pi(k) even for the same field (this repo's
    invert_abel_A6 exists specifically to convert between the two via an
    Abel transform -- they're related, not identical).
    One swath side is only ~27 pixels (~54 km) cross-track -- far too narrow
    for the 2D disk kernel / 2D FFT to mean anything -- so this script can
    only use the 1D method, matching what the real SWOT production pipeline
    in the parent repo does for the same reason. If you need a genuine 2D/
    isotropic estimate, your data has to be regular-grid-shaped instead (use
    regular_grid_pipeline.py). The closest stand-in available here is
    abel_invert_fourier_1d, which Abel-inverts the 1D Fourier transfer
    density into an isotropic-2D-equivalent Pi(Kh) -- an approximation, not
    an independently-computed 2D result, but validated in the parent repo
    against LLC4320's genuinely-computed 2D Fourier curve on the same
    region.

WHY TRANCHANT HERE, STENCIL-3 IN THE OTHER SCRIPT
    Two separate reasons a plain centered difference isn't good enough on a
    real SWOT swath, both handled by the Tranchant method
    (compute_ocean_diagnostics_from_eta):

    1. Noise and gaps. A centered difference needs every one of its stencil
       points valid and produces one noisy finite difference per point.
       Tranchant instead fits a 2D polynomial over a local circular kernel
       (n points wide) and reads the derivative off the fit -- tolerant of
       some missing points within the kernel (min_valid_points), and
       effectively low-pass-filters small-scale noise as part of the fit.

    2. Coordinate rotation. (num_lines, num_pixels) are along-track /
       cross-track, i.e. oriented along the satellite's ground track, not
       aligned with geographic east/north -- and the ground track's heading
       drifts with latitude, so that misalignment isn't even constant across
       one pass. A derivative taken directly in (num_lines, num_pixels)
       coordinates would be in the wrong frame. Tranchant computes the local
       track angle theta = compute_angle(lon, lat) at each point and rotates
       the fitted derivatives into the geographic (east, north) frame before
       returning ug, vg -- this rotation has no equivalent in the plain
       stencil path, which assumes its two axes already ARE east/north (true
       for a regular lat/lon model grid, false for a SWOT swath).

    This is what compute_for_pass in swot_func/spectralflux_utils.py (the
    real SWOT production pipeline in the parent repo) actually uses.

    One Tranchant fit gives BOTH the geostrophic velocity (ug, vg) AND SSH's
    second derivatives (dxx, dyy, dxy) directly -- the nonlinear advection
    term N1, N2 needed by the Fourier function is then derived analytically
    from those second derivatives (ux = -(g/f)*dxy, uy = -(g/f)*dyy, etc.),
    NOT by re-differentiating ug, vg with a second, noisier finite-difference
    pass. This is exactly what compute_for_pass does; see the formulas in
    process_one_side below.

NADIR GAP / LEFT-RIGHT SPLIT
    A real SWOT pass has two swaths (roughly 10-60 km either side of nadir)
    separated by a ~20 km gap with no data. This example's synthetic swath
    mimics that with a fixed band of always-NaN pixels near the middle of
    the cross-track dimension (empirically matched to a real downloaded
    SWOT pass: pixels ~30-37 out of 69 were >90% NaN in every one of 31
    cycles checked). The two sides are processed SEPARATELY
    (process_one_side called once per side) and their strip-level results
    combined afterward -- the same convention run_fourier_pass.py and
    run_cg_pass.py use for real SWOT data (RAW_LEFT_PIX / RAW_RIGHT_PIX).
    Feeding one array with an internal gap through the pipeline and relying
    on NaN-propagation to sort it out is NOT what this repo's production
    code does, and hasn't been validated here -- don't do that.

WHAT YOU NEED TO SUPPLY (per side, left and right separately)
    - ssh       : 2D sea surface height anomaly (num_lines, num_pixels_side), meters
    - lat, lon  : 2D coordinate arrays (num_lines, num_pixels_side), degrees

SIGN CONVENTION / AVERAGING-OVER-TIME NOTE: see
scripts/examples/regular_grid_pipeline.py -- both apply unchanged here.

DEPENDENCIES: same as regular_grid_pipeline.py, see
scripts/examples/README.md.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: never blocks on a plot window
import matplotlib.pyplot as plt

# Search upward for a directory containing swot_func/ -- see
# regular_grid_pipeline.py for why this isn't a fixed number of "..".
_here = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = None
for _candidate in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.isdir(os.path.join(_candidate, "swot_func")):
        REPO_ROOT = _candidate
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not find a swot_func/ directory above this script.")
sys.path.insert(0, REPO_ROOT)

import xarray as xr
from swot_func.coarse_graining import compute_sfs_energy_flux
from swot_func.spectralflux_utils import compute_transfer_and_flux_for_strip, invert_abel_A6
from swot_func.grid_utils import compute_dx_dy
from swot_func.Tranchant.diagnosis import compute_ocean_diagnostics_from_eta

G = 9.81            # m/s^2
OMEGA = 7.2921e-5    # Earth's rotation rate, rad/s

# Matches the production defaults in compute_for_pass (n=5) -- a 5-point-wide
# circular fitting kernel. Increase n for noisier data (more smoothing, less
# resolution); decrease for cleaner data.
TRANCHANT_PARAMS = dict(derivative="fit", n=5, min_valid_points=0.75, cyclostrophy="GW",
                         avoid_negative=False, second_derivative="dxdy", kernel="circular",
                         verbose=False)

# Empirically-measured layout of a real 69-pixel SWOT pass (see docstring above).
N_PIXELS_TOTAL = 69
LEFT_PIX = slice(3, 30)    # 27 pixels, matches the real "clean" left-swath pixels
RIGHT_PIX = slice(38, 65)  # 27 pixels, matches the real "clean" right-swath pixels
GAP_PIX = slice(30, 38)    # nadir gap: always NaN


# ---------------------------------------------------------------------------
# 1. Get your swath SSH: full (n_lines, N_PIXELS_TOTAL) ssh/lat/lon, with the
#    nadir gap as NaN -- then LEFT_PIX/RIGHT_PIX slice out the two sides.
# ---------------------------------------------------------------------------

def make_synthetic_swot_swath_ssh(n_lines=1400, dx_km=2.0, seed=0, dropout_frac=0.15):
    """Toy SSH shaped like one real SWOT pass swath: (n_lines, 69), with a
    structural nadir gap (always NaN, matching real data) plus scattered
    random dropout elsewhere (clouds/rain/quality flags -- ~15% is typical
    for real SWOT). REPLACE this whole function with your own swath-shaped
    data loading (e.g. reading a SWOT-simulator/OSSE output netCDF)."""
    rng = np.random.default_rng(seed)
    nx = N_PIXELS_TOTAL

    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(n_lines, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0

    amp = K ** (-1.5)
    phase = rng.uniform(0, 2 * np.pi, size=K.shape)
    ssh_hat = amp * np.exp(1j * phase)
    ssh_hat[0, 0] = 0.0
    ssh = np.real(np.fft.ifft2(ssh_hat))
    ssh = ssh / np.std(ssh) * 0.1

    # Structural nadir gap + edge padding, matching real measured layout.
    ssh[:, :3] = np.nan
    ssh[:, GAP_PIX] = np.nan
    ssh[:, -4:] = np.nan
    # Scattered random dropout elsewhere (clouds, rain, quality flags).
    valid_region = np.ones(nx, dtype=bool)
    valid_region[:3] = False
    valid_region[GAP_PIX] = False
    valid_region[-4:] = False
    dropout = rng.random(ssh.shape) < dropout_frac
    ssh[dropout & valid_region[None, :]] = np.nan

    lat0, lon0 = 30.0, 150.0  # arbitrary mid-latitude, along-track origin
    deg_per_km = 1.0 / 111.0
    y_idx, x_idx = np.meshgrid(np.arange(n_lines), np.arange(nx), indexing="ij")
    # A small cross-track tilt so the swath is track-oriented, not exactly
    # lat/lon-aligned -- closer to a real pass geometry than a pure rectangle.
    lat = lat0 + y_idx * dx_km * deg_per_km + x_idx * 0.1 * deg_per_km
    lon = lon0 + x_idx * dx_km * deg_per_km / np.cos(np.deg2rad(lat0)) - y_idx * 0.1 * deg_per_km

    return ssh, lat, lon


# ---------------------------------------------------------------------------
# 2. SSH -> geostrophic velocity (Tranchant fit) -> CG + Fourier, one side
#    at a time. This is the ONLY way velocity enters the pipeline below --
#    there is no code path that accepts u, v directly.
# ---------------------------------------------------------------------------

def gap_fill_scattered_nan(ssh_side):
    """Linearly interpolate over the scattered (non-structural) dropout
    within one already-sliced side -- clouds/rain/quality-flag gaps, not the
    nadir gap (already excluded by LEFT_PIX/RIGHT_PIX before this is called).
    Real SWOT production data goes through an equivalent griddata-based
    gap-fill before ever reaching CG or Fourier (see
    scripts/production/preprocess_ssha_variant.py in the parent repo) --
    skip this and CG's tophat filter chokes on interior NaN (verified: with
    ~15% scattered dropout and no gap-fill, compute_sfs_energy_flux comes
    back all-NaN)."""
    from scipy.interpolate import griddata
    ny, nx = ssh_side.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    valid = np.isfinite(ssh_side)
    if valid.all() or not valid.any():
        return ssh_side
    filled = griddata((yy[valid], xx[valid]), ssh_side[valid], (yy, xx), method="linear")
    # linear leaves NaN outside the convex hull of valid points; fall back to nearest there
    still_nan = np.isnan(filled)
    if still_nan.any():
        filled[still_nan] = griddata((yy[valid], xx[valid]), ssh_side[valid],
                                      (yy[still_nan], xx[still_nan]), method="nearest")
    return filled


def process_one_side(ssh_side, lat_side, lon_side, dx_m, dy_m):
    """Run the Tranchant-based CG + Fourier pipeline on one swath side (left
    or right, already sliced). Mirrors swot_func/spectralflux_utils.py's
    compute_for_pass exactly: one Tranchant fit gives ug, vg AND SSH's
    second derivatives, from which N1, N2 (nonlinear advection) are derived
    analytically -- see the module docstring for why.

    Returns pi_L, scale_km, Pi_fourier, k_fourier, T_fourier -- all 1D (one
    value per scale/k, already averaged over the cross-track pixels of this
    side); see "1D METHOD ONLY, NOT JUST A 1D OUTPUT ARRAY" in the module
    docstring for why there's no 2D version here. T_fourier is the transfer
    density itself (shares k_fourier), not a cumulative Pi -- see
    abel_invert_fourier_1d below for what it's for."""
    ny, nx = ssh_side.shape
    ssh_side = gap_fill_scattered_nan(ssh_side)
    ssh_xr = xr.DataArray(ssh_side, dims=("along_track", "cross_track"))
    diag = compute_ocean_diagnostics_from_eta(ssh_xr, lon_side, lat_side, **TRANCHANT_PARAMS)

    f = 2 * OMEGA * np.sin(np.deg2rad(lat_side))
    ug = np.asarray(diag["ug"])
    vg = np.asarray(diag["vg"])
    ux = -(G / f) * np.asarray(diag["dxy"])
    uy = -(G / f) * np.asarray(diag["dyy"])
    vx = (G / f) * np.asarray(diag["dxx"])
    vy = (G / f) * np.asarray(diag["dxy"])
    N1 = ug * ux + vg * uy
    N2 = ug * vx + vg * vy

    # 1d_spatial_tophat filters along the ALONG-TRACK axis (the long one, ny
    # lines) -- unlike the regular-grid example, a SWOT swath side is narrow
    # cross-track (nx ~27 pixels, ~54 km) and long along-track (ny ~1000s of
    # lines, ~1000s of km), so Lmax_m must scale with ny/dy, not nx/dx.
    # Getting this backwards silently produces almost no usable scale range
    # (verified: nx-based Lmax_m here gave only 3 scale points total).
    cg_result = compute_sfs_energy_flux(
        ug, vg, lon_side, lat_side, dx_m, dy_m,
        sca=1.2,
        Lmin_m=4 * np.nanmean(dy_m),
        Lmax_m=0.3 * ny * np.nanmean(dy_m),
        filter_method="1d_spatial_tophat",
    )
    pi_L = cg_result["energy_flux"].mean(dim=["along_track", "cross_track"]).values
    scale_km = cg_result["scale"].values

    Pi_list, T_list = [], []
    for ix in range(nx):
        Tk, Pik = compute_transfer_and_flux_for_strip(
            ug[:, ix], vg[:, ix], N1[:, ix], N2[:, ix], tukey_r=0.2, min_valid=0.75,
        )
        if Tk is not None:
            Pi_list.append(Pik)
            T_list.append(Tk)

    if not Pi_list:
        return pi_L, scale_km, None, None, None
    Pi_fourier = np.nanmean(np.array(Pi_list), axis=0)
    T_fourier = np.nanmean(np.array(T_list), axis=0)
    nk = len(Pi_fourier)
    ny_fft = nk * 2 - 2
    k_fourier = np.fft.rfftfreq(ny_fft, d=np.nanmean(dy_m) / 1000.0)
    return pi_L, scale_km, Pi_fourier, k_fourier, T_fourier


def process_one_snapshot(ssh, lat, lon, dx_m, dy_m):
    """Split into left/right swaths and process each separately, then
    average their strip-level results together -- matching how
    run_fourier_pass.py / run_cg_pass.py combine real SWOT left+right data."""
    results = []
    for pix_slice in (LEFT_PIX, RIGHT_PIX):
        r = process_one_side(ssh[:, pix_slice], lat[:, pix_slice], lon[:, pix_slice],
                              dx_m[:, pix_slice], dy_m[:, pix_slice])
        if r[2] is not None:  # Pi_fourier not None
            results.append(r)
    pi_L = np.nanmean([r[0] for r in results], axis=0)
    scale_km = results[0][1]
    Pi_fourier = np.nanmean([r[2] for r in results], axis=0)
    k_fourier = results[0][3]
    T_fourier = np.nanmean([r[4] for r in results], axis=0)
    return pi_L, scale_km, Pi_fourier, k_fourier, T_fourier


def abel_invert_fourier_1d(k, T, smooth_window=5, polyorder=1):
    """
    Convert the 1D along-track Fourier transfer density T(k) into an
    isotropic-2D-equivalent Pi(Kh), via invert_abel_A6 (swot_func/
    spectralflux_utils.py). There's no genuine 2D estimate possible here (a
    swath side is only ~27 pixels cross-track -- see "1D METHOD ONLY..." in
    the module docstring), so this is the closest stand-in for one:
    validated in the parent repo against LLC4320's genuinely-computed 2D
    Fourier curve on the same region.

    k, T: 1D arrays, same shape, increasing k -- pass T (the transfer
    density, e.g. T_fourier from process_one_snapshot), NOT Pi_fourier (the
    cumulative flux); invert_abel_A6 operates on the density.

    Returns (Kh, Pi_abel): the Abel-inverted radial wavenumber grid and the
    corresponding cumulative isotropic-equivalent flux.
    """
    valid = np.isfinite(k) & (k > 0) & np.isfinite(T)
    k_s, T_s = k[valid], T[valid]
    idx = np.argsort(k_s)
    k_s, T_s = k_s[idx], T_s[idx]
    dk = np.diff(k_s)
    dk = np.append(dk, dk[-1])
    T_density = T_s / dk
    k_center = k_s + 0.5 * dk
    Kh, Tr = invert_abel_A6(k_center, T_density, smooth_window=smooth_window, polyorder=polyorder)
    dKh = np.diff(Kh)
    dKh = np.append(dKh, dKh[-1])
    Pi_abel = np.cumsum((Tr[::-1] * dKh[::-1]))[::-1]
    return Kh, Pi_abel


# Grid geometry is fixed across the synthetic snapshots here, so compute it
# once. If your real data's swath geometry genuinely changes per pass/cycle,
# move this inside the loop instead.
ssh0, lat0, lon0 = make_synthetic_swot_swath_ssh(seed=0)
dy_m, dx_m = compute_dx_dy(lat0, lon0)
print(f"Grid spacing: dx={np.nanmean(dx_m):.0f} m, dy={np.nanmean(dy_m):.0f} m")

# --- REPLACE this loop with one over your own swath-data time steps/cycles,
# loading ssh, lat, lon for each and calling
# process_one_snapshot(ssh, lat, lon, dx_m, dy_m) ---
N_SNAPSHOTS = 20  # like averaging over ~20 SWOT cycles -- more snapshots -> smoother Pi(k)
pi_L_list, pi_fourier_list, T_fourier_list = [], [], []
scale_km = k_fourier = None
for it in range(N_SNAPSHOTS):
    ssh, lat, lon = make_synthetic_swot_swath_ssh(seed=it)
    pi_L, scale_km, Pi_fourier, k_fourier, T_fourier = process_one_snapshot(ssh, lat, lon, dx_m, dy_m)
    pi_L_list.append(pi_L)
    pi_fourier_list.append(Pi_fourier)
    T_fourier_list.append(T_fourier)
    print(f"  snapshot {it + 1}/{N_SNAPSHOTS} done")
# --- end of loop to replace ---

pi_L = np.nanmean(pi_L_list, axis=0)
Pi_fourier_mean = np.nanmean(pi_fourier_list, axis=0)
k_cg = 1.0 / scale_km  # cpkm

# Abel-invert the time-averaged 1D transfer density (not per-snapshot, and
# not the already-cumulative Pi_fourier -- see abel_invert_fourier_1d) into
# an isotropic-2D-equivalent estimate. This is the closest thing to a "2D"
# result this script can produce -- see "1D METHOD ONLY..." in the module
# docstring for why there's no genuine 2D estimate from a swath this narrow.
T_fourier_mean = np.nanmean(T_fourier_list, axis=0)
k_abel, pi_abel = abel_invert_fourier_1d(k_fourier, T_fourier_mean)

print(f"Input field: {ssh.shape} (full swath, before left/right split), "
      f"averaged over {N_SNAPSHOTS} snapshots")


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.plot(k_cg, pi_L * 1e9, "o-", label="CG $\\Pi_L$")
ax.plot(k_fourier, Pi_fourier_mean * 1e9, "s-", label="Fourier $\\Pi(k)$ (1D)")
ax.plot(k_abel, pi_abel * 1e9, "^:", label="Fourier $\\Pi(k)$ (1D, Abel-inverted)")
ax.set_xscale("log")
ax.set_xlabel("Wavenumber $k$ (cpkm)")
ax.set_ylabel(r"$\Pi$ (nW kg$^{-1}$)")
ax.set_title("Spectral energy flux from a SWOT-swath SSH: CG vs Fourier (1D estimates)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "swot_swath_pipeline_output.png")
plt.savefig(out_path, dpi=130)
print(f"Saved figure: {out_path}")
