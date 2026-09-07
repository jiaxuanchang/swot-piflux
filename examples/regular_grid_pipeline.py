"""
Example: spectral energy flux (Pi_L / Pi(k)) via coarse-graining
(CG) and via the Fourier transfer-spectrum method, BOTH 1D
(along-track strips) and 2D (genuinely isotropic), starting from SSH (sea
surface height) on a REGULAR GRID (not a SWOT-swath grid).

ONE OF TWO EXAMPLE SCRIPTS
    This one is for SSH on a regular, lat/lon-aligned model grid (HYCOM,
    MOM6, ROMS, LLC4320, ...) -- geostrophic velocity is derived with a
    simple stencil-3 centered difference, appropriate for that kind of
    clean, evenly-spaced, low-noise grid. Because the grid is a full 2D
    patch (not a narrow swath), BOTH 1D and 2D CG/Fourier are computed here.

    If your data is instead shaped like an actual SWOT pass (num_lines x
    num_pixels, track-oriented rather than lat/lon-aligned, with a real
    nadir gap and noisier per-pixel measurements -- e.g. HYCOM run through
    the official SWOT simulator/OSSE pipeline), use the OTHER script,
    scripts/examples/swot_swath_pipeline.py, instead -- it uses the
    fitting-kernel (Tranchant) derivative, which is what the real SWOT
    production pipeline in the parent repo uses for exactly that kind of
    grid, and is considerably more robust to its noise/gap characteristics
    than a plain centered difference would be. It's also 1D-only: a single
    SWOT swath side is only ~27 pixels (~54 km) wide cross-track, too narrow
    for an isotropic 2D estimate to mean anything -- the real SWOT
    production pipeline in this repo never computes 2D either.

WHO THIS IS FOR
    Anyone who has their own gridded ocean model SSH output and wants to
    compute the sub-filter-scale kinetic energy flux Pi_L (coarse-graining)
    and/or the spectral transfer Pi(k) (Fourier), without any of the
    SWOT-specific preprocessing in the rest of this repo.

    The pipeline deliberately takes SSH as its input, not u/v -- geostrophic
    velocity is derived INSIDE the pipeline (geostrophic_velocity_from_ssh
    below), the same way it is for every dataset in this repo (SWOT only
    measures SSH; LLC4320 is run through the same derivation here for
    apples-to-apples comparison). Don't feed u,v in directly.

WHAT YOU NEED TO SUPPLY
    - ssh       : 2D sea surface height anomaly (ny, nx), meters
    - lat, lon  : 2D coordinate arrays (ny, nx), degrees
    - f         : 2D (or scalar) Coriolis parameter (ny, nx) or (ny, nx), 1/s
                  -- pass your own if you have it, otherwise it's computed
                  from lat below (f = 2*Omega*sin(lat)).

WHAT THIS SCRIPT DOES
    1. Runs on a synthetic toy SSH field by default (20 averaged snapshots),
       so you can execute it immediately with no external data and see what
       the output should look like. Marks exactly where to swap in your own
       data (search for "REPLACE").
    2. Derives geostrophic u, v from SSH (geostrophic_velocity_from_ssh).
    3. Computes Pi_L(scale) via coarse-graining, 1D (along-track strips) and
       2D (isotropic circular kernel) -- compute_sfs_energy_flux with
       filter_method="1d_spatial_tophat" / "2d_spatial_tophat".
    4. Computes Pi(k) via the Fourier method, 1D (strip-averaged
       along the along-track direction, compute_transfer_and_flux_for_strip)
       and 2D (genuinely isotropic FFT over the whole patch,
       compute_transfer_and_flux_2d).
    5. Abel-inverts the 1D Fourier transfer density into a 5th,
       isotropic-2D-equivalent curve (abel_invert_fourier_1d, via
       invert_abel_A6) -- useful when comparing against a case where only
       the 1D estimate is available (see swot_swath_pipeline.py).
    6. Plots all five curves on the same axes for comparison.

SIGN CONVENTION (Aluie/Eyink): Pi > 0 means forward/downscale cascade
(energy moving from large to small scales); Pi < 0 means inverse/upscale
cascade. Both functions below follow this convention.

NOTE ON THE DEFAULT SYNTHETIC DATA: the toy field below is pure random-phase
broadband noise shaped to a target power spectrum -- it has no genuine
coherent structures (eddies, jets), so its Fourier Pi(k) stays noticeably
noisy even after averaging over N_SNAPSHOTS realizations (the nonlinear
transfer estimate amplifies small-scale noise -- see the derivative
discussion in swot_func/spectralflux_utils.py). Real ocean data (HYCOM
included) has actual spatially-coherent structure, so Pi(k) from real data
should come out visibly smoother than this synthetic demo, given a
comparable number of averaged time steps. Don't read too much into this
example's own noise level -- it's here to prove the code runs and to show
the calling pattern, not to be a realistic result. The Abel-inverted curve
(fourier_1d_abel) is expected to look noisier still on this synthetic
field: invert_abel_A6 differentiates T(k) as part of the inversion, which
amplifies whatever small-scale noise is already in the 1D estimate one
step further.

DEPENDENCIES: this repo's swot_func/ package (coarse_graining.py,
spectralflux_utils.py, geostrophy.py, grid_utils.py, Tranchant/) plus
numpy, scipy, xarray, opencv-python (cv2), matplotlib. See
scripts/examples/README.md for exactly which files to copy.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: never blocks on a plot
                        # window, works headless too. Comment out (and use
                        # plt.show() again) if you want an interactive window
                        # instead of just the saved PNG.
import matplotlib.pyplot as plt

# Search upward for a directory containing swot_func/, rather than hardcoding
# a fixed number of ".." -- this file lives two levels down from the repo
# root in the parent repo (scripts/examples/) but only one level down in the
# standalone release package (examples/), see build_release_package.py.
_here = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = None
for _candidate in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if os.path.isdir(os.path.join(_candidate, "swot_func")):
        REPO_ROOT = _candidate
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not find a swot_func/ directory above this script.")
sys.path.insert(0, REPO_ROOT)

from swot_func.coarse_graining import compute_sfs_energy_flux
from swot_func.spectralflux_utils import (compute_transfer_and_flux_for_strip,
                                           compute_transfer_and_flux_2d, invert_abel_A6)
from swot_func.grid_utils import compute_dx_dy

G = 9.81            # m/s^2
OMEGA = 7.2921e-5    # Earth's rotation rate, rad/s


# ---------------------------------------------------------------------------
# 1. Get your SSH field: ssh, lat, lon, all shape (ny, nx)
# ---------------------------------------------------------------------------

def make_synthetic_ssh(ny=256, nx=256, dx_km=2.0, seed=0):
    """Toy turbulence-like SSH field with a roughly -3 spectral slope, for a
    runnable example with no external data. REPLACE this whole function with
    your own HYCOM loading -- you just need to end up with ssh, lat, lon as
    (ny, nx) arrays (meters, degrees, degrees)."""
    rng = np.random.default_rng(seed)

    kx = np.fft.fftfreq(nx, d=dx_km)
    ky = np.fft.fftfreq(ny, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0, 0] = 1.0  # avoid divide-by-zero at DC

    # SSH spectrum ~ K^-3 (typical mesoscale slope), random phases
    amp = K ** (-1.5)
    phase = rng.uniform(0, 2 * np.pi, size=K.shape)
    ssh_hat = amp * np.exp(1j * phase)
    ssh_hat[0, 0] = 0.0
    ssh = np.real(np.fft.ifft2(ssh_hat))
    ssh = ssh / np.std(ssh) * 0.1  # normalize to ~0.1 m typical SSHA

    lat0, lon0 = 30.0, 150.0  # arbitrary mid-latitude origin
    deg_per_km = 1.0 / 111.0
    y_idx, x_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    lat = lat0 + y_idx * dx_km * deg_per_km
    lon = lon0 + x_idx * dx_km * deg_per_km / np.cos(np.deg2rad(lat0))

    return ssh, lat, lon


# ---------------------------------------------------------------------------
# 2. SSH -> geostrophic velocity. This is the ONLY way velocity enters the
#    pipeline below -- there is no code path that accepts u, v directly.
# ---------------------------------------------------------------------------

def geostrophic_velocity_from_ssh(ssh, lat, dx_m, dy_m):
    """u = -(g/f) d(ssh)/dy, v = (g/f) d(ssh)/dx, stencil-3 centered
    differences on the physical grid spacing (np.gradient at interior points
    IS the 3-point central difference: (f[i+1]-f[i-1])/(2*dx)) -- the right
    choice for a regular, well-behaved model grid (HYCOM, MOM6, LLC4320, ...).

    swot_func/geostrophy.py::geostrophic_from_ssh (included in this package,
    since swot_func/spectralflux_utils.py has a hard import of it too) offers
    the same relation plus a stencil=7 higher-order option and a Tranchant
    fit-based method -- worth knowing about if you need it. This script
    deliberately doesn't call it directly: it returns (L-2, P-2) arrays via
    its own internal 1-pixel crop, and threading that crop convention
    through this simpler script's own indexing wasn't worth it for a
    same-formula, same-result function.

    If your data is instead shaped like an actual SWOT pass (num_lines x
    num_pixels, track-oriented not lat/lon-aligned, with a real nadir gap and
    noisier per-pixel measurements), use
    scripts/examples/swot_swath_pipeline.py instead -- it uses
    compute_ocean_diagnostics_from_eta (Tranchant), a fitting-kernel
    derivative that's far more robust to that kind of noisy, gappy swath
    data, and is what the actual SWOT production pipeline in the parent repo
    uses."""
    f = 2 * OMEGA * np.sin(np.deg2rad(lat))
    dssh_dy, dssh_dx = np.gradient(ssh, np.nanmean(dy_m), np.nanmean(dx_m))
    u = -(G / f) * dssh_dy
    v = (G / f) * dssh_dx
    return u, v


def process_one_snapshot(ssh, lat, lon, dx_m, dy_m):
    """Run the full SSH -> velocity -> CG + Fourier pipeline on a single
    (ny, nx) snapshot, BOTH 1D (along-track strips, directly comparable to
    the SWOT-swath script) and 2D (genuinely isotropic, only meaningful on a
    full 2D patch like this one -- there's no 2D counterpart in
    swot_swath_pipeline.py, a single SWOT swath side is too narrow
    cross-track for an isotropic 2D estimate to mean anything).

    Returns a dict with four (Pi, k) pairs -- cg_1d, cg_2d, fourier_1d,
    fourier_2d -- plus fourier_1d_T (the 1D transfer density itself, not a
    (Pi, k) pair; see abel_invert_fourier_1d below for what it's for). Call
    this once per time step and average each pair's Pi across time (see the
    loop below), the same way the production pipeline averages across SWOT
    cycles/passes. A SINGLE snapshot's Fourier estimate is intrinsically
    noisy (each wavenumber bin is one realization); only the multi-snapshot
    average is meaningful.

    dx_m, dy_m are passed in (not recomputed here) since compute_dx_dy is a
    per-pixel haversine loop -- expensive, and unnecessary to repeat if your
    grid geometry is fixed across time steps (typical for model output)."""
    ny, nx = ssh.shape
    u, v = geostrophic_velocity_from_ssh(ssh, lat, dx_m, dy_m)

    # Nonlinear advection term N = u . grad(u): N1 = u*du/dx + v*du/dy,
    # N2 = u*dv/dx + v*dv/dy. Plain np.gradient is used here for simplicity;
    # the SWOT production pipeline in the parent (unpruned) repo uses a
    # validated 3-point stencil (swot_func.diff_utils.central_diff_3_variable_spacing,
    # not included in this trimmed package) for exact parity with the CG
    # derivative -- worth knowing about if you need bit-for-bit agreement.
    du_dy, du_dx = np.gradient(u, np.nanmean(dy_m), np.nanmean(dx_m))
    dv_dy, dv_dx = np.gradient(v, np.nanmean(dy_m), np.nanmean(dx_m))
    N1 = u * du_dx + v * du_dy
    N2 = u * dv_dx + v * dv_dy

    result = {}

    # --- CG: 1D (along-track strips) and 2D (isotropic circular kernel) ---
    for tag, filter_method in (("1d", "1d_spatial_tophat"), ("2d", "2d_spatial_tophat")):
        cg_result = compute_sfs_energy_flux(
            u, v, lon, lat, dx_m, dy_m,
            sca=1.2,                             # scale growth factor between successive filter scales
            Lmin_m=4 * np.nanmean(dx_m),         # smallest resolvable scale ~2x grid spacing
            Lmax_m=0.3 * nx * np.nanmean(dx_m),  # largest scale, well inside the domain
            filter_method=filter_method,
        )
        pi = cg_result["energy_flux"].mean(dim=["along_track", "cross_track"]).values
        result[f"cg_{tag}"] = (pi, 1.0 / cg_result["scale"].values)

    # --- Fourier 1D: one "strip" per cross-track column, matching the SWOT pipeline ---
    Pi_list, T_list = [], []
    for ix in range(nx):
        Tk, Pik = compute_transfer_and_flux_for_strip(
            u[:, ix], v[:, ix], N1[:, ix], N2[:, ix], tukey_r=0.2,
        )
        if Tk is not None:
            Pi_list.append(Pik)
            T_list.append(Tk)
    Pi_fourier_1d = np.nanmean(np.array(Pi_list), axis=0)  # mean over strips, still one snapshot
    T_fourier_1d = np.nanmean(np.array(T_list), axis=0)    # same, but the transfer density itself
    # (needed, not Pi, for the Abel inversion below -- see abel_invert_fourier_1d)
    nk = len(Pi_fourier_1d)
    ny_fft = nk * 2 - 2  # rfft length -> original along-track length used
    k_fourier_1d = np.fft.rfftfreq(ny_fft, d=np.nanmean(dy_m) / 1000.0)  # cpkm
    result["fourier_1d"] = (Pi_fourier_1d, k_fourier_1d)
    result["fourier_1d_T"] = T_fourier_1d  # shares k_fourier_1d, not a (Pi, k) pair itself

    # --- Fourier 2D: genuinely isotropic 2D FFT on the whole patch at once ---
    k_fourier_2d, _T_ring, Pi_fourier_2d = compute_transfer_and_flux_2d(
        u, v, N1, N2, np.nanmean(dx_m), np.nanmean(dy_m), tukey_r=0.2,
    )
    result["fourier_2d"] = (Pi_fourier_2d, k_fourier_2d)

    return result


def abel_invert_fourier_1d(k, T, smooth_window=5, polyorder=1):
    """
    Convert a 1D along-track Fourier transfer density T(k) into an
    isotropic-2D-equivalent Pi(Kh), via invert_abel_A6 (swot_func/
    spectralflux_utils.py). Meant as a comparison against fourier_2d /
    cg_2d when you only trust (or only have) the 1D along-track estimate --
    see the "1D METHOD ONLY..." discussion in swot_swath_pipeline.py's
    module docstring for why a 1D along-track T(k)/Pi(k) and a genuine 2D
    isotropic one are NOT the same calculation and generally don't agree
    exactly; this is a validated way to bridge between them, not a
    replacement for actually computing the 2D version when you can.

    k, T: 1D arrays, same shape, increasing k -- pass T (the transfer
    density, e.g. fourier_1d_T from process_one_snapshot), NOT Pi (the
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


# Grid geometry (lat/lon, hence dx/dy) is fixed across the synthetic
# snapshots here, so compute it once. If your HYCOM grid genuinely changes
# per time step, move this inside the loop instead.
_, lat0, lon0 = make_synthetic_ssh(seed=0)
dy_m, dx_m = compute_dx_dy(lat0, lon0)
print(f"Grid spacing: dx={np.nanmean(dx_m):.0f} m, dy={np.nanmean(dy_m):.0f} m")

# --- REPLACE this loop with one over your own HYCOM time steps, loading
# ssh, lat, lon for each and calling
# process_one_snapshot(ssh, lat, lon, dx_m, dy_m) ---
N_SNAPSHOTS = 20  # like averaging over ~20 SWOT cycles -- more snapshots -> smoother Pi(k)
CURVE_NAMES = ("cg_1d", "cg_2d", "fourier_1d", "fourier_2d")
pi_lists = {name: [] for name in CURVE_NAMES}
k_arrays = {}
T_fourier_1d_list = []
for it in range(N_SNAPSHOTS):
    ssh, lat, lon = make_synthetic_ssh(seed=it)
    result = process_one_snapshot(ssh, lat, lon, dx_m, dy_m)
    for name in CURVE_NAMES:
        pi, k = result[name]
        pi_lists[name].append(pi)
        k_arrays[name] = k  # same k-grid every snapshot (fixed geometry) -- last write is fine
    T_fourier_1d_list.append(result["fourier_1d_T"])
    print(f"  snapshot {it + 1}/{N_SNAPSHOTS} done")
# --- end of loop to replace ---

pi_mean = {name: np.nanmean(pi_lists[name], axis=0) for name in CURVE_NAMES}

# Abel-invert the time-averaged 1D transfer density (not per-snapshot, and
# not the already-cumulative Pi -- see abel_invert_fourier_1d) into an
# isotropic-2D-equivalent estimate, for comparison against fourier_2d/cg_2d.
T_fourier_1d_mean = np.nanmean(T_fourier_1d_list, axis=0)
k_abel, pi_abel = abel_invert_fourier_1d(k_arrays["fourier_1d"], T_fourier_1d_mean)

ny, nx = ssh.shape
print(f"Input field: {ny} x {nx}, averaged over {N_SNAPSHOTS} snapshots")


# ---------------------------------------------------------------------------
# 3. Plot -- five curves: 1D and 2D, CG and Fourier, plus the Abel-inverted
#    isotropic-equivalent of the 1D Fourier estimate
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.axhline(0, color="gray", lw=0.8, ls="--")
styles = {
    "cg_1d":       ("o-", "CG $\\Pi_L$ (1D)"),
    "cg_2d":       ("o--", "CG $\\Pi_L$ (2D, isotropic)"),
    "fourier_1d":  ("s-", "Fourier $\\Pi(k)$ (1D)"),
    "fourier_2d":  ("s--", "Fourier $\\Pi(k)$ (2D, isotropic)"),
}
for name, (fmt, label) in styles.items():
    ax.plot(k_arrays[name], pi_mean[name] * 1e9, fmt, label=label)
ax.plot(k_abel, pi_abel * 1e9, "^:", label="Fourier $\\Pi(k)$ (1D, Abel-inverted)")
ax.set_xscale("log")
ax.set_xlabel("Wavenumber $k$ (cpkm)")
ax.set_ylabel(r"$\Pi$ (nW kg$^{-1}$)")
ax.set_title("Spectral energy flux from SSH: 1D vs 2D, CG vs Fourier")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "regular_grid_pipeline_output.png")
plt.savefig(out_path, dpi=130)
print(f"Saved figure: {out_path} -- open it to view (Agg backend, no display window)")
