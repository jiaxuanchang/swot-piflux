"""
Shared SSH -> geostrophic-velocity engine, used by both coarse_graining.py
(CG) and spectralflux_utils.py (Fourier). Previously each pipeline had its
own independent implementation of "call Tranchant / do a stencil diff, then
apply f = 2*Omega*sin(lat)"; consolidated here so there's exactly one for
each derivative method.

geostrophic_from_ssh() and geostrophic_advection_from_ssh() are NOT the same
function despite both wrapping Tranchant: they use different border-crop
margins after the Tranchant fit (1 pixel here vs. n//2+1 in the advection
version), because they're tuned for different downstream consumers -- see
each function's docstring. Do not merge them without re-validating both
pipelines' output.
"""
import numpy as np

from .grid_utils import g, omega, R_earth, report, assert_same_shape, _compute_cell_spacings


def geostrophic_from_ssh(ssh2d, lat2d, lon2d, Dx=None, Dy=None, stencil=3,
                          method='stencil', tranchant_kwargs=None, debug=False):
    """
    Compute geostrophic velocities from SSH. Used by the coarse-graining (CG)
    pipeline (coarse_graining.py::run_sfs_flux_pipeline).

    method='stencil' (default)
      Variable-spacing stencil differentiation in SWOT track coordinates
      (along-track / cross-track).  stencil=3 or 7.
      When Dx and Dy are both provided together with stencil=3, they are used
      directly (backward-compatible path for model data with uniform spacing).

    method='tranchant'
      Surface-fit kernel method from swot_func.Tranchant (Tranchant et al. 2024).
      Derivatives are computed by fitting a 2D polynomial in a circular kernel,
      then ROTATED from SWOT track coordinates to geographic (East / North) frame
      using the local track angle θ = compute_angle(lon, lat).
      Requires swot_func to be on sys.path.
      tranchant_kwargs : dict passed to compute_ocean_diagnostics_from_eta.
        Defaults: derivative='fit', n=13, min_valid_points=0.75,
                  cyclostrophy='GW', second_derivative='dxdy',
                  kernel='circular', verbose=False

      Only crops a fixed 1 pixel off each border (to match the stencil
      branch's (L-2, P-2) output-shape contract), even though the Tranchant
      fit kernel leaves a wider NaN border (~n//2 pixels). The leftover
      along-track NaN border rows are filled inward instead of cropped away
      (see comment below); this is fine for CG since its top-hat filter is a
      local spatial operation. This is NOT what
      geostrophic_advection_from_ssh (used by the Fourier pipeline) does --
      that one crops the full NaN margin away instead of filling it, because
      the FFT-based Fourier flux is much more sensitive to residual
      fill-value artifacts at the edges than CG's local filter is.

    Inputs : ssh2d, lat2d, lon2d  (L, P) numpy arrays
    Returns: ug, vg, f_c, beta_c, Dx, Dy, lat_c, lon_c — all (L-2, P-2)
    """
    f    = 2.0 * omega * np.sin(np.radians(lat2d))
    beta = 2.0 * omega * np.cos(np.radians(lat2d)) / R_earth
    f_c    = f[1:-1, 1:-1]
    beta_c = beta[1:-1, 1:-1]
    lat_c  = lat2d[1:-1, 1:-1]
    lon_c  = lon2d[1:-1, 1:-1]

    # ── Tranchant method: fit-based derivates + geographic frame rotation ──
    if method == 'tranchant':
        try:
            from swot_func.Tranchant.diagnosis import compute_ocean_diagnostics_from_eta
        except ImportError:
            raise ImportError(
                "swot_func.Tranchant not importable. Add the repo root to sys.path "
                "or use method='stencil'.")
        import xarray as xr
        kw = dict(derivative='fit', n=13, min_valid_points=0.75,
                  cyclostrophy='GW', avoid_negative=False,
                  second_derivative='dxdy', kernel='circular', verbose=False)
        if tranchant_kwargs:
            kw.update(tranchant_kwargs)

        ssh_xr = xr.DataArray(ssh2d, dims=('along_track', 'cross_track'))
        diag = compute_ocean_diagnostics_from_eta(ssh_xr, lon2d, lat2d, **kw)

        ug = np.array(diag['ug'])[1:-1, 1:-1].copy()
        vg = np.array(diag['vg'])[1:-1, 1:-1].copy()

        # The Tranchant kernel leaves NaN at the border rows AND cols of the trimmed
        # array.  Along-track border rows (row 0 and row -1) fall inside the tophat
        # margin and must be filled so the filter's cumsum doesn't propagate NaN
        # into the interior.  Cross-track border cols (col 0 and col -1) are left
        # as NaN intentionally: filter_1d_spatial_tophat excludes them from boundary
        # detection (they are entirely NaN), the filter outputs NaN for those columns,
        # and that NaN propagates to the outermost pi_l pixels — masking the spurious
        # edge signals caused by finite-difference gradients hitting fill values.
        n_kernel = kw.get('n', 13)
        for _ in range(max(1, n_kernel // 2)):
            for arr in (ug, vg):
                arr[0, :]  = np.where(np.isnan(arr[0, :]),  arr[1, :],  arr[0, :])
                arr[-1, :] = np.where(np.isnan(arr[-1, :]), arr[-2, :], arr[-1, :])

        # Physical cell spacings for downstream gradient computation (same as stencil path)
        atrack_sp, ctrack_sp = _compute_cell_spacings(lat2d, lon2d)
        Dy = 0.5 * (atrack_sp[:-2, 1:-1] + atrack_sp[1:-1, 1:-1])  # along-track
        Dx = 0.5 * (ctrack_sp[1:-1, :-2] + ctrack_sp[1:-1, 1:-1])  # cross-track

        if debug:
            report("ug (tranchant)", ug)
            report("vg (tranchant)", vg)

        return ug, vg, f_c, beta_c, Dx, Dy, lat_c, lon_c

    # ── Stencil method ──
    if Dx is not None and Dy is not None and stencil == 3:
        # Backward-compatible path: use caller-supplied spacing with simple centered diff
        dssh_dy = (ssh2d[2:, 1:-1] - ssh2d[:-2, 1:-1]) / (2.0 * Dy)
        dssh_dx = (ssh2d[1:-1, 2:] - ssh2d[1:-1, :-2]) / (2.0 * Dx)
    else:
        # Compute haversine forward spacings for each adjacent cell pair
        atrack_sp, ctrack_sp = _compute_cell_spacings(lat2d, lon2d)
        # Local spacing at interior points = mean of the two adjacent forward spacings
        Dy = 0.5 * (atrack_sp[:-2, 1:-1] + atrack_sp[1:-1, 1:-1])   # (L-2, P-2) along-track
        Dx = 0.5 * (ctrack_sp[1:-1, :-2] + ctrack_sp[1:-1, 1:-1])    # (L-2, P-2) cross-track

        if stencil == 3:
            # 3-point variable-spacing central difference (vectorised)
            dssh_dy = (ssh2d[2:, 1:-1] - ssh2d[:-2, 1:-1]) / (2.0 * Dy)
            dssh_dx = (ssh2d[1:-1, 2:] - ssh2d[1:-1, :-2]) / (2.0 * Dx)

        elif stencil == 7:
            # 7-point high-order central difference with variable spacing (vectorised)
            # Coefficients: [-1, 9, -45, 0, 45, -9, 1] / 60
            _c = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0
            s  = np.nan_to_num(ssh2d, nan=0.0)
            ny, nx = ssh2d.shape

            # Along-track gradient (axis-0): valid at rows 3..ny-4
            cum_a  = np.vstack([np.zeros((1, nx)),
                                 np.cumsum(np.nan_to_num(atrack_sp), axis=0)])  # (ny+1, nx)
            dxm_a  = (cum_a[6:] - cum_a[:-6]) / 6.0          # (ny-5, nx)
            g0_num = (_c[0]*s[:-6] + _c[1]*s[1:-5] + _c[2]*s[2:-4]
                      + _c[4]*s[4:-2] + _c[5]*s[5:-1] + _c[6]*s[6:])  # (ny-6, nx)
            g0 = np.full((ny, nx), np.nan)
            g0[3:-3] = g0_num / dxm_a[:-1]

            # Cross-track gradient (axis-1): valid at cols 3..nx-4
            cum_c  = np.hstack([np.zeros((ny, 1)),
                                  np.cumsum(np.nan_to_num(ctrack_sp), axis=1)])  # (ny, nx+1)
            dxm_c  = (cum_c[:, 6:] - cum_c[:, :-6]) / 6.0    # (ny, nx-5)
            g1_num = (_c[0]*s[:, :-6] + _c[1]*s[:, 1:-5] + _c[2]*s[:, 2:-4]
                      + _c[4]*s[:, 4:-2] + _c[5]*s[:, 5:-1] + _c[6]*s[:, 6:])  # (ny, nx-6)
            g1 = np.full((ny, nx), np.nan)
            g1[:, 3:-3] = g1_num / dxm_c[:, :-1]

            dssh_dy = g0[1:-1, 1:-1]   # (L-2, P-2) along-track gradient
            dssh_dx = g1[1:-1, 1:-1]   # (L-2, P-2) cross-track gradient
        else:
            raise ValueError(f"stencil must be 3 or 7, got {stencil}")

    assert_same_shape(dssh_dy, dssh_dx, "dssh_dy", "dssh_dx")
    assert_same_shape(dssh_dy, f_c, "dssh_dy", "f_c")
    assert np.isfinite(Dx).any() and np.isfinite(Dy).any(), "Dx/Dy all-NaN!"

    if debug:
        report("|f|", np.abs(f_c))
        print("[check] any |f| < 1e-6 ? ->", np.any(np.abs(f_c) < 1e-6))
        report("Dx (m)", Dx)
        report("Dy (m)", Dy)

    ug = -(g / f_c) * dssh_dy
    vg =  (g / f_c) * dssh_dx

    if debug:
        report("ug", ug)
        report("vg", vg)

    return ug, vg, f_c, beta_c, Dx, Dy, lat_c, lon_c


def geostrophic_advection_from_ssh(ssha, lon2d, lat2d, n=5, min_valid_points=0.75,
                                    tranchant_kwargs=None):
    """
    Compute geostrophic velocity AND the nonlinear advection term (N1, N2)
    from SSH via the Tranchant fitting-kernel method. Used by the Fourier
    pipeline (spectralflux_utils.py::compute_for_pass) -- extracted from
    there verbatim, behavior unchanged.

    One Tranchant fit gives both the velocity and SSH's second derivatives
    directly, from which N1, N2 are computed analytically (ug*ux+vg*uy,
    ug*vx+vg*vy via geostrophic balance) rather than by re-differentiating
    velocity a second, noisier time.

    Crops ntrim = n//2+1 pixels off each border -- the full width of the
    Tranchant kernel's NaN margin, plus one -- rather than
    geostrophic_from_ssh's fixed 1-pixel crop (see that function's
    docstring for why: this pipeline is FFT-based and cannot tolerate
    residual fill-value artifacts at the edges the way CG's local top-hat
    filter can, so it crops the NaN margin away fully instead of filling it).

    Returns: ug, vg, N1, N2 (each (L - 2*ntrim, P - 2*ntrim)), ntrim
    """
    from swot_func.Tranchant.diagnosis import compute_ocean_diagnostics_from_eta

    params = dict(derivative='fit', n=n, min_valid_points=min_valid_points,
                  cyclostrophy='GW', avoid_negative=False,
                  second_derivative='dxdy', kernel='circular')
    if tranchant_kwargs:
        params.update(tranchant_kwargs)

    f = 2 * omega * np.sin(np.deg2rad(lat2d))
    diag = compute_ocean_diagnostics_from_eta(ssha, lon2d, lat2d, **params)

    ug = diag["ug"]
    vg = diag["vg"]
    ux = -(g/f) * diag['dxy'];  uy = -(g/f) * diag['dyy']
    vx =  (g/f) * diag['dxx'];  vy =  (g/f) * diag['dxy']
    N1 = ug*ux + vg*uy
    N2 = ug*vx + vg*vy

    ntrim = n // 2 + 1
    sl = slice(ntrim, -ntrim)
    # Match the original inline code's types exactly: ug/vg are left as
    # whatever compute_ocean_diagnostics_from_eta returned them as (plain
    # ndarrays), N1/N2 are explicitly pulled to .values (they come out as
    # xarray DataArrays from the ug*ux+vg*uy product, since ux/uy/vx/vy are
    # DataArrays via diag['dxy'] etc.).
    return ug[sl, sl], vg[sl, sl], N1[sl, sl].values, N2[sl, sl].values, ntrim
