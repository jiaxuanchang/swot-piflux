import numpy as np
import datetime
import scipy.signal as signal
import scipy.fft as fft
import xarray as xr
import os
import gc
import cv2
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter, uniform_filter1d, convolve
from scipy.signal import convolve2d

# Grid-geometry helpers and the geostrophic engine now live in grid_utils.py /
# geostrophy.py (shared with spectralflux_utils.py, avoids duplicating the
# SSH -> u,v derivation independently in each pipeline). Re-exported here so
# existing `from swot_func.coarse_graining import ...` call sites keep working.
from .grid_utils import g, omega, R_earth, report, assert_same_shape, haversine_km, _compute_cell_spacings
from .geostrophy import geostrophic_from_ssh


def filter_1d_spatial_butterworth(field, sca, Dx, Dy, axis=0, order=1):
    """
    1D zero-phase Butterworth low-pass filter along one axis.
    sca  : cutoff scale (meters)
    axis : 0 = along-track, 1 = cross-track
    """
    delta_mean = np.nanmean(Dy) if axis == 0 else np.nanmean(Dx)
    wn = np.clip(2.0 * delta_mean / sca, 0.001, 0.999)  # normalized cutoff; butter() requires (0, 1)
    b, a = signal.butter(order, wn, btype='low')
    return signal.filtfilt(b, a, field, axis=axis, padtype='odd')


def filter_2d_spatial_butterworth(field, sca, Dx, Dy, order=1):
    """
    2D zero-phase Butterworth low-pass filter (applied sequentially: Y then X).
    sca : cutoff scale (meters)
    """
    wn_y = np.clip(2.0 * np.nanmean(Dy) / sca, 0.001, 0.999)
    wn_x = np.clip(2.0 * np.nanmean(Dx) / sca, 0.001, 0.999)
    b_y, a_y = signal.butter(order, wn_y, btype='low')
    b_x, a_x = signal.butter(order, wn_x, btype='low')
    f_y = signal.filtfilt(b_y, a_y, field, axis=0, padtype='odd')
    return signal.filtfilt(b_x, a_x, f_y, axis=1, padtype='odd')


def filter_1d_spatial_gaussian(field, sca, Dx, Dy, axis=0):
    """
    1D Gaussian low-pass filter along one axis.
    sca  : cutoff scale L (meters); sigma = L / (2π * dx)
    axis : 0 = along-track, 1 = cross-track
    """
    sigma_y = sca / (2 * np.pi * np.nanmean(Dy)) if axis == 0 else 0.0
    sigma_x = sca / (2 * np.pi * np.nanmean(Dx)) if axis == 1 else 0.0
    return gaussian_filter(field, sigma=(sigma_y, sigma_x), mode='reflect')


def filter_2d_spatial_gaussian(field, sca, Dx, Dy):
    """
    2D Gaussian low-pass filter.
    sca : cutoff scale L (meters); sigma = L / (2π * dx)
    """
    sigma_y = sca / (2 * np.pi * np.nanmean(Dy))
    sigma_x = sca / (2 * np.pi * np.nanmean(Dx))
    return gaussian_filter(field, sigma=(sigma_y, sigma_x), mode='reflect')


def filter_1d_spatial_tophat(field, sca, Dx, Dy, axis=0):
    """
    1D top-hat (boxcar) filter along one axis.
    sca  : cutoff diameter (meters)
    axis : 0 = along-track, 1 = cross-track
    Boundary points within half the kernel width are set to NaN.
    """
    delta_mean = np.nanmean(Dy) if axis == 0 else np.nanmean(Dx)
    size = int(np.round(sca / delta_mean))
    if size % 2 == 0:
        size += 1
    size = max(size, 3)
    r = size // 2

    # uniform_filter1d uses a cumulative-sum: a single NaN propagates to ALL
    # downstream positions.  We fill NaN only when it is confined to leading/
    # trailing boundary rows — the only case where filling is safe (those rows
    # fall inside the explicit NaN margin set below and never enter the extracted
    # interior).  Interior NaN is left as-is so the filter result is honestly NaN
    # in that neighbourhood.
    field_safe = field
    if np.isnan(field).any():
        n = field.shape[axis]
        # Exclude perpendicular slices that are entirely NaN (e.g. cross-track border
        # columns left by the Tranchant kernel) from the boundary-NaN detection.
        # Those slices produce NaN in the filter output by themselves and must not
        # cause false "interior NaN" detection along the filter axis.
        perp_all_nan = np.isnan(field).all(axis=axis)
        if perp_all_nan.any():
            field_check = field[:, ~perp_all_nan] if axis == 0 else field[~perp_all_nan, :]
        else:
            field_check = field
        # find the leading and trailing NaN extent along the filter axis
        any_nan = np.isnan(field_check).any(axis=1 - axis)   # True for each slice with NaN
        lead  = int(np.argmin(any_nan))           # first non-NaN slice index
        trail = int(np.argmin(any_nan[::-1]))     # slices from the end that are NaN
        # Check: are ALL NaN confined to the leading / trailing bands?
        interior_nan = any_nan[lead : n - trail].any() if trail > 0 else any_nan[lead:].any()
        if not interior_nan:
            # Safe to fill boundary NaN with nearest valid slice
            field_safe = field.copy()
            if axis == 0:
                if lead  > 0: field_safe[:lead,  :] = field_safe[lead,  :]
                if trail > 0: field_safe[-trail:, :] = field_safe[-trail - 1, :]
            else:
                if lead  > 0: field_safe[:, :lead]  = field_safe[:, lead][:, None]
                if trail > 0: field_safe[:, -trail:] = field_safe[:, -trail - 1][:, None]
        else:
            # Interior NaN detected: do NOT fill (filling would hide real missing
            # data and introduce false values in the filter output).
            # uniform_filter1d's cumsum will propagate the NaN forward from the
            # first interior gap — warn so the caller knows why output is NaN.
            interior_rows = np.where(any_nan[lead : n - trail if trail > 0 else n])[0] + lead
            import warnings
            warnings.warn(
                f"filter_1d_spatial_tophat: interior NaN detected at positions "
                f"{interior_rows.tolist()} along axis {axis}. "
                f"Filter output will be NaN from row {interior_rows[0] - r} onward "
                f"(cumsum forward-propagation). Consider interpolating missing data first.",
                RuntimeWarning, stacklevel=2
            )
    filtered = uniform_filter1d(field_safe, size=size, axis=axis, mode='reflect')

    if axis == 0:
        filtered[:r, :] = np.nan
        filtered[-r:, :] = np.nan
    else:
        filtered[:, :r] = np.nan
        filtered[:, -r:] = np.nan

    return filtered


def _build_circular_kernel(sca, dy_mean):
    """
    Build a normalized circular top-hat convolution kernel.
    Size is forced odd so the kernel is symmetric about the center pixel.
    Returns (kernel, r) where r = size // 2 is the half-width in pixels.
    """
    r_pixel = (sca / 2.0) / dy_mean
    size = max(int(2 * r_pixel + 1), 3)
    if size % 2 == 0:
        size += 1
    # ogrid[-size//2 : size//2+1] gives exactly `size` points centered at 0
    y, x = np.ogrid[-size//2 : size//2 + 1, -size//2 : size//2 + 1]
    mask = (x**2 + y**2 <= r_pixel**2).astype(float)
    if mask.sum() == 0:
        mask[size//2, size//2] = 1.0
    return mask / mask.sum(), size // 2


def filter_2d_spatial_tophat(field, sca, Dx, Dy):
    """
    2D circular top-hat filter using OpenCV filter2D.
    sca : cutoff diameter (meters)
    Boundary points within half the kernel width are set to NaN.
    """
    kernel, r = _build_circular_kernel(sca, np.nanmean(Dy))
    filtered = cv2.filter2D(field, ddepth=-1, kernel=kernel,
                            borderType=cv2.BORDER_REFLECT_101)
    if r > 0:
        filtered[:r, :]  = np.nan
        filtered[-r:, :] = np.nan
        filtered[:, :r]  = np.nan
        filtered[:, -r:] = np.nan
    return filtered


def filter_2d_spatial_tophat_scipyndimage(field, sca, dx, dy):
    """
    2D circular top-hat filter using scipy.ndimage.convolve.
    sca : cutoff diameter (meters)
    """
    kernel, _ = _build_circular_kernel(sca, np.nanmean(dy))
    return convolve(field, kernel, mode='reflect')


def filter_2d_spatial_tophat_scipyconv2(field, sca, Dx, Dy):
    """
    2D circular top-hat filter using scipy.signal.convolve2d (float64).
    sca : cutoff diameter (meters)
    Boundary points within half the kernel width are set to NaN.
    """
    kernel, r = _build_circular_kernel(sca, np.nanmean(Dy))
    filtered = convolve2d(field.astype(np.float64), kernel.astype(np.float64),
                          mode='same', boundary='symm')
    if r > 0:
        filtered[:r, :]  = np.nan
        filtered[-r:, :] = np.nan
        filtered[:, :r]  = np.nan
        filtered[:, -r:] = np.nan
    return filtered


FILTER_REGISTRY = {
    "2d_spatial_gaussian":            filter_2d_spatial_gaussian,
    "2d_spatial_butterworth":         filter_2d_spatial_butterworth,
    "2d_spatial_tophat":              filter_2d_spatial_tophat,
    "2d_spatial_tophat_scipyndimage": filter_2d_spatial_tophat_scipyndimage,
    "2d_spatial_tophat_scipyconv2":   filter_2d_spatial_tophat_scipyconv2,
    "1d_spatial_gaussian":            filter_1d_spatial_gaussian,
    "1d_spatial_butterworth":         filter_1d_spatial_butterworth,
    "1d_spatial_tophat":              filter_1d_spatial_tophat,
    "1d_x_spatial_tophat":            lambda field, sca, Dx, Dy: filter_1d_spatial_tophat(field, sca, Dx, Dy, axis=1),
}


class SpectralFilterManager:
    """
    FFT-based spectral filter manager for SFS energy flux computation.

    FFTs of all required fields are computed once at initialization.
    get_filtered_field() applies different cutoff scales without re-transforming.

    Supports:
      - 2D spectral filters (method name contains "2d_spectral")
      - 1D directional filters (axis=0 along-track, axis=1 cross-track)

    Windowing: a Tukey window is applied once to both linear (u, v) and
    quadratic (uu, vv, uv) terms after per-axis demeaning, so all terms
    see the same window weight (avoids W² bias on quadratic terms).
    """
    def __init__(self, method_name, u_prime, v_prime, Dx, Dy, axis=0, alpha=0.2):
        self.method_name = method_name
        self.axis = axis
        self.dx = np.nanmean(Dx)
        self.dy = np.nanmean(Dy)
        self.ny, self.nx = u_prime.shape

        if "2d_spectral" in method_name:
            self.kx = 2.0 * np.pi * fft.fftfreq(self.nx, d=self.dx)
            self.ky = 2.0 * np.pi * fft.fftfreq(self.ny, d=self.dy)
            self.KX, self.KY = np.meshgrid(self.kx, self.ky)
            self.k_mag = np.sqrt(self.KX**2 + self.KY**2)
            self.is_2d_fft = True

            W_raw = np.outer(tukey(self.ny, alpha=alpha), tukey(self.nx, alpha=alpha))
            W2d = W_raw / np.sqrt(np.mean(W_raw**2))  # energy-corrected window

            # Apply W once to linear and quadratic terms for consistent treatment
            u_p  = u_prime * W2d
            v_p  = v_prime * W2d
            uu_p = (u_prime ** 2) * W2d
            vv_p = (v_prime ** 2) * W2d
            uv_p = (u_prime * v_prime) * W2d

            self.FFT_DATA = {
                "u":  fft.fft2(u_p),
                "v":  fft.fft2(v_p),
                "uu": fft.fft2(uu_p),
                "vv": fft.fft2(vv_p),
                "uv": fft.fft2(uv_p),
            }

        else:
            self.is_2d_fft = False

            if axis == 0:
                self.k = 2.0 * np.pi * fft.fftfreq(self.ny, d=self.dy)
                W_raw = tukey(self.ny, alpha=alpha)
                # Per-column demeaning to reduce spectral leakage along the filter axis
                u_col = u_prime - u_prime.mean(axis=0, keepdims=True)
                v_col = v_prime - v_prime.mean(axis=0, keepdims=True)
                W_bc = W_raw[:, np.newaxis]
            else:
                self.k = 2.0 * np.pi * fft.fftfreq(self.nx, d=self.dx)
                W_raw = tukey(self.nx, alpha=alpha)
                # Per-row demeaning
                u_col = u_prime - u_prime.mean(axis=1, keepdims=True)
                v_col = v_prime - v_prime.mean(axis=1, keepdims=True)
                W_bc = W_raw[np.newaxis, :]

            W_bc = W_bc / np.sqrt(np.mean(W_raw**2))  # energy-corrected window

            # Apply W once to linear and quadratic terms for consistent treatment
            u_p  = u_col * W_bc
            v_p  = v_col * W_bc
            uu_p = (u_col ** 2) * W_bc
            vv_p = (v_col ** 2) * W_bc
            uv_p = (u_col * v_col) * W_bc

            self.FFT_DATA = {
                "u":  fft.fft(u_p,  axis=self.axis),
                "v":  fft.fft(v_p,  axis=self.axis),
                "uu": fft.fft(uu_p, axis=self.axis),
                "vv": fft.fft(vv_p, axis=self.axis),
                "uv": fft.fft(uv_p, axis=self.axis),
            }

    def get_filtered_field(self, field_name, L):
        """Apply spectral low-pass at cutoff scale L (meters); return real-space result."""
        kc = 2.0 * np.pi / L

        if "gaussian" in self.method_name:
            if self.is_2d_fft:
                filter_mask = np.exp(-0.5 * (self.k_mag / kc)**2)
            else:
                raw_mask = np.exp(-0.5 * (self.k / kc)**2)
                filter_mask = raw_mask[:, None] if self.axis == 0 else raw_mask[None, :]
        elif "boxcar" in self.method_name:
            if self.is_2d_fft:
                filter_mask = (self.k_mag <= kc).astype(float)
            else:
                raw_mask = (np.abs(self.k) <= kc).astype(float)
                filter_mask = raw_mask[:, None] if self.axis == 0 else raw_mask[None, :]
        else:
            raise ValueError(f"Unknown spectral filter method: {self.method_name}")

        filtered_fft = self.FFT_DATA[field_name] * filter_mask

        if self.is_2d_fft:
            return np.real(fft.ifft2(filtered_fft))
        else:
            return np.real(fft.ifft(filtered_fft, axis=self.axis))


def compute_scales_and_tophat_margin(Lmin_m, Lmax_m, sca, filter_method, Dx, Dy):
    """
    Log-spaced cutoff scales, plus the boundary margin (my, mx, in pixels) a
    top-hat filter needs, sized to the LARGEST SCALE ACTUALLY USED
    (scales[-1]) so all scales share the same spatial domain -- not the
    requested Lmax_m upper bound, which the geometric sequence (Lmin*sca^i,
    i truncated by int()) essentially never hits exactly. Sizing the kernel
    to Lmax_m instead of scales[-1] always over-trims (scales[-1] <= Lmax_m
    always, so this is a strict improvement: never trims more than before,
    sometimes noticeably less -- e.g. sca=1.2, Lmin=4km, Lmax=1200km lands
    scales[-1] ~=1007km, ~16% smaller kernel than sizing to the full 1200km).
    1D filters trim only along their axis. Non-tophat (spectral)
    filter_methods return my=mx=0.

    Shared by compute_sfs_energy_flux (CG) and spectralflux_utils.py's
    compute_for_pass (Fourier, to size its Tukey window so its flat-top
    region matches CG's output domain for the same Lmin_m/Lmax_m/sca/
    filter_method) -- keeping this logic in one place is what makes that
    cross-pipeline alignment reliable: previously it was computed
    independently in run_sfs_flux_pipeline too and silently drifted out of
    sync with this function's own margin when this function switched from
    Lmax_m to scales[-1] (fixed 2026-09-07 by having run_sfs_flux_pipeline
    read the margin back out of this function's own output shape instead).

    Returns: scales (1D array, meters), my, mx (int, pixels)
    """
    nscale = int(np.log(Lmax_m / Lmin_m) / np.log(sca))
    scales = np.array([Lmin_m * (sca ** i) for i in range(nscale)])
    L_margin = scales[-1] if len(scales) > 0 else Lmax_m

    my, mx = 0, 0
    if "tophat" in filter_method:
        if "1d" in filter_method:
            if "1d_x" in filter_method:   # cross-track (x) filter
                _sz = max(int(np.round(L_margin / np.nanmean(Dx))), 3)
                if _sz % 2 == 0:
                    _sz += 1
                mx = _sz // 2
            else:                          # along-track (y) filter
                _sz = max(int(np.round(L_margin / np.nanmean(Dy))), 3)
                if _sz % 2 == 0:
                    _sz += 1
                my = _sz // 2
        else:
            _, r_max = _build_circular_kernel(L_margin, np.nanmean(Dy))
            my = mx = r_max
    return scales, my, mx


def cg_final_along_track_length(L_raw, Lmin_m, Lmax_m, sca, filter_method, Dy_mean):
    """
    What CG's final along-track output length would be for a raw SSH input
    of L_raw lines, WITHOUT actually running geostrophic_from_ssh /
    compute_sfs_energy_flux -- just the two margin steps those apply along
    the way: geostrophic_from_ssh crops 1 pixel off each end (-2), then
    compute_sfs_energy_flux/run_sfs_flux_pipeline crop `1 + my` off each end
    for the gradient + filter margin (-2 - 2*my), where `my` comes from
    compute_scales_and_tophat_margin. Total: L_raw - 4 - 2*my.

    Dy_mean: mean along-track grid spacing (meters). Only the along-track
    margin matters here (Dx is irrelevant unless filter_method is a
    cross-track '1d_x' variant), so Dy_mean is passed for both the Dx and
    Dy arguments of compute_scales_and_tophat_margin.

    Used by spectralflux_utils.py's compute_for_pass to size its Tukey
    window so its flat-top region matches this length for the same
    Lmin_m/Lmax_m/sca/filter_method -- see that function's docstring for why
    (apples-to-apples Pi_L vs Pi(k) comparison against CG).
    """
    _, my, _ = compute_scales_and_tophat_margin(Lmin_m, Lmax_m, sca, filter_method,
                                                  Dy_mean, Dy_mean)
    return L_raw - 4 - 2 * my


def compute_sfs_energy_flux(ug, vg, lon2d, lat2d, Dx, Dy,
                            sca=1.2, Lmin_m=1e3, Lmax_m=1e7,
                            debug=False,
                            filter_method=None):
    """
    Compute Sub-Filter-Scale (SFS) energy flux Pi_L across a range of scales.

    Parameters:
      ug, vg       : geostrophic velocity fields (Ny, Nx)
      lon2d, lat2d : coordinate arrays (Ny, Nx)
      Dx, Dy       : grid spacing arrays (Ny, Nx) in meters
      sca          : scale growth factor (e.g. 1.2 gives ~17 scales per decade)
      Lmin_m       : minimum cutoff scale (meters)
      Lmax_m       : maximum cutoff scale (meters)
      filter_method: key from FILTER_REGISTRY or spectral method name

    Returns:
      xr.Dataset with energy_flux and filtered fields at each scale (nscale, Ny-2, Nx-2)
    """
    Ny, Nx = ug.shape

    if debug:
        Ly_est = float(np.nanmean(np.nansum(Dy, axis=0)))
        Lx_est = float(np.nanmean(np.nansum(Dx, axis=1)))
        print(f"[SFS] Input: Ny={Ny}, Nx={Nx}, domain ~{Ly_est/1e3:.1f}x{Lx_est/1e3:.1f} km")

    # Global demeaning before filtering
    u_prime = ug - np.nanmean(ug)
    v_prime = vg - np.nanmean(vg)

    if "spectral" in filter_method:
        spec_manager = SpectralFilterManager(filter_method, u_prime, v_prime, Dx, Dy, axis=0)
        u_filter  = lambda L: spec_manager.get_filtered_field("u",  L)
        v_filter  = lambda L: spec_manager.get_filtered_field("v",  L)
        uu_filter = lambda L: spec_manager.get_filtered_field("uu", L)
        vv_filter = lambda L: spec_manager.get_filtered_field("vv", L)
        uv_filter = lambda L: spec_manager.get_filtered_field("uv", L)
    else:
        if filter_method not in FILTER_REGISTRY:
            raise ValueError(f"Unknown filter method: {filter_method}")
        spatial_func = FILTER_REGISTRY[filter_method]
        uu_prime = u_prime ** 2
        vv_prime = v_prime ** 2
        uv_prime = u_prime * v_prime
        u_filter  = lambda L: spatial_func(u_prime,  L, Dx, Dy)
        v_filter  = lambda L: spatial_func(v_prime,  L, Dx, Dy)
        uu_filter = lambda L: spatial_func(uu_prime, L, Dx, Dy)
        vv_filter = lambda L: spatial_func(vv_prime, L, Dx, Dy)
        uv_filter = lambda L: spatial_func(uv_prime, L, Dx, Dy)

    scales, my, mx = compute_scales_and_tophat_margin(Lmin_m, Lmax_m, sca, filter_method, Dx, Dy)
    nscale = len(scales)

    Ny_c, Nx_c = Ny - 2 - 2 * my, Nx - 2 - 2 * mx
    pi_3d     = np.zeros((nscale, Ny_c, Nx_c))
    u_bar_3d  = np.zeros((nscale, Ny_c, Nx_c))
    v_bar_3d  = np.zeros((nscale, Ny_c, Nx_c))
    uu_bar_3d = np.zeros((nscale, Ny_c, Nx_c))
    vv_bar_3d = np.zeros((nscale, Ny_c, Nx_c))
    uv_bar_3d = np.zeros((nscale, Ny_c, Nx_c))

    # Slices for extracting the valid interior (computed once, shared by all scales)
    py0, py1 = my, (-my if my > 0 else None)
    px0, px1 = mx, (-mx if mx > 0 else None)
    uy0 = 1 + my;  uy1 = -(1 + my) if my > 0 else -1
    ux0 = 1 + mx;  ux1 = -(1 + mx) if mx > 0 else -1

    for s_idx, L in enumerate(scales):
        u_bar  = u_filter(L)
        v_bar  = v_filter(L)
        uu_bar = uu_filter(L)
        vv_bar = vv_filter(L)
        uv_bar = uv_filter(L)

        # SFS stress tensor: tau_ij = bar(u_i u_j) - bar(u_i) bar(u_j)
        tau_uu = uu_bar - u_bar ** 2
        tau_vv = vv_bar - v_bar ** 2
        tau_uv = uv_bar - u_bar * v_bar

        # Velocity gradients via centered differences on interior
        du_dy = (u_bar[2:, 1:-1] - u_bar[:-2, 1:-1]) / (2.0 * Dy[1:-1, 1:-1])
        du_dx = (u_bar[1:-1, 2:] - u_bar[1:-1, :-2]) / (2.0 * Dx[1:-1, 1:-1])
        dv_dy = (v_bar[2:, 1:-1] - v_bar[:-2, 1:-1]) / (2.0 * Dy[1:-1, 1:-1])
        dv_dx = (v_bar[1:-1, 2:] - v_bar[1:-1, :-2]) / (2.0 * Dx[1:-1, 1:-1])

        # Pi_L = -tau_ij S_ij (energy flux to sub-filter scales)
        pi_l = -(tau_uu[1:-1, 1:-1] * du_dx
                 + tau_uv[1:-1, 1:-1] * (du_dy + dv_dx)
                 + tau_vv[1:-1, 1:-1] * dv_dy)

        pi_3d[s_idx]     = pi_l[py0:py1, px0:px1]
        u_bar_3d[s_idx]  = u_bar[uy0:uy1, ux0:ux1]
        v_bar_3d[s_idx]  = v_bar[uy0:uy1, ux0:ux1]
        uu_bar_3d[s_idx] = uu_bar[uy0:uy1, ux0:ux1]
        vv_bar_3d[s_idx] = vv_bar[uy0:uy1, ux0:ux1]
        uv_bar_3d[s_idx] = uv_bar[uy0:uy1, ux0:ux1]

    lat2d_c = lat2d[uy0:uy1, ux0:ux1]
    lon2d_c = lon2d[uy0:uy1, ux0:ux1]

    ds = xr.Dataset(
        data_vars={
            "energy_flux": (("scale", "along_track", "cross_track"), pi_3d),
            "u_bar":       (("scale", "along_track", "cross_track"), u_bar_3d),
            "v_bar":       (("scale", "along_track", "cross_track"), v_bar_3d),
            "uu_bar":      (("scale", "along_track", "cross_track"), uu_bar_3d),
            "vv_bar":      (("scale", "along_track", "cross_track"), vv_bar_3d),
            "uv_bar":      (("scale", "along_track", "cross_track"), uv_bar_3d),
        },
        coords={
            "scale": scales / 1000.0,  # convert to km
            "lat":   (("along_track", "cross_track"), lat2d_c),
            "lon":   (("along_track", "cross_track"), lon2d_c),
        }
    )

    return ds

