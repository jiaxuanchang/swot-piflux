import numpy as np
from scipy.signal import savgol_filter, detrend


def tukey_1d_custom(N, r=0.2):
    if r <= 0:
        return np.ones(N)
    t = np.linspace(0, 1, N)
    t_half = t[:N//2]
    window_half = 0.5 * (1 + np.cos(np.pi * (2 * t_half / r - 1)))

    # Set center part to 1
    cutoff = int(np.floor(r * (N - 1) / 2)) + 1
    window_half[cutoff:] = 1

    # Reconstruct full window
    if N % 2 == 0:
        window = np.concatenate([window_half, window_half[::-1]])
    else:
        window = np.concatenate([window_half, [1], window_half[::-1]])

    return window


def tukey2d(ny, nx, r=0.2):
    """Separable 2D Tukey window with shape (nx, ny)."""
    wx = tukey_1d_custom(nx, r)
    wy = tukey_1d_custom(ny, r)
    return np.outer(wy, wx)


def build_k_edges(kx, ky, nb=60, spacing="geom", kmin=None, kmax=None):
    """
        Build radial wavenumber bin edges based on frequency axes kx, ky ([rad/m], as produced by np.fft.fftfreq).
        - spacing: "geom" for geometric binning (recommended) or "linear"
        - Returns: edges ([rad/m], length nb+1)
        """
    kx = np.asarray(kx); ky = np.asarray(ky)
    # Take the smallest nonzero |kx|, |ky| as the fundamental wavenumber; avoid 0 causing geometric binning failure
    def _min_pos(a):
        a = np.unique(np.abs(a))
        a = a[a > 0]
        return a.min() if a.size else 0.0

    kmin_def = min(_min_pos(kx), _min_pos(ky))
    kxmax = np.max(np.abs(kx)) if kx.size else 0.0
    kymax = np.max(np.abs(ky)) if ky.size else 0.0
    # Largest radius at the corners
    corner_radius = np.sqrt(kxmax**2 + kymax**2)
    kmax_def = corner_radius

    eps = 0.5
    kmin = float(kmin) if kmin is not None else (eps * kmin_def if kmin_def > 0 else kmax_def/ (1000.0*nb))
    kmax = float(kmax) if kmax is not None else kmax_def
    kmax = max(kmax, kmin*(1+1e-6))

    if spacing == "geom":
        edges = np.geomspace(kmin, kmax, nb + 1)
    elif spacing == "linear":
        edges = np.linspace(kmin, kmax, nb + 1)
    else:
        raise ValueError("spacing must be 'geom' or 'linear'")

    return edges


def ring_sum_and_flux(T2D, kx, ky, edges, shifted=False, window_power_mean=None):
    """
        Unified shell integration and threshold flux definition.
        Parameters
        ----------
        T2D : (ny, nx) 2D transfer field (e.g., Re[conj(ψ̂)Ĵ]/(N^2)), units handled by user
        kx, ky : 1D frequency axes [rad/m] (aligned with T2D axes)
        edges : radial k bin edges [rad/m], increasing, length nb+1
        shifted : If T2D/kx/ky are sorted by fftshift, set True; otherwise False
        window_power_mean : If windowing was applied, provide mean(W^2) for Parseval correction; otherwise None

        Returns
        -------
        out : dict, containing
          - 'k_mid'        : geometric mean of each shell [rad/m]
          - 'q_cpkm'       : midpoints converted to cycles per km
          - 'T_ring'       : total T in each shell (sum-over-modes)
          - 'Pi'           : threshold flux Π(k) = Σ_{|k|≥k_edge_i} T
          - 'edges'        : original edges [rad/m]
          - 'edges_cpkm'   : edges converted to cpkm
        """
    T = np.asarray(T2D, float)
    kx = np.asarray(kx, float)
    ky = np.asarray(ky, float)
    edges = np.asarray(edges, float)

    if shifted:
        #  shift the ordering back to fftfreq style
        T = np.fft.ifftshift(T)
        kx = np.fft.ifftshift(kx)
        ky = np.fft.ifftshift(ky)

    # radial wavenumber grid
    KX, KY = np.meshgrid(kx, ky, indexing='xy')  # 形狀 (ny, nx)
    KR = np.sqrt(KX**2 + KY**2)

    # 1D for binning
    KRv = KR.ravel()
    Tv  = T.ravel()

    # remove NaN/Inf
    m = np.isfinite(KRv) & np.isfinite(Tv)
    KRv = KRv[m]; Tv = Tv[m]

    # binning
    which = np.digitize(KRv, edges) - 1  # 0..nb-1 falls [edge_i, edge_{i+1})
    nb = len(edges) - 1
    T_ring = np.zeros(nb, float)
    for i in range(nb):
        sel = which == i
        if np.any(sel):
            T_ring[i] = Tv[sel].sum()

    # cumulative flux over radial thresholds (discrete cumulative from high-k to low-k)
    Pi = np.cumsum(T_ring[::-1])[::-1]

    # correct for windowing loss
    if window_power_mean is not None and np.isfinite(window_power_mean) and window_power_mean > 0:
        T_ring = T_ring / window_power_mean
        Pi     = Pi / window_power_mean

    # midpoints
    k_mid = np.sqrt(edges[:-1] * edges[1:])

    # convert to cpkm
    def to_cpkm(k_rad_m):
        return (k_rad_m / (2*np.pi)) * 1000.0

    q_cpkm = to_cpkm(k_mid)
    edges_cpkm = to_cpkm(edges)

    return dict(
        k_mid=k_mid,
        q_cpkm=q_cpkm,
        T_ring=T_ring,
        Pi=Pi,
        edges=edges,
        edges_cpkm=edges_cpkm
    )


# -------------------------
#  API
# -------------------------

def unified_flux_postprocess(T2D, kx, ky, nb=60, spacing="geom",
                             shifted=False, window_power_mean=None,
                             kmin=None, kmax=None):
    """
        One-step postprocessing: build bin edges -> shell sum -> threshold flux -> axis conversion
        """
    edges = build_k_edges(kx, ky, nb=nb, spacing=spacing, kmin=kmin, kmax=kmax)
    return ring_sum_and_flux(T2D, kx, ky, edges, shifted=shifted,
                             window_power_mean=window_power_mean)


def _tukey_flat_top_len(M, r):
    """Number of samples in tukey_1d_custom(M, r) that are exactly 1.0 (the
    untapered flat-top region)."""
    if M <= 0:
        return 0
    return int(np.count_nonzero(tukey_1d_custom(M, r) >= 1.0 - 1e-12))


def _solve_tukey_window_for_flat_top(target_flat_top, r, max_M):
    """
    Find the Tukey window length M (<= max_M) whose actual flat-top length
    (per tukey_1d_custom, the exact function used to window the FFT input --
    not scipy's tukey, which rounds its taper cutoff slightly differently)
    is closest to target_flat_top. Not a closed-form solve: the taper
    cutoff involves a floor(), so this searches a small neighborhood around
    the algebraic estimate M ~= target_flat_top / (1 - r) rather than
    trusting that estimate directly.
    """
    target_flat_top = max(0, min(target_flat_top, max_M))
    if target_flat_top <= 0:
        return 0
    if r <= 0:
        return min(target_flat_top, max_M)
    guess = int(round(target_flat_top / (1.0 - r)))
    lo = max(1, guess - 5)
    hi = min(max_M, guess + 5)
    best_M, best_diff = lo, abs(_tukey_flat_top_len(lo, r) - target_flat_top)
    for M in range(lo, hi + 1):
        diff = abs(_tukey_flat_top_len(M, r) - target_flat_top)
        if diff < best_diff:
            best_diff, best_M = diff, M
    return best_M


def invert_abel_A6(k, T1, smooth_window=11, polyorder=3, eps=1e-12):
    """
    Abel inverse transform (Eq. A6) recovering the isotropic 2D radial
    transfer spectrum Tr(Kh) from the 1D along-track transfer spectrum T1(k).

    Needed to compare the Fourier pipeline's along-track T(k)/Pi(k) against
    the coarse-graining pipeline's Pi_L, which is a genuinely 2D (isotropic)
    quantity — the raw 1D Pi(k) is NOT directly comparable to CG's Pi_L.

        Tr(Kh) = -Kh * integral_{Kh}^{k_max} [dT1/dk] / sqrt(k^2 - Kh^2) dk

    k : 1D array, wavenumber grid, increasing, >= 0.
    T1 : 1D array, same shape as k, 1D transfer spectrum density per unit k
         (if you have per-bin values, divide by dk first).
    Returns (Kh, Tr): cleaned wavenumber grid and the radial transfer density.
    """
    k = np.asarray(k, dtype=float)
    T1 = np.asarray(T1, dtype=float)
    if k.ndim != 1 or T1.shape != k.shape:
        raise ValueError("k and T1 must be 1D arrays with the same shape.")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(T1)):
        raise ValueError("k and T1 must be finite.")
    if np.any(k < 0):
        raise ValueError("k must be >= 0.")

    k_uniq, idx = np.unique(k, return_index=True)
    T1_uniq = T1[idx]
    mask_inc = np.diff(k_uniq, prepend=-np.inf) > 0
    k = k_uniq[mask_inc]
    T1 = T1_uniq[mask_inc]
    N = len(k)
    if N < 3:
        raise ValueError("k must have at least 3 strictly increasing points.")

    dk = np.diff(k, prepend=k[0])
    if N >= 2:
        dk[0] = k[1] - k[0]

    if (smooth_window is not None) and (smooth_window >= 3):
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        w = min(w, N - (1 - N % 2))
        T1s = savgol_filter(T1, w, min(polyorder, w - 1), mode="interp") if w >= 3 else T1
    else:
        T1s = T1

    dT1 = np.gradient(T1s, k, edge_order=1)

    Kh = k.copy()
    Tr = np.zeros_like(Kh)
    for i, Ki in enumerate(Kh):
        first_term = 0.0
        if i < N - 1:
            dki = k[i+1] - k[i]
            if dki > 0 and Ki > 0:
                first_term = - np.sqrt(2.0 * Ki) * np.sqrt(dki) * dT1[i]
        if i + 1 < N:
            j = np.arange(i + 1, N)
            diff = k[j]**2 - Ki**2
            denom = np.sqrt(np.clip(diff, eps, None))
            weights = -Ki * dk[j] / denom
            rest = np.sum(weights * dT1[j])
        else:
            rest = 0.0
        Tr[i] = first_term + rest

    return Kh, Tr

def detrend_nanaware(x):
    """
    Remove linear trend ignoring NaNs.
    x: 1D array (e.g., along-track slice)
    """
    x = np.asarray(x)
    mask = np.isfinite(x)
    if mask.sum() < 2:   # not enough points to fit a trend
        return x.copy()

    t = np.arange(len(x))
    p = np.polyfit(t[mask], x[mask], 1)
    trend = np.polyval(p, t)

    y = x - trend
    return y


def _reflect_pad(x, pad):
    """Mirror-extend a 1D array by `pad` samples on each side (no new
    independent information — see spectralflux_utils module note on
    compute_transfer_and_flux_for_strip's pad_lines argument)."""
    return np.concatenate([x[pad-1::-1], x, x[-1:-pad-1:-1]])


def compute_transfer_and_flux_for_strip(ud, vd, N1d, N2d, tukey_r=0.2, min_valid=0.75, pad_lines=0):
    """
    Compute 1D transfer spectrum T(k) and cumulative flux Π(k) for a single across-track strip (column).
    Returns (Tk, Pik) or (None, None) if insufficient valid data.

    pad_lines: if > 0, mirror-reflect-pad the strip by this many samples on
    each side before windowing/FFT, giving a finer k-grid. tukey_r is
    automatically rescaled so the ABSOLUTE taper width (in lines) matches
    what it would be on the unpadded strip -- i.e. padding does not change
    how much of the real data gets tapered, only what's appended beyond it.
    NOTE: reflect-padding does not add independent physical information (the
    padded samples are a mirror of data you already have); empirically it
    changes both the variance AND the mean of the estimate at a given
    nominal wavelength (verified 2026-08-21), so treat this as an
    experimental option, not a validated improvement -- compare against
    pad_lines=0 before trusting results computed with it.
    """
    valid = np.isfinite(ud) & np.isfinite(vd) & np.isfinite(N1d) & np.isfinite(N2d)
    if valid.mean() < min_valid:
        return None, None

    ny_orig = len(ud)
    if pad_lines > 0:
        ud, vd, N1d, N2d = (_reflect_pad(np.asarray(a), pad_lines) for a in (ud, vd, N1d, N2d))
        taper_width = tukey_r * ny_orig / 2.0
        tukey_r = 2.0 * taper_width / len(ud)

    ny = len(ud)
    W1d = tukey_1d_custom(ny, r=tukey_r)
    norm = ny

    ud = detrend_nanaware(ud)
    vd = detrend_nanaware(vd)
    N1d = detrend_nanaware(N1d)
    N2d = detrend_nanaware(N2d)

    ud  = ud  - np.nanmean(ud)
    vd  = vd  - np.nanmean(vd)
    N1d = N1d - np.nanmean(N1d)
    N2d = N2d - np.nanmean(N2d)

    # NaN fill before FFT: np.fft.rfft propagates any single NaN to all
    # output bins, discarding the entire strip even after the min_valid check.
    ud  = np.where(np.isfinite(ud),  ud,  0.0)
    vd  = np.where(np.isfinite(vd),  vd,  0.0)
    N1d = np.where(np.isfinite(N1d), N1d, 0.0)
    N2d = np.where(np.isfinite(N2d), N2d, 0.0)

    uw, vw = ud*W1d, vd*W1d
    N1w, N2w = N1d*W1d, N2d*W1d

    Uh  = np.fft.rfft(uw) / norm
    Vh  = np.fft.rfft(vw) / norm
    N1h = np.fft.rfft(N1w) / norm
    N2h = np.fft.rfft(N2w) / norm

    # dE(k)/dt = -Re[u_hat*(k) . N_hat(k)] (momentum eq: du/dt = -N).
    # Verified numerically (convergence test, dt->0): Tk must carry the
    # leading minus sign so that Tk == dE(k)/dt, matching CG's Pi_L sign
    # convention (positive = forward/downscale cascade).
    Tk = -np.real(Uh * np.conj(N1h) + Vh * np.conj(N2h))
    if Tk.size > 2:
        Tk[1:-1] *= 2.0
    # signal is demeaned -> DC bin is physically zero (not mixed with k[1]).
    Tk[0] = 0.0

    # Cumulative flux Π(k) = \int_k^{kmax} T(q) via trapezoidal rule
    Pik = np.flip(np.cumsum(np.flip(Tk)))

    return Tk, Pik


def compute_transfer_and_flux_2d(ud, vd, N1d, N2d, dx, dy, tukey_r=0.2, nb=60):
    """
    Isotropic 2D transfer spectrum T(k) and cumulative flux Pi(k) for a single
    2D patch (genuinely 2D FFT, not the per-strip 1D version above).

    ud, vd   : 2D velocity patch (ny, nx) [m/s]
    N1d, N2d : 2D nonlinear advection terms, N1 = u du/dx + v du/dy,
               N2 = u dv/dx + v dv/dy [m/s^2]
    dx, dy   : scalar grid spacing (meters) -- mean spacing is fine for a
               roughly-regular patch.
    tukey_r  : Tukey window taper fraction (applied via a separable 2D window).
    nb       : number of geometric-spaced radial k bins for the isotropic ring sum.

    Returns (k_cpkm, T_ring, Pi) -- radial wavenumber [cycles/km], per-shell
    transfer, and cumulative flux, in the same physical units (W/kg per
    wavenumber, W/kg) as compute_transfer_and_flux_for_strip.

    Sign convention: T_ring/Pi follow the SAME dE(k)/dt = -Re[u_hat* . N_hat]
    identity as the 1D function above (positive Pi = forward/downscale
    cascade, matching CG's Pi_L). Verified via the same convergence-test
    methodology (dt->0 synthetic check against a direct forward-Euler energy
    measurement) -- see scripts/dev/verify_2d_sign.py.
    """
    from scipy.signal import detrend as _detrend

    ny, nx = ud.shape

    def _prep(a):
        return _detrend(_detrend(a, axis=0, type='linear'), axis=1, type='linear')

    ud, vd, N1d, N2d = _prep(ud), _prep(vd), _prep(N1d), _prep(N2d)

    W2d = tukey2d(ny, nx, r=tukey_r)
    norm = nx * ny

    Uh  = np.fft.fft2(ud * W2d) / norm
    Vh  = np.fft.fft2(vd * W2d) / norm
    N1h = np.fft.fft2(N1d * W2d) / norm
    N2h = np.fft.fft2(N2d * W2d) / norm

    # dE(k)/dt = -Re[u_hat* . N_hat] (NO extra factor of 2 here). By Parseval for
    # this FFT normalization (divide by nx*ny), sum_k |u_hat(k)|^2 = mean(u^2)
    # exactly (verified numerically) -- so the PHYSICALLY-normalized kinetic
    # energy density (the one that sums to the standard (1/2)<u^2+v^2> kinetic
    # energy, matching CG's Pi_L convention) is e(k) = (1/2)(|u_hat|^2+|v_hat|^2),
    # and de(k)/dt = (1/2)*d(|u_hat|^2+|v_hat|^2)/dt = -Re[u_hat*.N_hat+v_hat*.N_hat2]
    # -- the (1/2) here exactly cancels the factor of 2 from the d|z|^2/dt=2Re(z*dz/dt)
    # calculus identity, leaving no net factor. (An earlier version of this
    # function had an erroneous extra *2, added after validating against
    # E=|u_hat|^2+|v_hat|^2 with no (1/2) -- that reference itself was wrong, not
    # physically normalized, so its ratio=2.0 "confirmation" was validating
    # against the wrong target. Confirmed via convergence test against the
    # correctly-normalized (1/2)-included energy: ratio -> 1.0000 exactly, and
    # the existing 1D function was independently re-checked the same way and
    # needs no change -- see scripts/dev/verify_2d_sign.py.)
    T2D = -np.real(Uh * np.conj(N1h) + Vh * np.conj(N2h))

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)

    out = unified_flux_postprocess(T2D, kx, ky, nb=nb, spacing="geom",
                                    window_power_mean=np.mean(W2d**2))
    return out['q_cpkm'], out['T_ring'], out['Pi']

