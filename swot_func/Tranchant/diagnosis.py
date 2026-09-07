# From https://github.com/treden/SwotDiag (Tranchant et al.), MIT License --
# see LICENSE in this directory. Extended here with second-derivative support.
import numpy as np
import scipy
from .misc import *

g = 9.81
omega = 7.292115e-5  # (1/s) (Groten, 2004).

def compute_f(lat):
    """
    Computes the Coriolis parameter.

    Parameters:
        lat : float
            Latitude in degrees.

    Returns:
        float
            The Coriolis parameter.
    """
     
    f = 2 * omega * np.sin(np.radians(lat))
    return np.asanyarray(f)
    

def compute_ocean_diagnostics_from_eta(
    eta, x, y, proj='lonlat', derivative='fit', lat=None, n=9, 
    parallel=True, axis=(-2, -1), verbose=True, kernel='circular', 
    min_valid_points=0.5, second_derivative='dxdy', cyclostrophy='GW', avoid_negative = False):
    """
    Compute geostrophic and cyclo-geostrophic currents, strain rate, 
    relative vorticity, and other diagnostics from sea surface height (SSH).

    Parameters:
    ----------
    eta : array_like
        2D array of sea surface height (SSH) data.
    x : array_like
        X-coordinates (longitude or Cartesian, depending on `proj`).
    y : array_like
        Y-coordinates (latitude or Cartesian, depending on `proj`).
    proj : str
        Coordinate projection type:
        - 'lonlat': longitude-latitude (default)
        - 'xy': Cartesian coordinates
    derivative : str
        Method for calculating derivatives:
        - 'dxdy': finite differences
        - 'fit': fitting kernel method (default)
    lat : array_like, optional
        Latitude values, required if `proj='xy'`.
    n : int
        Size of the stencil or kernel for derivative calculations (default: 9).
    parallel : bool
        Whether to use parallel processing for derivative calculations (default: True).
    axis : tuple of int
        Axes along which derivatives are computed (default: (-2, -1)).
    verbose : bool
        If True, prints progress and debugging information (default: True).
    kernel : str
        Kernel shape for the fitting method ('circular' by default).
        - 'circular': circular kernel (default)
        - 'square': square kernel 
    min_valid_points : float
        Ratio of minimum valid points for fitting kernels (default: 0.5).
    second_derivative : str
        Method for calculating second derivatives when using fitting kernels:
        - 'dxdy': finite differences of first differences (default)
        - 'fit': second derivatives picked directly from surface curvature
    cyclostrophy : str
        Method for cyclo-geostrophic velocity computation:
        - 'GW': Gradient-wind method (default)
        - 'PENVEN': First approximation method.

    Returns:
    -------
    diag : dict
        Dictionary containing the computed diagnostics:
        - 'ug', 'vg': Geostrophic eastward and northward currents
        - 'ucg', 'vcg': Cyclo-geostrophic eastward and northward currents
        - 'S': Strain rate
        - 'zeta': Relative vorticity
        - 'OW': Okubo-Weiss parameter
        - 'dx', 'dy': First derivatives of SSH
        - 'dxx', 'dyy', 'dxy': Second derivatives of SSH
        - 'theta': Rotation angles for coordinate transformations

    """
    
    # Initialize diagnostics dictionary
    diag = {}

    # Ensure input arrays are numpy arrays
    x, y = np.asanyarray(x), np.asanyarray(y)

    # Compute Coriolis parameter based on projection
    if proj == 'lonlat':
        f = compute_f(y)
    elif proj == 'xy' and lat is not None:
        f = compute_f(lat)
    else:
        print('Invalid "proj" argument (must be "lonlat" or "xy") or "lat" is missing.')
        return
    
    # Compute first and second derivatives
    if derivative == 'dxdy':
        # Compute first derivatives by finite differences
        dy, dx = first_derivative(eta, n=n, axis=axis)
        dyy, dyx = first_derivative(dy, n=n, axis=axis)
        dxy, dxx = first_derivative(dx, n=n, axis=axis)
    elif derivative == 'fit':
        # Fit derivatives using the fitting kernel method
        dxx, dxy, dx, dyy, dy, eta_fit = fit_derivatives(eta, n=n, parallel=parallel, order=2, verbose=verbose, kernel = kernel, return_ssh = True, min_valid_points = min_valid_points)
        diag['eta_fit'] = eta_fit

        if second_derivative == 'dxdy':
            dyx, dxx = first_derivative(dx, n = 3, axis = axis)
            dyy, dxy = first_derivative(dy, n = 3, axis = axis)
    
    # third derivatives for jacobian calculation (added by JC)
    dyxx, dxxx = first_derivative(dxx, n = 3, axis = axis) 
    dyyy, dxyy = first_derivative(dyy, n = 3, axis = axis) 
    dxxy, dxyx = first_derivative(dxy, n = 3, axis = axis)

    # Ensure the Coriolis parameter has the correct shape
    if f.shape != dx.shape:
        f = (np.ones((len(x), len(y))) * f).T

    # Compute first derivatives of Coriolis parameter (added by JC)
    fy, _   = first_derivative(f, n=3, axis=axis) 
    fyy, _  = first_derivative(fy, n=3, axis=axis)

    # Compute scale factors
    e1, e2 = scale_factor(x, y, proj=proj)

    # Normalize derivatives by scale factors
    dxx, dxy, dx, dyy, dy = (
        dxx / (e1 ** 2), 
        dxy / (e1 * e2), 
        dx / np.abs(e1), 
        dyy / (e2 ** 2), 
        dy / np.abs(e2)
    )
    # Normalize third derivatives by scale factors (added by JC)
    dxxx = dxxx / (e1**3)
    dxxy = dxxy / (e1**2 * e2)
    dxyy = dxyy / (e1 * e2**2)
    dyyy = dyyy / (e2**3)

    # Normalize derivatives of Coriolis parameter by scale factors (added by JC)
    fy = fy / np.abs(e2)
    fyy = fyy / (e2**2)

    # Re-adjust the Coriolis parameter shape if necessary
    if f.shape != dxx.shape:
        f = (np.ones((len(x), len(y))) * f).T

    # Reproject derivative from across/along-track components to x/y frame    

    theta = compute_angle(x, y, proj = proj)

    _dx = dx*np.cos(theta) - dy * np.sin(theta)
    _dy = dx*np.sin(theta) + dy * np.cos(theta)

    _dxx = dxx*np.cos(theta)**2 + dyy*np.sin(theta)**2 - dxy*np.sin(2*theta)
    _dyy = dxx*np.sin(theta)**2 + dyy*np.cos(theta)**2 + dxy*np.sin(2*theta)
    _dxy = dxy 

    dx, dy = _dx, _dy
    dxx, dyy, dxy = _dxx, _dyy, _dxy

    # Reproject third derivatives (added by JC)
    _dxxx = (
        np.cos(theta)**3 * dxxx
        - 3 * np.cos(theta)**2 * np.sin(theta) * dxxy
        + 3 * np.cos(theta) * np.sin(theta)**2 * dxyy
        - np.sin(theta)**3 * dyyy
    )

    _dxxy = (
        np.cos(theta)**2 * np.sin(theta) * dxxx
        + (np.cos(theta)**3 - 2 * np.cos(theta) * np.sin(theta)**2) * dxxy
        + (np.sin(theta)**3 - 2 * np.cos(theta)**2 * np.sin(theta)) * dxyy
        + np.cos(theta) * np.sin(theta)**2 * dyyy
    )

    _dxyy = (
        np.cos(theta) * np.sin(theta)**2 * dxxx
        + (2 * np.sin(theta) * np.cos(theta)**2 - np.sin(theta)**3) * dxxy
        + (np.cos(theta)**3 - 2 * np.sin(theta)**2 * np.cos(theta)) * dxyy
        - np.sin(theta) * np.cos(theta)**2 * dyyy
    )

    _dyyy = (
        np.sin(theta)**3 * dxxx
        + 3 * np.sin(theta)**2 * np.cos(theta) * dxxy
        + 3 * np.sin(theta) * np.cos(theta)**2 * dxyy
        + np.cos(theta)**3 * dyyy
    )

    dxxx, dyyy, dxxy, dxyy = _dxxx, _dyyy, _dxxy, _dxyy

    ### Calculate geostrophic currents and Cyclo-geostrophic currents (see Tranchant et al., 2024)
    ug, vg = (-g / f * dy), (g / f * dx)  # Geostrophic currents

    if cyclostrophy == 'PENVEN': # First iteration of PENVEN Method, see Penven, 2014
        ucg, vcg = (
            (-g / f * dy) + (g ** 2 / f ** 3) * (dy * dxx - dx * dxy), 
            (g / f * dx) + (g ** 2 / f ** 3) * (dy * dxy - dx * dyy)
        )  # Cyclo-geostrophic currents (first approximation)
        # ucg1, vcg1 = iterative_method(ug, vg, ucg, vcg, x, y, max_iter = 0, threshold = 0.01, derivative = 'dxdy', n = 3, verbose = verbose, proj = proj)

    elif cyclostrophy == 'GW': # Gradient-wind method based on flow curvature K, see Jan Jaap, 2022

        K = (-dyy*(dx**2)-dxx*(dy**2)+ 2*(dxy*dx*dy))/((dx**2+dy**2)**(3/2))
        K = np.sign(f) * - K

        Ug = np.sqrt(ug**2 + vg**2)

        c = 1+4*K*np.abs(Ug)/f

        if avoid_negative:
            c = np.maximum(c, 0.01)

        
        ucg = 2*ug/(1+np.sqrt(c))
        vcg = 2*vg/(1+np.sqrt(c))

    ### Higher order diagnostics : Vorticity, Strain rate and Okubo-Weiss

    # # Calculate strain rate
    Sn = -g/f * dxy - g/f * dxy
    Ss = g/f * dxx -g/f * dyy
    S = np.sqrt(Sn ** 2 + Ss ** 2) / np.abs(f)

    # Relative vorticity for geostrophic (and cyclostrophic currents, to do)
    zeta_wob = ((g / f * dxx) + (g / f * dyy)) # without beta term name modified by JC

    # compute derivatives of vorticity for Jacobian calculation (added by JC)
    L  = dxx + dyy
    Lx = dxxx + dxyy
    Ly = dxxy + dyyy
    B  = fy * dy
    By = fyy * dy + fy * dyy

    eps    = 1e-10
    f_safe = np.where(np.abs(f) < eps, np.nan, f)
    zeta   = (g / f_safe) * L - g * B / (f_safe**2)
    zeta_x = (g/f_safe) * Lx - g * (fy * dxy) / (f_safe**2)
    zeta_y = g * (Ly / f_safe - L * fy / (f_safe**2)) \
         - g * (By / (f_safe**2) - 2 * (fy**2) * dy / (f_safe**3))
    Jg = ug * zeta_x + vg * zeta_y

    # Okubo Weiss
    OW = (Sn)**2 + (Ss)**2 - (zeta*f)**2
    OW = OW/(f**2)

    # Store currents in diagnostics dictionary
    
    diag['dx'] = dx
    diag['dy'] = dy
    diag['dxy'] = dxy
    diag['dxx'] = dxx
    diag['dyy'] = dyy
    diag['theta'] = theta

    diag['ug'] = ug
    diag['vg'] = vg
    diag['ucg'] = ucg
    diag['vcg'] = vcg

    diag['S'] = S
    diag['zeta_wob'] = zeta_wob  # without beta term name modified by JC
    
    diag['OW'] = OW

    diag['zeta'] = zeta # with beta term added by JC
    diag['Jg'] = Jg  # Jacobian of geostrophic vorticity (added by JC)
    diag['zeta_x'] = zeta_x  # Derivative of vorticity in x (added by JC)
    diag['zeta_y'] = zeta_y  # Derivative of vorticity in y (added by JC)
    diag['dxxx'] = dxxx  # Third derivative of SSH in x (added by JC)
    diag['dyyy'] = dyyy  # Third derivative of SSH in y (added by JC)
    diag['dxxy'] = dxxy  # Mixed third derivative of SSH (added by JC)
    diag['dxyy'] = dxyy  # Mixed third derivative of SSH (added by JC)

    return diag