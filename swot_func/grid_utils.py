from haversine import haversine_vector, Unit
import numpy as np

# --- constants ---
g = 9.81
omega = 7.292115e-5    # Earth's rotation rate [rad/s]
R_earth = 6_371_000.0  # [m]


# ---------- Debug helpers ----------
def report(name, arr):
    """Print quick stats for an array."""
    if arr is None:
        print(f"[{name}] None")
        return
    a = np.asarray(arr)
    n = a.size
    n_nan = np.isnan(a).sum()
    n_inf = np.isinf(a).sum()
    finite = a[np.isfinite(a)]
    vmin = np.nanmin(a) if np.isfinite(a).any() else np.nan
    vmax = np.nanmax(a) if np.isfinite(a).any() else np.nan
    print(f"[{name}] shape={a.shape}, finite={finite.size}/{n} "
          f"(nan={n_nan}, inf={n_inf}), min={vmin:.3e}, max={vmax:.3e}")


def assert_same_shape(a, b, namea="A", nameb="B"):
    if np.asarray(a).shape != np.asarray(b).shape:
        raise ValueError(f"Shape mismatch: {namea}{np.asarray(a).shape} vs {nameb}{np.asarray(b).shape}")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/lon arrays of identical shape."""
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


def _compute_cell_spacings(lat2d, lon2d):
    """
    Forward haversine spacing between every pair of adjacent cells (meters).
      atrack_sp[i,j] : distance from (i,j) → (i+1,j)  [along-track, axis-0]
      ctrack_sp[i,j] : distance from (i,j) → (i,j+1)  [cross-track, axis-1]
    Last row / column is filled by repeating its neighbour.
    """
    ny, nx = lat2d.shape
    atrack_sp = np.empty((ny, nx))
    ctrack_sp = np.empty((ny, nx))
    atrack_sp[:-1, :] = haversine_km(lat2d[:-1, :], lon2d[:-1, :],
                                      lat2d[1:,  :], lon2d[1:,  :]) * 1000.0
    atrack_sp[-1, :]  = atrack_sp[-2, :]
    ctrack_sp[:, :-1] = haversine_km(lat2d[:, :-1], lon2d[:, :-1],
                                      lat2d[:, 1: ], lon2d[:, 1: ]) * 1000.0
    ctrack_sp[:, -1]  = ctrack_sp[:, -2]
    return atrack_sp, ctrack_sp


def compute_dx_dy(lat2d, lon2d):
    """
    Estimate grid spacing (dx, dy) in meters from 2D longitude/latitude.
    - dx: east–west distance between adjacent columns (same  j, i -> j+1)
    - dy: north–south distance between adjacent rows (same j, i -> i+1)
    Edge points use one-sided differences by copying the previous spacing.
    """

    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)

    # Normalize longitude/latitude to conventional ranges
    if lon2d.min() >= 0 and lon2d.max() > 180:
        lon2d = ((lon2d + 180) % 360) - 180   # convert 0–360 → -180–180
    if lat2d.min() >= 0 and lat2d.max() > 90:
        #print(lat2d.min(), lat2d.max())
        lat2d = lat2d - 90                    # convert 0–180 → -90–90


    ny, nx = lat2d.shape
    dx = np.empty((ny, nx), dtype=float)
    dy = np.empty((ny, nx), dtype=float)

    # dx: distance along x-direction (east–west; columns change)
    for i in range(nx):
        for j in range(ny):
            if j < ny - 1:
                p1 = (lat2d[j,   i],   lon2d[j, i])
                p2 = (lat2d[j+1, i], lon2d[j+1, i])
                #print (p1, p2)
                dx[j, i] = haversine_vector([p1], [p2], Unit.METERS)[0]
            else:
                # right edge: copy from the previous column
                dx[j, i] = dx[j-1, i]

    # dy: distance along y-direction (north–south; rows change)
    for i in range(nx):
        for j in range(ny):
            if i < nx - 1:
                p1 = (lat2d[j, i  ], lon2d[j, i  ])
                p2 = (lat2d[j, i+1], lon2d[j, i+1])
                #print (p1, p2)
                dy[j, i] = haversine_vector([p1], [p2], Unit.METERS)[0]
            else:
                # bottom edge: copy from the previous row
                dy[j, i] = dy[j, i-1]

    return dy, dx  # return in (y, x) order to match array indexing
