"""
Spectral-flux core: coarse-graining and Fourier spectral energy flux.
See examples/regular_grid_pipeline.py for usage.
"""
from .coarse_graining import compute_sfs_energy_flux
from .spectralflux_utils import compute_transfer_and_flux_for_strip, compute_transfer_and_flux_2d
from .grid_utils import compute_dx_dy
