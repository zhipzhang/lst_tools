from ctapipe.containers import EventType
from ctapipe.core import Component
from ctapipe.core.traits import Dict, Float, Int, List


class IRFBinning(Component):
    true_energy_min = Float(
        help="Minimum true energy for binning",
        default_value=0.1,
    ).tag(config=True)
    true_energy_max = Float(
        help="Maximum true energy for binning",
        default_value=100.0,
    ).tag(config=True)
    true_energy_bins = Int(
        help="Number of true energy bins in log space",
        default_value=20,
    ).tag(config=True)
    reco_energy_min = Float(
        help="Minimum reconstructed energy for binning",
        default_value=0.1,
    ).tag(config=True)
    reco_energy_max = Float(
        help="Maximum reconstructed energy for binning",
        default_value=100.0,
    ).tag(config=True)
    reco_energy_bins = Int(
        help="Number of reconstructed energy bins in log space",
        default_value=20,
    ).tag(config=True)
    fov_offset_min = Float(
        help="Minimum value for FoV Offset bins",
        default_value=0.1,
    ).tag(config=True)
    fov_offset_max = Float(
        help="Maximum value for FoV offset bins",
        default_value=1.1,
    ).tag(config=True)
    fov_offset_n_edges = Int(
        help="Number of edges for FoV offset bins",
        default_value=9,
    ).tag(config=True)
    source_offset_min = Float(
        help="Minimum value for Source offset for PSF IRF",
        default_value=0,
    ).tag(config=True)
    source_offset_max = Float(
        help="Maximum value for Source offset for PSF IRF",
        default_value=1,
    ).tag(config=True)
    source_offset_n_edges = Int(
        help="Number of edges for Source offset for PSF IRF",
        default_value=101,
    ).tag(config=True)
