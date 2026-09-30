import pandas as pd

from lst_tools.dl2 import LSTDL2EventTable

"""
Background Maker working on the LSTDL2EventTable and return the useful structure.
To work on dl2 file, cuts condition have to be applied:
    - intensity cuts
    - energy-dependent gammaness cuts
the difficulty is we have to pass the filter class to the BkgMaker,
so it's better to have a separate class for the filter.

The return value should contain:
    - livetime: float
    - pointing_ra: float
    - pointing_dec: float
    - pointing_alt: float
    - pointing_az: float
    - camera_x: np.ndarray
    - camera_y: np.ndarray
    - reco_energy: np.ndarray
"""


class BkgMaker:
    def __init__(self):
        pass

    def __call__(self, dl2: LSTDL2EventTable) -> dict:
        pass
