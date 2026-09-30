from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table
from ctapipe.core import ToolConfigurationError
from gammapy.irf import Background2D

from lst_tools.bkg import SkyOffsetEvents
from lst_tools.scripts import build_dl2_background as background_module
from lst_tools.scripts.build_dl2_background import (
    BuildDL2Background,
    build_dl2_background,
    find_offrun_dl2_files,
    read_gh_cuts,
)


def write_gh_cuts(path):
    table = Table(
        {
            "low": [0.1, 1.0] * u.TeV,
            "high": [1.0, 10.0] * u.TeV,
            "cut": [0.7, 0.9],
        }
    )
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, name="GH_CUTS")]).writeto(path)


def test_find_offrun_dl2_files_excludes_validation_runs(tmp_path):
    included = tmp_path / "dl2_LST-1.Run00043.h5"
    excluded = tmp_path / "dl2_LST-1.Run00044.h5"
    ignored = tmp_path / "other.h5"
    for path in (included, excluded, ignored):
        path.touch()

    result = find_offrun_dl2_files(tmp_path, exclude_run_numbers=[44])

    assert result == [included.resolve()]


def test_read_gh_cuts_converts_energy_columns_to_tev(tmp_path):
    target_dl3 = tmp_path / "dl3_LST-1.Run00042.fits"
    write_gh_cuts(target_dl3)

    energy_low, energy_high, gh_cuts = read_gh_cuts(target_dl3)

    np.testing.assert_allclose(energy_low, [0.1, 1.0])
    np.testing.assert_allclose(energy_high, [1.0, 10.0])
    np.testing.assert_allclose(gh_cuts, [0.7, 0.9])


def test_build_dl2_background_filters_events_bins_them_and_writes_fits(tmp_path, monkeypatch):
    target_dl2 = tmp_path / "dl2_LST-1.Run00042.h5"
    target_dl3 = tmp_path / "dl3_LST-1.Run00042.fits"
    offrun_dir = tmp_path / "offdl2"
    output_file = tmp_path / "background2d.fits"
    offrun_dir.mkdir()
    target_dl2.touch()
    write_gh_cuts(target_dl3)
    training_file = offrun_dir / "dl2_LST-1.Run00043.h5"
    validation_file = offrun_dir / "dl2_LST-1.Run00044.h5"
    training_file.touch()
    validation_file.touch()
    loaded_files = []

    def load_dl2(path):
        path = path.resolve()
        loaded_files.append(path)
        if path == target_dl2.resolve():
            return SimpleNamespace(intensity_cuts=80)
        return SimpleNamespace(
            data=pd.DataFrame(
                {
                    "reco_energy": [0.2, 2.0, 0.5, 2.0],
                    "gammaness": [0.8, 0.95, 0.8, 0.8],
                    "intensity": [100, 100, 50, 100],
                }
            ),
            t_eff=10,
        )

    def convert_events(dl2, event_filter):
        selected = event_filter(dl2.data)
        return SkyOffsetEvents(
            x=np.full(len(selected), 0.5) * u.deg,
            y=np.zeros(len(selected)) * u.deg,
            energy=selected["reco_energy"].to_numpy() * u.TeV,
            livetime=dl2.t_eff * u.s,
        )

    monkeypatch.setattr(background_module, "LSTDL2EventTable", load_dl2)
    monkeypatch.setattr(background_module, "sky_offset_events_from_lstdl2", convert_events)

    result = build_dl2_background(
        target_dl2,
        target_dl3,
        offrun_dir,
        output_file,
        energy_edges=[0.1, 1.0, 10.0],
        theta_edges=[0.0, 1.0, 2.0],
        exclude_run_numbers=[44],
    )

    assert result == output_file.resolve()
    assert loaded_files == [target_dl2.resolve(), training_file.resolve()]
    background = Background2D.read(output_file)
    assert background.data.shape == (2, 2)
    assert np.count_nonzero(background.data) == 2
    assert background.meta["LIVETIME"] == pytest.approx(10)


def test_standalone_tool_requires_background_axes():
    tool = BuildDL2Background()

    with pytest.raises(ToolConfigurationError, match="energy_edges"):
        tool.setup()
