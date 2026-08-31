import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from lst_tools.catalog import CatalogSource
from lst_tools.datacheck import DataCheckTables, DataFilter
from lst_tools.scripts.prepare_offruns import (
    create_dl2_links,
    save_datacheck_file,
    select_offruns,
)


def test_select_offruns_applies_advanced_cuts_but_not_source_angle_cut():
    statistics = pd.DataFrame(
        {
            "run_number": [1, 2, 3],
            "n_subruns": [2, 2, 2],
            "date": [20240101, 20240101, 20240101],
            "n_flatfield": [1, 1, 1],
            "n_pedestal": [1, 1, 1],
            "mean_ra": [10.0, 20.0, 30.0],
            "mean_dec": [0.0, 0.0, 0.0],
            "mean_cos_zd": [0.8, 0.8, 0.8],
            "pointing_dec_std": [0.001, 0.001, 0.001],
            "mean_diffuse_nsb_std": [2.0, 2.0, 2.0],
            "mean_intensity_threshold": [40.0, 40.0, 40.0],
            "mean_fit_p_value": [0.5, 0.5, 0.5],
            "mean_index": [-2.2, -2.2, -2.2],
            "mean_R422": [1.6, 1.0, 1.6],
            "fraction_around_mode_R422": [0.9, 0.9, 0.9],
        }
    )
    data_filter = DataFilter(source_ra=0, source_dec=0)
    sources = (
        CatalogSource(
            name="near-run-3",
            coord=SkyCoord(ra=30, dec=0, unit="deg"),
            catalog="test",
        ),
    )

    selected = select_offruns(
        statistics,
        data_filter,
        sources,
        min_separation=2 * u.deg,
    )

    # Run 1 is far outside DataFilter's normal 0.3–0.5 degree source-angle
    # range but remains selected. Run 2 fails an advanced cut; run 3 is too
    # close to the supplied catalog source.
    assert selected["run_number"].tolist() == [1]


def test_create_dl2_links_is_idempotent(tmp_path):
    source = tmp_path / "source" / "dl2_LST-1.Run00001.h5"
    source.parent.mkdir()
    source.touch()
    selected = pd.DataFrame({"run_number": [1], "date": [20240101]})

    def find_path(date, run_number, level):
        assert (date, run_number, level) == (20240101, 1, "dl2")
        return str(source)

    output = tmp_path / "offruns" / "dl2"
    first = create_dl2_links(selected, output, path_finder=find_path)
    second = create_dl2_links(selected, output, path_finder=find_path)

    assert first == {"dl2:created": 1}
    assert second == {"dl2:existing": 1}
    assert (output / source.name).resolve() == source.resolve()


def test_save_datacheck_file_contains_only_selected_runs(tmp_path):
    tables = DataCheckTables(
        flatfield=pd.DataFrame({"runnumber": [1, 2]}),
        cosmics_intensity_spectrum=pd.DataFrame({"runnumber": [1, 2]}),
        runsummary=pd.DataFrame({"runnumber": [1, 2]}),
    )
    selected = pd.DataFrame({"run_number": [2]})

    output_file = save_datacheck_file(tables, selected, tmp_path / "data_check")
    loaded = DataCheckTables.from_files([str(output_file)])

    assert output_file == tmp_path / "data_check" / "DL1_datacheck_offruns.h5"
    assert loaded.flatfield["runnumber"].tolist() == [2]
    assert loaded.cosmics_intensity_spectrum["runnumber"].tolist() == [2]
    assert loaded.runsummary["runnumber"].tolist() == [2]
