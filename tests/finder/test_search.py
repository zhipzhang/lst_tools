from pathlib import Path
from unittest.mock import patch

import pytest

from lst_tools.finder.search import find_laplama_files
from lst_tools.finder.search_results import (
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3LevelSearchResult,
)

DL1_OLD = "/fefs/aswg/data/real/DL1/20250323/v0.10/tailcut1005/dl1_LST-1.Run20463.h5"
DL1_NEW = "/fefs/onsite/data/lst-pipe/LSTN-01/DL1/20250323/v0.11/tailcut1005/dl1_LST-1.Run20463.h5"

DL3_PREFIX = "/fefs/onsite/data/lst-pipe/LSTN-01/DL3/20230821"
DL3_OLD = (
    f"{DL3_PREFIX}/v0.10/tailcut1005/nsb_tuning_0.38/0.10.18/20250212_v0.10.17_x/src_indep/"
    "full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"
)
DL3_NEW_A = (
    f"{DL3_PREFIX}/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/20260101_v0.11.0_x/src_indep/"
    "full-enclosure/ring-wobble/gheff0.7_thetacont0.7/irf_interp/dl3_LST-1.Run14083.fits"
)
DL3_NEW_B = (
    f"{DL3_PREFIX}/v0.11/tailcut1005/nsb_tuning_0.38/0.11.0/20260101_v0.11.0_x/src_indep/"
    "point/diffuse/gheff0.9_thetacont0.9/irf_interp/dl3_LST-1.Run14083.fits"
)


@patch("lst_tools.finder.search.glob.glob")
def test_find_laplama_files_dl1_returns_newest_single_file(mock_glob):
    # LAPLAMA_DATA_PATHS order: onsite first, then aswg
    mock_glob.side_effect = [[DL1_NEW], [DL1_OLD]]

    result = find_laplama_files(20250323, 20463, level="dl1")

    assert isinstance(result, DL1LevelSearchResult)
    assert len(result) == 2
    assert result.newest_attributes.path == Path(DL1_NEW)


@patch("lst_tools.finder.search.glob.glob")
def test_find_laplama_files_dl2_returns_newest_single_file(mock_glob):
    dl2_old = DL1_OLD.replace("/DL1/", "/DL2/").replace("dl1_LST-1", "dl2_LST-1").replace(
        "tailcut1005/", "tailcut1005/nsb_tuning_0.38/"
    )
    dl2_new = DL1_NEW.replace("/DL1/", "/DL2/").replace("dl1_LST-1", "dl2_LST-1").replace(
        "tailcut1005/", "tailcut1005/nsb_tuning_0.38/"
    )
    mock_glob.side_effect = [[dl2_new], [dl2_old]]

    result = find_laplama_files(20250323, 20463, level="dl2")

    assert isinstance(result, DL2LevelSearchResult)
    assert result.newest_attributes.path == Path(dl2_new)


@patch("lst_tools.finder.search.glob.glob")
def test_find_laplama_files_dl3_returns_list_of_newest_version(mock_glob):
    mock_glob.side_effect = [[DL3_NEW_A, DL3_NEW_B], [DL3_OLD]]

    result = find_laplama_files(20230821, 14083, level="dl3")

    assert isinstance(result, DL3LevelSearchResult)
    newest = result.newest_attributes
    assert {attrs.path for attrs in newest} == {Path(DL3_NEW_A), Path(DL3_NEW_B)}


@patch("lst_tools.finder.search.glob.glob")
def test_find_laplama_files_no_match_warns_and_returns_empty(mock_glob):
    mock_glob.return_value = []

    with pytest.warns(UserWarning, match="No file found"):
        result = find_laplama_files(20250323, 99999, level="dl1")

    assert isinstance(result, DL1LevelSearchResult)
    assert len(result) == 0
    assert result.newest_attributes is None


def test_find_laplama_files_invalid_level():
    with pytest.raises(ValueError, match="Invalid level"):
        find_laplama_files(20250323, 20463, level="dl0")
