from unittest.mock import patch

from lst_tools.finder.datalevel_attributes import DLAttributes
from lst_tools.finder.search_results import (
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3LevelSearchResult,
)


def make_attrs(lstchain_version):
    return DLAttributes(
        path=None,
        run_number=0,
        tailcuts_level=(8, 4),
        lstchain_version=lstchain_version,
        nsb_level=None,
    )


@patch("lst_tools.finder.search_results.parse_dl1_path")
def test_dl1_level_search_result_parses_all_files(mock_parse_dl1_path):
    files = ["a.h5", "b.h5", "c.h5"]
    mock_parse_dl1_path.side_effect = make_attrs

    result = DL1LevelSearchResult(files)

    assert result.files == files
    assert mock_parse_dl1_path.call_count == len(files)
    assert [f.args[0] for f in mock_parse_dl1_path.call_args_list] == files


@patch("lst_tools.finder.search_results.parse_dl1_path")
def test_newest_attributes_returns_highest_lstchain_version(mock_parse_dl1_path):
    newest = make_attrs("0.10")
    mock_parse_dl1_path.side_effect = [make_attrs("0.9"), newest, make_attrs("0.7")]

    result = DL1LevelSearchResult(["v0.9.h5", "v0.10.h5", "v0.7.h5"])

    assert result.newest_attributes is newest


@patch("lst_tools.finder.search_results.parse_dl1_path")
def test_newest_attributes_single_file(mock_parse_dl1_path):
    attrs = make_attrs("0.9")
    mock_parse_dl1_path.return_value = attrs

    result = DL1LevelSearchResult(["only.h5"])

    assert result.newest_attributes is attrs


def test_newest_attributes_empty_result():
    assert DL1LevelSearchResult([]).newest_attributes is None
    assert DL2LevelSearchResult([]).newest_attributes is None
    assert DL3LevelSearchResult([]).newest_attributes == []


@patch("lst_tools.finder.search_results.parse_dl2_path")
def test_dl2_newest_attributes_returns_highest_lstchain_version(mock_parse_dl2_path):
    newest = make_attrs("0.10")
    mock_parse_dl2_path.side_effect = [make_attrs("0.9"), newest]

    result = DL2LevelSearchResult(["v0.9.h5", "v0.10.h5"])

    assert result.newest_attributes is newest


@patch("lst_tools.finder.search_results.parse_dl3_path")
def test_dl3_newest_attributes_returns_list_of_newest_version(mock_parse_dl3_path):
    newest_a = make_attrs("0.10")
    newest_b = make_attrs("0.10")
    mock_parse_dl3_path.side_effect = [make_attrs("0.9"), newest_a, newest_b]

    result = DL3LevelSearchResult(["old.fits", "new_a.fits", "new_b.fits"])

    newest = result.newest_attributes
    assert isinstance(newest, list)
    assert newest == [newest_a, newest_b]
