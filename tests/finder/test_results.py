from unittest.mock import patch

from lst_tools.finder import DL1SearchResult, DL2SearchResult, DL3SearchResult, ObservedDataMetadata


def make_metadata(lstchain_version):
    return ObservedDataMetadata(
        path=None,
        run_number=0,
        tailcuts_level=(8, 4),
        lstchain_version=lstchain_version,
        nsb_level=None,
    )


@patch("lst_tools.finder.results.parse_dl1_path")
def test_dl1_search_result_parses_all_files(mock_parse_dl1_path):
    files = ["a.h5", "b.h5", "c.h5"]
    mock_parse_dl1_path.side_effect = make_metadata

    result = DL1SearchResult(files)

    assert result.files == files
    assert mock_parse_dl1_path.call_count == len(files)
    assert [f.args[0] for f in mock_parse_dl1_path.call_args_list] == files


@patch("lst_tools.finder.results.parse_dl1_path")
def test_newest_metadata_returns_highest_lstchain_version(mock_parse_dl1_path):
    newest = make_metadata("0.10")
    mock_parse_dl1_path.side_effect = [make_metadata("0.9"), newest, make_metadata("0.7")]

    result = DL1SearchResult(["v0.9.h5", "v0.10.h5", "v0.7.h5"])

    assert result.newest_metadata is newest


@patch("lst_tools.finder.results.parse_dl1_path")
def test_newest_metadata_single_file(mock_parse_dl1_path):
    attrs = make_metadata("0.9")
    mock_parse_dl1_path.return_value = attrs

    result = DL1SearchResult(["only.h5"])

    assert result.newest_metadata is attrs


def test_newest_metadata_empty_result():
    assert DL1SearchResult([]).newest_metadata is None
    assert DL2SearchResult([]).newest_metadata is None
    assert DL3SearchResult([]).newest_metadata == []


@patch("lst_tools.finder.results.parse_dl2_path")
def test_dl2_newest_metadata_returns_highest_lstchain_version(mock_parse_dl2_path):
    newest = make_metadata("0.10")
    mock_parse_dl2_path.side_effect = [make_metadata("0.9"), newest]

    result = DL2SearchResult(["v0.9.h5", "v0.10.h5"])

    assert result.newest_metadata is newest


@patch("lst_tools.finder.results.parse_dl3_path")
def test_dl3_newest_metadata_returns_list_of_newest_version(mock_parse_dl3_path):
    newest_a = make_metadata("0.10")
    newest_b = make_metadata("0.10")
    mock_parse_dl3_path.side_effect = [make_metadata("0.9"), newest_a, newest_b]

    result = DL3SearchResult(["old.fits", "new_a.fits", "new_b.fits"])

    newest = result.newest_metadata
    assert isinstance(newest, list)
    assert newest == [newest_a, newest_b]
