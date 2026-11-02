from types import SimpleNamespace

import astropy.units as u
import pandas as pd
import pytest
from astropy.table import Table

from lst_tools.scripts import summarize_lstana


def _patch_readers(monkeypatch):
    selected = pd.DataFrame({"run_number": [13647], "date": [20230709]})
    monkeypatch.setattr(
        summarize_lstana.DataCheckTables,
        "from_files",
        lambda files: SimpleNamespace(statistics=selected),
    )
    headers = {
        "EVENTS": {
            "RA_PNT": 299.24,
            "DEC_PNT": 28.07,
            "AZ_PNT": 93.96,
            "ALT_PNT": 83.45,
            "LIVETIME": 1148.0,
            "NAXIS2": 3,
        },
        "POINTING": {"B_DELTA": "52.34995 deg"},
    }
    monkeypatch.setattr(summarize_lstana.fits, "getheader", lambda path, extname: headers[extname])
    monkeypatch.setattr(summarize_lstana, "LSTDL2EventTable", lambda path: SimpleNamespace(nsb_level=0.22))


def test_summarize_selected_run(tmp_path, monkeypatch):
    _patch_readers(monkeypatch)
    dl3_dir = tmp_path / "dl3"
    dl2_dir = tmp_path / "dl2"
    dl3_dir.mkdir()
    dl2_dir.mkdir()
    (dl3_dir / "dl3_LST-1.Run13647.fits").touch()
    (dl2_dir / "dl2_LST-1.Run13647.h5").touch()

    table = summarize_lstana.summarize_runs(tmp_path / "selection.h5", dl3_dir, dl2_dir)

    assert len(table) == 1
    row = table[0]
    assert row["run_number"] == 13647
    assert row["date"] == 20230709
    assert row["RA_PNT"] == pytest.approx(299.24)
    assert row["DEC_PNT"] == pytest.approx(28.07)
    assert row["AZ_PNT"] == pytest.approx(93.96)
    assert row["ZE_PNT"] == pytest.approx(6.55)
    assert row["B_DELTA"] == pytest.approx(52.34995)
    assert row["LIVETIME"] == pytest.approx(1148.0)
    assert row["N_EVENTS"] == 3
    assert row["NSB_LEVEL"] == pytest.approx(0.22)
    assert table["LIVETIME"].unit == u.s


def test_rejects_multiple_dl3_products_for_one_run(tmp_path, monkeypatch):
    _patch_readers(monkeypatch)
    dl3_dir = tmp_path / "dl3"
    for reduction in ("standard", "hard"):
        directory = dl3_dir / reduction
        directory.mkdir(parents=True)
        (directory / "dl3_LST-1.Run13647.fits").touch()

    with pytest.raises(ValueError, match="Multiple DL3 files"):
        summarize_lstana.summarize_runs(tmp_path / "selection.h5", dl3_dir)


def test_main_reads_paths_from_pipeline_config(tmp_path, monkeypatch, capsys):
    analysis_dir = tmp_path / "analysis"
    dl2_dir = tmp_path / "linked_dl2"
    dl3_dir = tmp_path / "linked_dl3"
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f"""
        [source]
        name = "Test Source"
        ra = 83.632
        dec = 22.0145

        [dl2]
        enabled = true
        source = "link"
        [dl2.link]
        output_directory = "{dl2_dir}"

        [dl3]
        enabled = true
        source = "link"
        [dl3.link]
        output_directory = "{dl3_dir}"
        """
    )
    calls = {}

    def fake_summarize(datacheck, dl3_dir, dl2_dir):
        calls.update(datacheck=datacheck, dl3_dir=dl3_dir, dl2_dir=dl2_dir)
        return Table({"run_number": [13647]})

    monkeypatch.setattr(summarize_lstana, "summarize_runs", fake_summarize)
    summary_file = analysis_dir / "summary.csv"

    summarize_lstana.main([str(config_file), "-o", str(analysis_dir), "--summary-file", str(summary_file)])

    assert calls == {
        "datacheck": analysis_dir / "data_check" / "data_check_Test_Source_ra_83.632_dec_22.0145.h5",
        "dl3_dir": dl3_dir,
        "dl2_dir": dl2_dir,
    }
    assert summary_file.read_text().splitlines() == ["run_number", "13647"]
    assert "Wrote 1 runs" in capsys.readouterr().out


def test_config_file_defaults_to_project_config():
    args = summarize_lstana.build_parser().parse_args([])

    assert args.config_file == summarize_lstana.Path("config/config.toml")
