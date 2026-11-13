from pathlib import Path

import pytest

from lst_tools.scripts import dl1_to_dl2


class TestBuildParser:
    def test_required_arguments(self):
        parser = dl1_to_dl2.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_valid_arguments(self, tmp_path):
        parser = dl1_to_dl2.build_parser()
        f1 = tmp_path / "dl1_run1.h5"
        f2 = tmp_path / "dl1_run2.h5"
        out = tmp_path / "dl2"
        model = tmp_path / "models"

        args = parser.parse_args(
            [
                "--input",
                str(f1),
                str(f2),
                "--output-dir",
                str(out),
                "--model-dir",
                str(model),
                "--overwrite",
            ]
        )
        assert args.input_files == [f1, f2]
        assert args.output_dir == out
        assert args.model_dir == model
        assert args.overwrite is True


class TestMain:
    def test_nonexistent_model_dir_raises(self, tmp_path):
        f = tmp_path / "dl1.h5"
        f.touch()
        non_existent_model = tmp_path / "no_model"

        with pytest.raises(SystemExit):
            dl1_to_dl2.main(
                [
                    "--input",
                    str(f),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--model-dir",
                    str(non_existent_model),
                ]
            )

    def test_nonexistent_input_file_raises(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        missing_file = tmp_path / "missing.h5"

        with pytest.raises(SystemExit):
            dl1_to_dl2.main(
                [
                    "--input",
                    str(missing_file),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--model-dir",
                    str(model_dir),
                ]
            )

    def test_passes_arguments_to_reconstruction_and_reports_actual_outputs(self, tmp_path, monkeypatch, capsys):
        dl1_file = tmp_path / "dl1_0001.h5"
        output_directory = tmp_path / "out_dl2"
        model_directory = tmp_path / "models"
        captured = {}

        def fake_reconstruct(files, output_directory, model_directory_template, overwrite=False):
            captured["files"] = list(files)
            captured["output_directory"] = output_directory
            captured["model_directory_template"] = model_directory_template
            captured["overwrite"] = overwrite
            return [Path(output_directory) / "dl2_0001.h5"]

        monkeypatch.setattr(dl1_to_dl2, "reconstruct_dl1_to_dl2", fake_reconstruct)

        dl1_to_dl2.main(
            [
                "-i",
                str(dl1_file),
                "-o",
                str(output_directory),
                "-m",
                str(model_directory),
                "--overwrite",
            ]
        )

        assert captured == {
            "files": [dl1_file],
            "output_directory": output_directory,
            "model_directory_template": model_directory,
            "overwrite": True,
        }
        stdout = capsys.readouterr().out
        assert "1 of 1 input file(s) have DL2 outputs" in stdout
