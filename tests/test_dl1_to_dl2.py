from pathlib import Path
from types import SimpleNamespace

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

    def test_successful_execution_calls_batch(self, tmp_path, monkeypatch, capsys):
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        dl1_file = tmp_path / "dl1_0001.h5"
        dl1_file.touch()
        out_dir = tmp_path / "out_dl2"

        captured = {}

        def fake_batch(files, out_root, model_directory, overwrite=False):
            captured["files"] = list(files)
            captured["out_root"] = out_root
            captured["model_directory"] = model_directory
            captured["overwrite"] = overwrite

        monkeypatch.setattr(dl1_to_dl2, "dl1_to_dl2_batch", fake_batch)

        dl1_to_dl2.main(
            [
                "-i",
                str(dl1_file),
                "-o",
                str(out_dir),
                "-m",
                str(model_dir),
                "--overwrite",
            ]
        )

        assert captured["files"] == [dl1_file]
        assert captured["out_root"] == out_dir
        assert captured["model_directory"] == model_dir
        assert captured["overwrite"] is True
        assert out_dir.is_dir()

        stdout = capsys.readouterr().out
        assert "Reconstructed 1 DL1 file(s)" in stdout

    def test_nsb_template_groups_runs_by_resolved_model(self, tmp_path, monkeypatch):
        dl1_files = [tmp_path / f"dl1_000{run}.h5" for run in (1, 2, 3)]
        for dl1_file in dl1_files:
            dl1_file.touch()
        model_template = tmp_path / "models" / "nsb_$NSB_LEVEL"
        model_directories = [tmp_path / "models" / "nsb_0.22", tmp_path / "models" / "nsb_0.38"]
        for model_directory in model_directories:
            model_directory.mkdir(parents=True)
        nsb_by_file = dict(zip(dl1_files, (0.22, 0.38, 0.22), strict=True))
        calls = []

        def fake_batch(files, out_root, model_directory, overwrite=False):
            calls.append((list(files), Path(out_root), Path(model_directory), overwrite))

        monkeypatch.setattr(dl1_to_dl2, "dl1_to_dl2_batch", fake_batch)
        monkeypatch.setattr(
            "lst_tools.dl1.dl1event.LSTDL1Event",
            lambda path: SimpleNamespace(nsb_level=nsb_by_file[Path(path)]),
        )

        dl1_to_dl2.main(
            [
                "--input",
                *(str(file) for file in dl1_files),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-dir",
                str(model_template),
            ]
        )

        assert calls == [
            ([dl1_files[0], dl1_files[2]], tmp_path / "out", model_directories[0], False),
            ([dl1_files[1]], tmp_path / "out", model_directories[1], False),
        ]
