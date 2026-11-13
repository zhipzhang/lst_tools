from pathlib import Path

from lst_tools.dl2 import reconstruction


def test_dl2_output_path_uses_the_dl1_file_name(tmp_path):
    dl1_path = Path("/data/dl1_LST-1.Run10623.h5")

    assert reconstruction.dl2_output_path(dl1_path, tmp_path) == (
        tmp_path / "dl2_LST-1.Run10623.h5"
    )


def test_literal_model_template_preserves_input_order():
    dl1_files = [Path("dl1_0002.h5"), Path("dl1_0001.h5")]

    groups = reconstruction.group_dl1_by_model_directory(dl1_files, "/models")

    assert groups == {Path("/models"): dl1_files}


def test_nsb_template_groups_files_by_resolved_model(monkeypatch):
    dl1_files = [Path(f"dl1_000{run}.h5") for run in (1, 2, 3)]
    nsb_by_file = dict(zip(dl1_files, (0.22, 0.38, 0.22), strict=True))
    monkeypatch.setattr(reconstruction, "resolve_dl1_nsb_level", lambda path: nsb_by_file[path])

    groups = reconstruction.group_dl1_by_model_directory(dl1_files, "/models/nsb_$NSB_LEVEL")

    assert groups == {
        Path("/models/nsb_0.22"): [dl1_files[0], dl1_files[2]],
        Path("/models/nsb_0.38"): [dl1_files[1]],
    }


def test_empty_input_does_not_create_output_or_call_lstchain(tmp_path, monkeypatch):
    def unexpected_batch(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("empty input must not call DL1ToDL2Tool")

    monkeypatch.setattr(reconstruction, "dl1_to_dl2_batch", unexpected_batch)
    output_directory = tmp_path / "dl2"

    assert reconstruction.reconstruct_dl1_to_dl2([], output_directory, tmp_path / "missing_models") == []
    assert not output_directory.exists()


def test_reconstruct_returns_only_outputs_available_afterwards(tmp_path, monkeypatch):
    dl1_files = [tmp_path / f"dl1_000{run}.h5" for run in (1, 2)]
    for dl1_file in dl1_files:
        dl1_file.touch()
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    output_directory = tmp_path / "dl2"
    calls = []

    def fake_batch(files, output_directory, model_directory, overwrite=False):
        files = list(files)
        calls.append((files, Path(output_directory), Path(model_directory), overwrite))
        reconstruction.dl2_output_path(files[0], output_directory).touch()

    monkeypatch.setattr(reconstruction, "dl1_to_dl2_batch", fake_batch)

    output_paths = reconstruction.reconstruct_dl1_to_dl2(
        dl1_files,
        output_directory,
        model_directory,
        overwrite=True,
    )

    assert calls == [(dl1_files, output_directory, model_directory, True)]
    assert output_paths == [output_directory / "dl2_0001.h5"]
