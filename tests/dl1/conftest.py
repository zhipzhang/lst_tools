import pytest


@pytest.fixture
def make_dl1_path(tmp_path):
    def _make(run_number: int = 10623):
        path = tmp_path / "v0.10" / "tailcut1005" / f"dl1_LST-1.Run{run_number:05d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    return _make
