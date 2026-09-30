from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (side effect: registers compression filters with h5py)
import numpy as np

from .helper import find_lst_subrun_files

PARAMETERS_DATASET = "/dl1/event/telescope/parameters/LST_LSTCam"
IMAGE_DATASET = "/dl1/event/telescope/image/LST_LSTCam"


class ImageLoader:
    def __init__(self, date: int, run_number: int):
        self.date = date
        self.run_number = run_number
        self.subrun_files = find_lst_subrun_files(date, run_number)
        self.index: list[dict] | None = None

    def build_subrun_index(self) -> None:
        """Read the first/last event id of each subrun to enable event lookup."""
        index = []
        for file in self.subrun_files:
            with h5py.File(file, "r") as f:
                event_id = f[PARAMETERS_DATASET]["event_id"]  # pyright: ignore
                first_id = event_id[0]  # pyright: ignore
                last_id = event_id[-1]  # pyright: ignore
                subrun = int(Path(file).stem.rsplit(".", 1)[-1])
                index.append(
                    {
                        "subrun": subrun,
                        "first_id": first_id,
                        "last_id": last_id,
                        "path": file,
                    }
                )
        self.index = index

    def find_subrun(self, event_id: int) -> dict[str, int | str]:
        """Return the subrun file path and subrun number containing ``event_id``.

        Builds the subrun index first if it has not been built yet.

        Raises
        ------
        ValueError
            If ``event_id`` does not fall within any subrun of this run.
        """
        if self.index is None:
            self.build_subrun_index()
        assert self.index is not None

        for entry in self.index:
            if entry["first_id"] <= event_id <= entry["last_id"]:
                return {"subrun": entry["subrun"], "path": entry["path"]}

        raise ValueError(f"event_id {event_id} not found in any subrun of Run{self.run_number:05d}")

    def get_image(self, event_id: int) -> np.void:
        """Return the image entry whose ``event_id`` field matches ``event_id``.

        The image dataset is a structured array, so the entry is looked up by
        its ``event_id`` field rather than by row index.
        """
        subrun_info = self.find_subrun(event_id)
        with h5py.File(subrun_info["path"], "r") as f:
            images = f[IMAGE_DATASET]  # pyright: ignore
            rows = np.flatnonzero(images["event_id"] == event_id)  # pyright: ignore
            if len(rows) == 0:
                raise ValueError(f"event_id {event_id} not found in image dataset of subrun {subrun_info['subrun']}")
            return images[rows[0]]  # pyright: ignore
