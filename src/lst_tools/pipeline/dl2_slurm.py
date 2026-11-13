"""SLURM submission for batches of DL1-to-DL2 reconstruction."""

import shlex
from collections.abc import Sequence
from pathlib import Path

from .slurm_config import SlurmConfig


class DL2SlurmSubmitter:
    """Submit pending DL1 files as batched DL2 reconstruction jobs."""

    def __init__(
        self,
        output_directory: Path,
        log_directory: Path,
        config: SlurmConfig,
        worker_command: str = "lst-dl1-to-dl2",
    ) -> None:
        self.output_directory = output_directory
        self.log_directory = log_directory
        self.config = config
        self.worker_command = worker_command

    def submit(
        self,
        dl1_files: Sequence[Path],
        model_directory_template: str,
        overwrite: bool = False,
    ) -> list[int]:
        if not dl1_files:
            return []

        resources = self.config.dl2
        job_ids = []
        for batch_index, start in enumerate(range(0, len(dl1_files), resources.runs_per_job), start=1):
            batch = dl1_files[start : start + resources.runs_per_job]
            files_argument = " ".join(shlex.quote(str(path)) for path in batch)
            command = (
                f"{self.worker_command} --input {files_argument} "
                f"--output-dir {shlex.quote(str(self.output_directory))} "
                f"--model-dir {shlex.quote(model_directory_template)}"
            )
            if overwrite:
                command += " --overwrite"
            job_ids.append(
                self.config.submit_job(
                    job_name=f"lst_dl2_{batch_index}",
                    resources=resources,
                    log_directory=self.log_directory,
                    command=command,
                )
            )

        return job_ids
