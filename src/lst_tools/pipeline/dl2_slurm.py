"""SLURM submission for batches of DL1-to-DL2 reconstruction."""

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .slurm_config import SlurmConfig


class DL2SlurmSubmitter:
    """Submit pending DL1 files as batched DL2 reconstruction jobs."""

    def __init__(
        self,
        output_root: Path,
        output_directory: Path,
        config: SlurmConfig,
        worker_command: str = "lst-dl1-to-dl2",
    ) -> None:
        self.output_root = output_root
        self.output_directory = output_directory
        self.config = config
        self.worker_command = worker_command

    def submit(
        self,
        dl1_files: Sequence[Path],
        model_directory_template: str,
        overwrite: bool = False,
    ) -> list[int]:
        from simple_slurm import Slurm

        if not dl1_files:
            return []

        log_directory = self.output_root / self.config.log_directory
        log_directory.mkdir(parents=True, exist_ok=True)
        resources = self.config.dl2
        slurm_arguments: dict[str, Any] = {
            "partition": self.config.partition,
            "cpus_per_task": resources.cpus_per_task,
            "mem": resources.mem,
            "time": resources.time,
            "output": str(log_directory / "%x_%j.out"),
            "error": str(log_directory / "%x_%j.err"),
        }
        if self.config.account is not None:
            slurm_arguments["account"] = self.config.account
        slurm_arguments.update(self.config.extra_args)
        slurm_arguments.update(resources.extra_args)

        job_ids = []
        for batch_index, start in enumerate(range(0, len(dl1_files), resources.runs_per_job), start=1):
            batch = dl1_files[start : start + resources.runs_per_job]
            slurm = Slurm(**{**slurm_arguments, "job_name": f"lst_dl2_{batch_index}"})
            slurm.set_shell("/bin/bash")
            if self.config.env_setup:
                slurm.add_cmd(self.config.env_setup)

            files_argument = " ".join(shlex.quote(str(path)) for path in batch)
            command = (
                f"{self.worker_command} --input {files_argument} "
                f"--output-dir {shlex.quote(str(self.output_directory))} "
                f"--model-dir {shlex.quote(model_directory_template)}"
            )
            if overwrite:
                command += " --overwrite"
            slurm.add_cmd(command)
            job_ids.append(slurm.sbatch())

        return job_ids
