"""SLURM submission for named DL3 reductions."""

import shlex
from collections.abc import Sequence
from pathlib import Path

from .configs import DL3ReductionOptions, SourceConfig
from .slurm_config import SlurmConfig


class DL3SlurmSubmitter:
    """Submit named DL3 reductions as SLURM jobs."""

    def __init__(
        self,
        output_directory: Path,
        irf_directory: Path,
        log_directory: Path,
        config: SlurmConfig,
        worker_command: str = "lst-dl2-to-dl3",
    ) -> None:
        self.output_directory = output_directory
        self.irf_directory = irf_directory
        self.log_directory = log_directory
        self.config = config
        self.worker_command = worker_command

    def submit(
        self,
        dl2_files: Sequence[Path],
        *,
        source: SourceConfig,
        reductions: Sequence[DL3ReductionOptions],
        mc_dl2_directory: str,
        interp_method: str = "linear",
        use_nearest_irf_node: bool = False,
        overwrite: bool = False,
    ) -> list[int]:
        if not dl2_files or not reductions:
            return []

        resources = self.config.dl3
        files_argument = " ".join(shlex.quote(str(path.absolute())) for path in dl2_files)
        job_ids = []
        for reduction in reductions:
            reduction_irf_dir = (self.irf_directory / reduction.name).absolute()
            reduction_output_dir = (self.output_directory / reduction.name).absolute()
            irf_config_path = reduction.irf_config.write_irf_config(reduction_irf_dir / "irf_config.json")
            command = (
                f"{self.worker_command} --input {files_argument} "
                f"--output-dir {shlex.quote(str(reduction_output_dir))} "
                f"--irf-dir {shlex.quote(str(reduction_irf_dir))} "
                f"--irf-config {shlex.quote(str(irf_config_path))} "
                f"--mc-dir {shlex.quote(mc_dl2_directory)} "
                f"--source-name {shlex.quote(source.name)} "
                f"--source-ra {source.ra} "
                f"--source-dec {source.dec} "
                f"--interp-method {shlex.quote(interp_method)} "
                f"--max-workers {resources.cpus_per_task} "
                f"--runs-per-worker {resources.runs_per_worker}"
            )
            if use_nearest_irf_node:
                command += " --use-nearest-irf-node"
            if overwrite:
                command += " --overwrite"
            job_ids.append(
                self.config.submit_job(
                    job_name=f"lst_dl3_{reduction.name}",
                    resources=resources,
                    log_directory=self.log_directory,
                    command=command,
                )
            )

        return job_ids
