"""Process execution and SLURM submission for a selected DL3 build."""

import os
import pickle
import shlex
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from .dl3 import DL3Build, DL3Reduction, IRFSet
from .slurm_config import SlurmConfig

Task = TypeVar("Task")
Result = TypeVar("Result")


class DL3Executor:
    def __init__(self, max_workers: int | None = None):
        if max_workers is not None and max_workers <= 0:
            raise ValueError("DL3 max_workers must be positive")
        self.max_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)

    def prepare_irfs(self, irf_sets: Sequence[IRFSet]) -> None:
        self._map(IRFSet.prepare, irf_sets)

    def reduce_observations(self, reductions: Sequence[DL3Reduction]) -> Counter:
        counts: Counter = Counter()
        for result in self._map(DL3Reduction.reduce_observations, reductions):
            counts.update(result)
        return counts

    def _map(self, worker: Callable[[Task], Result], tasks: Sequence[Task]) -> list[Result]:
        if not tasks:
            return []
        with ProcessPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
            return list(executor.map(worker, tasks))


def submit_dl3_build(build: DL3Build, output_root: Path, config: SlurmConfig, worker_command: str) -> Counter:
    from simple_slurm import Slurm

    build.validate_inputs()
    if not build.observations:
        return Counter()
    job_dir = (output_root / "slurm_jobs").absolute()
    job_dir.mkdir(parents=True, exist_ok=True)
    build_pickle = job_dir / f"dl3_build_{uuid4().hex}.pkl"
    with build_pickle.open("wb") as file_handle:
        pickle.dump(build, file_handle)

    log_dir = (output_root / config.log_directory).absolute()
    log_dir.mkdir(parents=True, exist_ok=True)
    resources = config.dl3
    slurm_kwargs = {
        "partition": config.partition,
        "cpus_per_task": resources.cpus_per_task,
        "mem": resources.mem,
        "time": resources.time,
        "output": str(log_dir / "%x_%j.out"),
        "error": str(log_dir / "%x_%j.err"),
        "job_name": "lst_dl3_build",
    }
    if config.account is not None:
        slurm_kwargs["account"] = config.account
    slurm_kwargs.update(config.extra_args)
    slurm_kwargs.update(resources.extra_args)

    slurm = Slurm(**slurm_kwargs)
    slurm.set_shell("/bin/bash")
    if config.env_setup:
        slurm.add_cmd(config.env_setup)
    slurm.add_cmd(
        f"{worker_command} --build-pickle {shlex.quote(str(build_pickle))} "
        f"--max-workers {int(slurm_kwargs['cpus_per_task'])}"
    )
    slurm.sbatch()
    return Counter({"dl3:slurm_jobs": 1, "dl3:slurm_reductions": len(build.reductions)})
