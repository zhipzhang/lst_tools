"""Typed SLURM configuration for distributed pipeline stages.

Parsed from the optional ``[slurm]`` section in the pipeline TOML config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


class ConfigError(ValueError):
    """Raised when a configuration section is malformed."""


def _unknown_keys(section: dict, known: set[str], where: str) -> None:
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"Unknown keys in [{where}]: {sorted(unknown)}")


@dataclass(frozen=True)
class SlurmDL2Config:
    """SLURM batching options for the DL2 reconstruction stage."""

    runs_per_job: int = 5
    cpus_per_task: int = 4
    mem: str = "8GB"
    time: str = "02:00:00"
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            return cls()
        _unknown_keys(section, {"runs_per_job", "cpus_per_task", "mem", "time", "extra_args"}, "slurm.dl2")

        try:
            runs_per_job = int(section.get("runs_per_job", 5))
        except (TypeError, ValueError) as err:
            raise ConfigError(f"[slurm.dl2] runs_per_job must be an integer: {err}") from err
        if runs_per_job <= 0:
            raise ConfigError(f"[slurm.dl2] runs_per_job must be positive, got {runs_per_job}")

        try:
            cpus_per_task = int(section.get("cpus_per_task", 4))
        except (TypeError, ValueError) as err:
            raise ConfigError(f"[slurm.dl2] cpus_per_task must be an integer: {err}") from err
        if cpus_per_task <= 0:
            raise ConfigError(f"[slurm.dl2] cpus_per_task must be positive, got {cpus_per_task}")

        mem = str(section.get("mem", "8GB"))
        time = str(section.get("time", "02:00:00"))
        extra_args = dict(section.get("extra_args", {}))

        return cls(
            runs_per_job=runs_per_job,
            cpus_per_task=cpus_per_task,
            mem=mem,
            time=time,
            extra_args=extra_args,
        )


@dataclass(frozen=True)
class SlurmDL3Config:
    """Resources and worker batching for one multiprocess DL3 build job."""

    runs_per_worker: int = 5
    cpus_per_task: int = 4
    mem: str = "16GB"
    time: str = "04:00:00"
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            return cls()
        _unknown_keys(
            section,
            {"runs_per_worker", "cpus_per_task", "mem", "time", "extra_args"},
            "slurm.dl3",
        )
        try:
            runs_per_worker = int(section.get("runs_per_worker", 5))
        except (TypeError, ValueError) as err:
            raise ConfigError(f"[slurm.dl3] runs_per_worker must be an integer: {err}") from err
        if runs_per_worker <= 0:
            raise ConfigError(f"[slurm.dl3] runs_per_worker must be positive, got {runs_per_worker}")
        try:
            cpus_per_task = int(section.get("cpus_per_task", 4))
        except (TypeError, ValueError) as err:
            raise ConfigError(f"[slurm.dl3] cpus_per_task must be an integer: {err}") from err
        if cpus_per_task <= 0:
            raise ConfigError(f"[slurm.dl3] cpus_per_task must be positive, got {cpus_per_task}")
        return cls(
            runs_per_worker=runs_per_worker,
            cpus_per_task=cpus_per_task,
            mem=str(section.get("mem", "16GB")),
            time=str(section.get("time", "04:00:00")),
            extra_args=dict(section.get("extra_args", {})),
        )


@dataclass(frozen=True)
class SlurmConfig:
    """General SLURM configuration for submitting pipeline stages as cluster jobs."""

    enabled: bool = False
    partition: str = "short"
    account: str | None = None
    log_directory: Path = Path("slurm_logs")
    env_setup: str | None = None
    dl2: SlurmDL2Config = field(default_factory=SlurmDL2Config)
    dl3: SlurmDL3Config = field(default_factory=SlurmDL3Config)
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            return cls(enabled=False)
        _unknown_keys(
            section,
            {"enabled", "partition", "account", "log_directory", "env_setup", "dl2", "dl3", "extra_args"},
            "slurm",
        )

        enabled = bool(section.get("enabled", False))
        partition = str(section.get("partition", "short"))
        account = section.get("account")
        if account is not None:
            account = str(account)
        log_directory = Path(section.get("log_directory", "slurm_logs"))
        env_setup = section.get("env_setup")
        if env_setup is not None:
            env_setup = str(env_setup)

        dl2 = SlurmDL2Config.parse(section.get("dl2"))
        dl3 = SlurmDL3Config.parse(section.get("dl3"))
        extra_args = dict(section.get("extra_args", {}))

        return cls(
            enabled=enabled,
            partition=partition,
            account=account,
            log_directory=log_directory,
            env_setup=env_setup,
            dl2=dl2,
            dl3=dl3,
            extra_args=extra_args,
        )

    def submit_job(
        self,
        *,
        job_name: str,
        resources: SlurmDL2Config | SlurmDL3Config,
        log_directory: Path,
        command: str,
    ) -> int:
        from simple_slurm import Slurm

        log_directory.mkdir(parents=True, exist_ok=True)
        slurm_arguments: dict[str, Any] = {
            "job_name": job_name,
            "partition": self.partition,
            "cpus_per_task": resources.cpus_per_task,
            "mem": resources.mem,
            "time": resources.time,
            "output": str(log_directory / "%x_%j.out"),
            "error": str(log_directory / "%x_%j.err"),
        }
        if self.account is not None:
            slurm_arguments["account"] = self.account
        slurm_arguments.update(self.extra_args)
        slurm_arguments.update(resources.extra_args)

        slurm = Slurm(**slurm_arguments)
        slurm.set_shell("/bin/bash")
        if self.env_setup:
            slurm.add_cmd(self.env_setup)
        slurm.add_cmd(command)
        return int(slurm.sbatch())
