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
class SlurmConfig:
    """General SLURM configuration for submitting pipeline stages as cluster jobs."""

    enabled: bool = False
    partition: str = "short"
    account: str | None = None
    log_directory: Path = Path("slurm_logs")
    env_setup: str | None = None
    dl2: SlurmDL2Config = field(default_factory=SlurmDL2Config)
    extra_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            return cls(enabled=False)
        _unknown_keys(
            section,
            {"enabled", "partition", "account", "log_directory", "env_setup", "dl2", "extra_args"},
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

        dl2_section = section.get("dl2")
        dl2 = SlurmDL2Config.parse(dl2_section)
        extra_args = dict(section.get("extra_args", {}))

        return cls(
            enabled=enabled,
            partition=partition,
            account=account,
            log_directory=log_directory,
            env_setup=env_setup,
            dl2=dl2,
            extra_args=extra_args,
        )
