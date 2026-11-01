from pathlib import Path

import pytest

from lst_tools.pipeline import PipelineConfig, SlurmConfig, SlurmDL2Config, SlurmDL3Config
from lst_tools.pipeline.slurm_config import ConfigError


class TestSlurmDL2Config:
    def test_default_values_when_none(self):
        config = SlurmDL2Config.parse(None)
        assert config.runs_per_job == 5
        assert config.cpus_per_task == 4
        assert config.mem == "8GB"
        assert config.time == "02:00:00"
        assert config.extra_args == {}

    def test_default_values_when_empty_dict(self):
        config = SlurmDL2Config.parse({})
        assert config.runs_per_job == 5
        assert config.cpus_per_task == 4
        assert config.mem == "8GB"
        assert config.time == "02:00:00"

    def test_custom_values(self):
        config = SlurmDL2Config.parse(
            {
                "runs_per_job": 10,
                "cpus_per_task": 8,
                "mem": "16GB",
                "time": "04:30:00",
                "extra_args": {"gres": "gpu:1"},
            }
        )
        assert config.runs_per_job == 10
        assert config.cpus_per_task == 8
        assert config.mem == "16GB"
        assert config.time == "04:30:00"
        assert config.extra_args == {"gres": "gpu:1"}

    @pytest.mark.parametrize("invalid_runs", [0, -1, "invalid"])
    def test_invalid_runs_per_job_raises(self, invalid_runs):
        with pytest.raises(ConfigError, match=r"runs_per_job"):
            SlurmDL2Config.parse({"runs_per_job": invalid_runs})

    @pytest.mark.parametrize("invalid_cpus", [0, -2, "abc"])
    def test_invalid_cpus_per_task_raises(self, invalid_cpus):
        with pytest.raises(ConfigError, match=r"cpus_per_task"):
            SlurmDL2Config.parse({"cpus_per_task": invalid_cpus})

    def test_unknown_keys_rejected(self):
        with pytest.raises(ConfigError, match="Unknown keys in \\[slurm.dl2\\]"):
            SlurmDL2Config.parse({"unknown_key": 123})


class TestSlurmDL3Config:
    def test_default_values(self):
        config = SlurmDL3Config.parse(None)
        assert config.cpus_per_task == 4
        assert config.mem == "16GB"
        assert config.time == "04:00:00"
        assert config.extra_args == {}

    def test_custom_values(self):
        config = SlurmDL3Config.parse(
            {"cpus_per_task": 12, "mem": "32GB", "time": "06:00:00", "extra_args": {"qos": "priority"}}
        )
        assert config.cpus_per_task == 12
        assert config.mem == "32GB"
        assert config.time == "06:00:00"
        assert config.extra_args == {"qos": "priority"}


class TestSlurmConfig:
    def test_default_values_when_none(self):
        config = SlurmConfig.parse(None)
        assert config.enabled is False
        assert config.partition == "short"
        assert config.account is None
        assert config.log_directory == Path("slurm_logs")
        assert config.env_setup is None
        assert config.dl2 == SlurmDL2Config()
        assert config.dl3 == SlurmDL3Config()
        assert config.extra_args == {}

    def test_custom_values(self):
        config = SlurmConfig.parse(
            {
                "enabled": True,
                "partition": "long",
                "account": "aswg",
                "log_directory": "/data/logs",
                "env_setup": "conda activate lst-env",
                "dl2": {"runs_per_job": 3},
                "dl3": {"cpus_per_task": 12},
                "extra_args": {"qos": "priority"},
            }
        )
        assert config.enabled is True
        assert config.partition == "long"
        assert config.account == "aswg"
        assert config.log_directory == Path("/data/logs")
        assert config.env_setup == "conda activate lst-env"
        assert config.dl2.runs_per_job == 3
        assert config.dl3.cpus_per_task == 12
        assert config.extra_args == {"qos": "priority"}

    def test_unknown_keys_rejected(self):
        with pytest.raises(ConfigError, match="Unknown keys in \\[slurm\\]"):
            SlurmConfig.parse({"unsupported_field": True})


class TestPipelineConfigWithSlurm:
    def test_pipeline_config_parses_slurm(self):
        raw = {
            "dl1": {},
            "slurm": {
                "enabled": True,
                "partition": "cpu",
                "dl2": {"runs_per_job": 2},
            },
        }
        pipeline = PipelineConfig.parse(raw)
        assert pipeline.slurm.enabled is True
        assert pipeline.slurm.partition == "cpu"
        assert pipeline.slurm.dl2.runs_per_job == 2

    def test_pipeline_config_defaults_slurm_when_missing(self):
        pipeline = PipelineConfig.parse({"dl1": {}})
        assert pipeline.slurm.enabled is False
        assert pipeline.slurm.partition == "short"
        assert pipeline.slurm.dl2.runs_per_job == 5
