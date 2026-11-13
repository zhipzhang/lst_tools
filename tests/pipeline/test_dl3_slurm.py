import json

from lst_tools.irf import IRFConfig
from lst_tools.pipeline.configs import DL3ReductionOptions, SourceConfig
from lst_tools.pipeline.dl3_slurm import DL3SlurmSubmitter
from lst_tools.pipeline.slurm_config import SlurmConfig, SlurmDL3Config


def make_reduction(name: str = "standard", intensity_cut: float = 80) -> DL3ReductionOptions:
    return DL3ReductionOptions(
        name=name,
        irf_config=IRFConfig.from_cuts(intensity_cut, 1.0, gh_cut=0.7),
    )


def test_submits_one_job_per_reduction_and_writes_irf_config_json(tmp_path, monkeypatch):
    config = SlurmConfig(
        enabled=True,
        partition="long",
        account="aswg",
        env_setup="conda activate lst-env",
        dl3=SlurmDL3Config(runs_per_worker=3, cpus_per_task=8, mem="32GB", time="06:00:00"),
    )
    submitter = DL3SlurmSubmitter(
        output_directory=tmp_path / "out" / "dl3",
        irf_directory=tmp_path / "out" / "irfs",
        log_directory=tmp_path / "logs",
        config=config,
    )
    dl2_files = [
        tmp_path / "dl2_LST-1.Run10001.h5",
        tmp_path / "dl2_LST-1.Run10002.h5",
    ]
    reductions = (
        make_reduction("standard", 80),
        make_reduction("tight", 150),
    )
    submitted_scripts = []

    def fake_sbatch(slurm):
        submitted_scripts.append(slurm.script())
        return len(submitted_scripts)

    import simple_slurm

    monkeypatch.setattr(simple_slurm.Slurm, "sbatch", fake_sbatch)

    job_ids = submitter.submit(
        dl2_files,
        source=SourceConfig("Crab Nebula", 83.632, 22.0145),
        reductions=reductions,
        mc_dl2_directory="/mc/nsb_$NSB_LEVEL",
        interp_method="nearest",
        use_nearest_irf_node=True,
        overwrite=True,
    )

    assert job_ids == [1, 2]
    assert len(submitted_scripts) == 2

    for index, reduction in enumerate(reductions):
        config_path = tmp_path / "out" / "irfs" / reduction.name / "irf_config.json"
        assert config_path.is_file()
        assert json.loads(config_path.read_text()) == reduction.irf_config.to_irf_config()

        script = submitted_scripts[index]
        assert f"#SBATCH --job-name            lst_dl3_{reduction.name}" in script
        assert "#SBATCH --account             aswg" in script
        assert "#SBATCH --partition           long" in script
        assert "#SBATCH --cpus-per-task       8" in script
        assert "#SBATCH --mem                 32GB" in script
        assert "conda activate lst-env" in script
        assert str(dl2_files[0]) in script and str(dl2_files[1]) in script
        assert str(tmp_path / "out" / "dl3" / reduction.name) in script
        assert str(config_path) in script
        assert "--source-name 'Crab Nebula'" in script
        assert "--source-ra 83.632 --source-dec 22.0145" in script
        assert "--interp-method nearest" in script
        assert "--max-workers 8 --runs-per-worker 3" in script
        assert "--use-nearest-irf-node --overwrite" in script


def test_empty_inputs_submit_no_job(tmp_path, monkeypatch):
    import simple_slurm

    def unexpected_submission(slurm):  # pragma: no cover - must not be called
        raise AssertionError("an empty input list must not create a SLURM job")

    monkeypatch.setattr(simple_slurm.Slurm, "sbatch", unexpected_submission)
    submitter = DL3SlurmSubmitter(
        output_directory=tmp_path / "out" / "dl3",
        irf_directory=tmp_path / "out" / "irfs",
        log_directory=tmp_path / "logs",
        config=SlurmConfig(enabled=True),
    )

    assert (
        submitter.submit(
            [],
            source=SourceConfig("Crab", 83.6, 22.0),
            reductions=(make_reduction(),),
            mc_dl2_directory="/mc",
        )
        == []
    )
    assert not (tmp_path / "logs").exists()
