from pathlib import Path

from lst_tools.pipeline.dl2_slurm import DL2SlurmSubmitter
from lst_tools.pipeline.slurm_config import SlurmConfig, SlurmDL2Config


def test_submits_pending_files_in_batches(tmp_path, monkeypatch):
    config = SlurmConfig(
        enabled=True,
        partition="short",
        account="aswg",
        env_setup="conda activate lst-env",
        dl2=SlurmDL2Config(runs_per_job=2, cpus_per_task=8, mem="40GB", time="02:00:00"),
    )
    submitter = DL2SlurmSubmitter(tmp_path / "out", tmp_path / "out" / "dl2", config)
    pending = [Path(f"/data/dl1_LST-1.Run{run_number:05d}.h5") for run_number in (10001, 10002, 10003)]
    submitted_scripts = []

    def fake_sbatch(slurm):
        submitted_scripts.append(slurm.script())
        return len(submitted_scripts)

    import simple_slurm

    monkeypatch.setattr(simple_slurm.Slurm, "sbatch", fake_sbatch)

    job_ids = submitter.submit(
        pending,
        model_directory_template="/models/nsb_tuning_$NSB_LEVEL/dec_347",
        overwrite=True,
    )

    assert job_ids == [1, 2]
    assert len(submitted_scripts) == 2
    assert str(pending[0]) in submitted_scripts[0]
    assert str(pending[1]) in submitted_scripts[0]
    assert str(pending[2]) not in submitted_scripts[0]
    assert str(pending[2]) in submitted_scripts[1]
    assert "#SBATCH --account             aswg" in submitted_scripts[0]
    assert "#SBATCH --partition           short" in submitted_scripts[0]
    assert "#SBATCH --cpus-per-task       8" in submitted_scripts[0]
    assert "#SBATCH --mem                 40GB" in submitted_scripts[0]
    assert "conda activate lst-env" in submitted_scripts[0]
    assert r"--model-dir '/models/nsb_tuning_\$NSB_LEVEL/dec_347'" in submitted_scripts[0]
    assert submitted_scripts[0].rstrip().endswith("--overwrite")


def test_empty_pending_submits_no_job(tmp_path, monkeypatch):
    import simple_slurm

    def unexpected_submission(slurm):  # pragma: no cover - must not be called
        raise AssertionError("an empty pending list must not create a SLURM job")

    monkeypatch.setattr(simple_slurm.Slurm, "sbatch", unexpected_submission)
    submitter = DL2SlurmSubmitter(tmp_path / "out", tmp_path / "out" / "dl2", SlurmConfig(enabled=True))

    assert submitter.submit([], "/models") == []
    assert not (tmp_path / "out" / "slurm_logs").exists()
