"""
Data level producers: each owns one data level of the workspace.

A producer knows two operations for its level — ``link`` (place files
the standard pipeline already produced on the La Palma server) and
``build`` (produce the level ourselves) — selected by the config's
``source`` mode. The base class owns everything the levels share: the
server search, newest-version selection, and idempotent linking. A subclass
is a declaration: its level, its variant rule, its placement rule, its build
recipe.
"""

import os
import warnings
from abc import ABC
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal

from lst_tools.dl2.reconstruction import dl2_output_path, reconstruct_dl1_to_dl2
from lst_tools.finder import (
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3Attributes,
    DL3LevelSearchResult,
    DLAttributes,
    find_la_palma_files,
)

from .configs import DL1ProducerConfig, DL2ProducerConfig, DL3ProducerConfig, SourceConfig
from .dl2_inputs import find_pending_dl1_files
from .dl2_slurm import DL2SlurmSubmitter
from .dl3 import DL2Observation, DL3Build, DL3Reduction
from .dl3_execution import submit_dl3_build
from .selection import RunSelection
from .slurm_config import SlurmConfig

Level = Literal["dl1", "dl2", "dl3"]

# Each subclass returns a different shape from ``newest_attributes``, so the
# hooks take the union; the DL3 override validates the shape it receives.
AnySearchResult = DL1LevelSearchResult | DL2LevelSearchResult | DL3LevelSearchResult


def create_safe_link(source: Path, destination: Path) -> str:
    """Create a symlink without replacing an existing file or different link.

    Returns "created", "existing" (same target already linked), or
    "conflict" (destination taken; a warning is emitted).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_symlink():
        if Path(os.path.realpath(destination)) == Path(os.path.realpath(source)):
            return "existing"
        warnings.warn(f"Link destination already points elsewhere: {destination}", stacklevel=2)
        return "conflict"

    if destination.exists():
        warnings.warn(f"Refusing to replace existing file: {destination}", stacklevel=2)
        return "conflict"

    destination.symlink_to(source)
    return "created"


class DLProducer(ABC):
    """Owns one data level of the workspace: how its files are placed
    (link) and how the level is produced (build)."""

    level: ClassVar[Level]

    def __init__(self, config, output_root: Path, slurm_config: SlurmConfig | None = None) -> None:
        self.config = config
        self.output_root = output_root
        self.slurm_config = slurm_config

    @property
    def output_dir(self) -> Path:
        """Where this level's files live for the configured source mode:
        the mode options' output_directory, else <output_root>/<level>.

        Link and build may target different directories; only the active
        mode's options survive config parsing (the other mode's are
        cleared with a warning).
        """
        options = self.config.link if self.config.source == "link" else self.config.build
        if options is not None and options.output_directory is not None:
            return options.output_directory
        return self.output_root / self.level

    # ---- public entry -------------------------------------------------
    def produce(self, selection: RunSelection) -> Counter:
        """Produce this level according to the config's source mode."""
        if not self.config.enabled:
            return Counter()
        if self.config.source == "link":
            return self.link(selection)
        if self.slurm_config and self.slurm_config.enabled:
            return self.slurm_build(selection, self.slurm_config)
        return self.build(selection)

    # ---- operations: override only what the level supports ------------
    def link(self, selection: RunSelection) -> Counter:
        raise NotImplementedError(f"{self.level}: linking is not supported")

    def build(self, selection: RunSelection) -> Counter:
        raise NotImplementedError(f"{self.level}: building is not supported yet")

    def slurm_build(self, selection: RunSelection, slurm_config: SlurmConfig) -> Counter:
        raise NotImplementedError(f"{self.level}: SLURM building is not supported yet")

    # ---- linking machinery, shared by all levels ----------------------
    def _link_selection(self, selection: RunSelection) -> Counter:
        """Per run: search the server, select the variant(s), place, safe-link."""
        counts: Counter = Counter()
        for run in selection:
            result = find_la_palma_files(run.date, run.run_number, self.level)
            for attrs in self._select_variants(result):
                destination = self.output_dir / self._link_relpath(attrs)
                counts[f"{self.level}:{create_safe_link(attrs.path, destination)}"] += 1
        return counts

    def _select_variants(self, result: AnySearchResult) -> Sequence[DLAttributes]:
        """DL1/DL2: at most the single newest file. DL3 overrides for variants."""
        newest = result.newest_attributes
        if newest is None:
            return []
        if isinstance(newest, list):  # DL3 shape; DL3 producers override this hook
            return newest
        return [newest]

    def _link_relpath(self, attrs: DLAttributes) -> Path:
        """Place DL1/DL2 files directly below output_dir."""
        return Path(attrs.path.name)


class DL1Producer(DLProducer):
    """DL1 is the external input: linked from the server today; building
    it ourselves is planned and lands as a build() implementation."""

    level: ClassVar[Level] = "dl1"
    config: DL1ProducerConfig

    def link(self, selection: RunSelection) -> Counter:
        return self._link_selection(selection)


class DL2Producer(DLProducer):
    level: ClassVar[Level] = "dl2"
    worker_command: ClassVar[str] = "lst-dl1-to-dl2"
    config: DL2ProducerConfig

    def __init__(
        self,
        config: DL2ProducerConfig,
        output_root: Path,
        dl1_directory: str | Path | None = None,
        slurm_config: SlurmConfig | None = None,
    ):
        super().__init__(config, output_root, slurm_config=slurm_config)
        self.dl1_directory = Path(dl1_directory) if dl1_directory is not None else output_root / "dl1"

    def link(self, selection: RunSelection) -> Counter:
        return self._link_selection(selection)

    def slurm_build(self, selection: RunSelection, slurm_config: SlurmConfig) -> Counter:
        """Submit reconstruction jobs for the selected pending DL1 files."""
        build_options = self.config.build
        if build_options is None or build_options.rf_directory is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("dl2: building requires [dl2.build] with rf_directory")
        rf_template = build_options.rf_directory
        overwrite = build_options.overwrite

        pending, existing_count = find_pending_dl1_files(
            selection,
            self.dl1_directory,
            self.output_dir,
            overwrite=overwrite,
        )
        counts: Counter = Counter()
        if existing_count:
            counts["dl2:existing"] = existing_count
        if not pending:
            return counts

        submitter = DL2SlurmSubmitter(
            output_root=self.output_root,
            output_directory=self.output_dir,
            config=slurm_config,
            worker_command=self.worker_command,
        )
        job_ids = submitter.submit(
            pending,
            model_directory_template=rf_template,
            overwrite=overwrite,
        )
        counts["dl2:slurm_jobs"] = len(job_ids)
        counts["dl2:slurm_runs"] = len(pending)
        return counts

    def build(self, selection: RunSelection) -> Counter:
        """Reconstruct DL2 from the selected DL1 files.

        Selected runs with missing or ambiguous DL1 inputs are skipped with
        a warning. Runs whose DL2 already exists are kept unless
        ``overwrite = true``.

        The rf_directory template's ``$NSB_LEVEL`` placeholder is resolved
        per run from the run's DL1 file (``LSTDL1File.nsb_level``); runs
        are grouped by the resolved model directory before reconstruction,
        so one ``dl1_to_dl2_batch`` call handles all runs of one MC group.
        A template without the placeholder is used as-is for every run.
        """
        build_options = self.config.build
        if build_options is None or build_options.rf_directory is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("dl2: building requires [dl2.build] with rf_directory")
        rf_template = build_options.rf_directory
        overwrite = build_options.overwrite

        pending, existing_count = find_pending_dl1_files(
            selection,
            self.dl1_directory,
            self.output_dir,
            overwrite=overwrite,
        )
        counts: Counter = Counter()
        if existing_count:
            counts["dl2:existing"] = existing_count
        if not pending:
            return counts

        output_paths = reconstruct_dl1_to_dl2(pending, self.output_dir, rf_template, overwrite=overwrite)
        available_outputs = set(output_paths)
        for dl1_file in pending:
            if dl2_output_path(dl1_file, self.output_dir) in available_outputs:
                counts["dl2:built"] += 1
            else:
                warnings.warn(f"DL1ToDL2Tool did not create the DL2 file for {dl1_file}", stacklevel=2)
                counts["dl2:failed"] += 1

        return counts


class DL3Producer(DLProducer):
    level: ClassVar[Level] = "dl3"
    worker_command: ClassVar[str] = "lst-dl3-build"
    config: DL3ProducerConfig

    def __init__(
        self,
        config: DL3ProducerConfig,
        output_root: Path,
        source: SourceConfig | None = None,
        dl2_directory: str | Path | None = None,
        slurm_config: SlurmConfig | None = None,
    ):
        super().__init__(config, output_root, slurm_config=slurm_config)
        self.source = source
        self.dl2_directory = Path(dl2_directory) if dl2_directory is not None else output_root / "dl2"

    def link(self, selection: RunSelection) -> Counter:
        return self._link_selection(selection)

    def _select_variants(self, result: AnySearchResult) -> Sequence[DL3Attributes]:
        """All variants of the newest version, narrowed by the link options."""
        variants = result.newest_attributes
        if not isinstance(variants, list):
            raise TypeError(f"DL3 linking expects a DL3 search result, got {type(result).__name__}")
        if self.config.link is not None and self.config.link.irf_type is not None:
            variants = [variant for variant in variants if variant.irf_type == self.config.link.irf_type]
        return variants

    def _link_relpath(self, attrs: DLAttributes) -> Path:
        """Mirror the server's variant tree below output_dir."""
        if not isinstance(attrs, DL3Attributes):
            raise TypeError(f"DL3 linking expects DL3 attributes, got {type(attrs).__name__}")
        cut_label = f"gheff{attrs.gh_eff}_thetacont{attrs.theta_containment}"
        return Path(attrs.irf_type) / attrs.mc_data_type / cut_label / attrs.path.name

    def build(self, selection: RunSelection) -> Counter:
        return self.prepare_build(selection).run()

    def slurm_build(self, selection: RunSelection, slurm_config: SlurmConfig) -> Counter:
        return submit_dl3_build(self.prepare_build(selection), self.output_root, slurm_config, self.worker_command)

    def prepare_build(self, selection: RunSelection) -> DL3Build:
        options = self.config.build
        if options is None or not options.reductions:
            raise ValueError("dl3: building requires at least one [[dl3.build.reduction]] table")
        if self.source is None:
            raise ValueError("dl3: building requires the analysis source (SourceConfig with name, ra, dec)")
        observations = DL2Observation.from_selection(selection, self.dl2_directory)
        reductions = tuple(
            DL3Reduction.from_options(
                reduction=reduction,
                options=options,
                source=self.source,
                observations=observations,
                irfs_root=options.irf_directory or self.output_root / "irfs",
                output_dir=self.output_dir,
            )
            for reduction in options.reductions
        )
        return DL3Build(observations, reductions, options.mc_dl2_directory)
