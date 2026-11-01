"""
Data level producers: each owns one data level of the workspace.

A producer knows two operations for its level — ``link`` (place files
the standard pipeline already produced on the La Palma server) and
``build`` (produce the level ourselves) — selected by the config's
``source`` mode. The base class owns everything the levels share: the
server search, newest-version selection, idempotent linking, and the
unbinned-run warning. A subclass is a declaration: its level, its
variant rule, its placement rule, its build recipe.
"""

import os
import shlex
import warnings
from abc import ABC
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from lst_tools.dl1.dl1event import dl1_to_dl2_batch, group_dl1_by_model_directory
from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.dl3 import DL3Reducer
from lst_tools.finder import (
    MC_DL2_PATH,
    DL1LevelSearchResult,
    DL2LevelSearchResult,
    DL3LevelSearchResult,
    find_la_palma_files,
    find_mc_dl2_files,
)
from lst_tools.finder.datalevel_attributes import DL3Attributes, DLAttributes
from lst_tools.irf import IRFConfig, IRFGenerator, IRFNode, MCNode

from .configs import (
    NSB_LEVEL_VARIABLE,
    DL1ProducerConfig,
    DL2ProducerConfig,
    DL3ProducerConfig,
    DL3ReductionOptions,
    SourceConfig,
)
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
                destination = self.output_dir / self._link_relpath(attrs, run.zenith_bin)
                counts[f"{self.level}:{create_safe_link(attrs.path, destination)}"] += 1
        if selection.unbinned_run_numbers:
            warnings.warn(
                f"{self.level}: {len(selection.unbinned_run_numbers)} runs have no zenith bin; skipped",
                stacklevel=2,
            )
            counts[f"{self.level}:unbinned"] += len(selection.unbinned_run_numbers)
        return counts

    def _select_variants(self, result: AnySearchResult) -> Sequence[DLAttributes]:
        """DL1/DL2: at most the single newest file. DL3 overrides for variants."""
        newest = result.newest_attributes
        if newest is None:
            return []
        if isinstance(newest, list):  # DL3 shape; DL3 producers override this hook
            return newest
        return [newest]

    def _link_relpath(self, attrs: DLAttributes, zenith_bin: str) -> Path:
        """Placement below output_dir: <zd_bin>/<file> for DL1/DL2."""
        return Path(zenith_bin) / attrs.path.name


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
        """Submit DL2 batch reconstruction jobs to SLURM via simple_slurm."""
        from simple_slurm import Slurm

        build_options = self.config.build
        if build_options is None or build_options.rf_directory is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("dl2: building requires [dl2.build] with rf_directory")
        rf_template = build_options.rf_directory
        overwrite = build_options.overwrite

        counts: Counter = Counter()
        if not self.dl1_directory.is_dir():
            warnings.warn(f"dl2: no DL1 directory at {self.dl1_directory}; skipping", stacklevel=2)
            return counts

        log_dir = self.output_root / slurm_config.log_directory
        log_dir.mkdir(parents=True, exist_ok=True)

        slurm_kwargs: dict[str, Any] = {
            "partition": slurm_config.partition,
            "cpus_per_task": slurm_config.dl2.cpus_per_task,
            "mem": slurm_config.dl2.mem,
            "time": slurm_config.dl2.time,
            "output": str(log_dir / "%x_%j.out"),
            "error": str(log_dir / "%x_%j.err"),
        }
        if slurm_config.account is not None:
            slurm_kwargs["account"] = slurm_config.account
        slurm_kwargs.update(slurm_config.extra_args)
        slurm_kwargs.update(slurm_config.dl2.extra_args)

        runs_per_job = slurm_config.dl2.runs_per_job

        for dl1_dir in sorted(self.dl1_directory.iterdir()):
            if not dl1_dir.is_dir():
                continue
            output_dir = self.output_dir / dl1_dir.name
            dl1_files = sorted(dl1_dir.glob("dl1_*.h5"))
            pending = [
                dl1_file
                for dl1_file in dl1_files
                if overwrite or not (output_dir / dl1_file.name.replace("dl1", "dl2", 1)).exists()
            ]
            counts["dl2:existing"] += len(dl1_files) - len(pending)
            if not pending:
                continue

            batch_idx = 1
            for i in range(0, len(pending), runs_per_job):
                chunk = pending[i : i + runs_per_job]
                job_name = f"lst_dl2_{dl1_dir.name}_{batch_idx}"
                batch_idx += 1
                slurm = Slurm(**{**slurm_kwargs, "job_name": job_name})
                slurm.set_shell("/bin/bash")
                if slurm_config.env_setup:
                    slurm.add_cmd(slurm_config.env_setup)

                files_arg = " ".join(shlex.quote(str(file)) for file in chunk)
                cmd = (
                    f"{self.worker_command} --input {files_arg} "
                    f"--output-dir {shlex.quote(str(output_dir))} "
                    f"--model-dir {shlex.quote(rf_template)}"
                )
                if overwrite:
                    cmd += " --overwrite"
                slurm.add_cmd(cmd)

                slurm.sbatch()
                counts["dl2:slurm_jobs"] += 1
                counts["dl2:slurm_runs"] += len(chunk)

        return counts

    def build(self, selection: RunSelection) -> Counter:
        """Reconstruct DL2 from the linked DL1 files, per zenith bin.

        Runs whose DL2 already exists are kept unless ``overwrite = true``.

        The rf_directory template's ``$NSB_LEVEL`` placeholder is resolved
        per run from the run's DL1 file (``LSTDL1Event.nsb_level``); runs
        are grouped by the resolved model directory before reconstruction,
        so one ``dl1_to_dl2_batch`` call handles all runs of one MC group.
        A template without the placeholder is used as-is for every run.
        """
        build_options = self.config.build
        if build_options is None or build_options.rf_directory is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("dl2: building requires [dl2.build] with rf_directory")
        rf_template = build_options.rf_directory
        overwrite = build_options.overwrite

        counts: Counter = Counter()
        if not self.dl1_directory.is_dir():
            warnings.warn(f"dl2: no DL1 directory at {self.dl1_directory}; skipping", stacklevel=2)
            return counts

        for dl1_dir in sorted(self.dl1_directory.iterdir()):
            if not dl1_dir.is_dir():
                continue
            output_dir = self.output_dir / dl1_dir.name
            dl1_files = sorted(dl1_dir.glob("dl1_*.h5"))
            pending = [
                dl1_file
                for dl1_file in dl1_files
                if overwrite or not (output_dir / dl1_file.name.replace("dl1", "dl2", 1)).exists()
            ]
            counts["dl2:existing"] += len(dl1_files) - len(pending)
            if not pending:
                continue

            for model_directory, group in group_dl1_by_model_directory(pending, rf_template).items():
                dl1_to_dl2_batch(group, output_dir, model_directory, overwrite=overwrite)
                for dl1_file in group:
                    if (output_dir / dl1_file.name.replace("dl1", "dl2", 1)).exists():
                        counts["dl2:built"] += 1
                    else:
                        warnings.warn(f"DL1ToDL2Tool did not create the DL2 file for {dl1_file}", stacklevel=2)
                        counts["dl2:failed"] += 1

        return counts


def _mc_campaign_directory(directory_template: str, nsb_level: float) -> Path:
    directory = Path(directory_template.replace(NSB_LEVEL_VARIABLE, f"{nsb_level:.2f}")).expanduser()
    return directory if directory.is_absolute() else MC_DL2_PATH / directory


def _mc_nodes_for_observation(
    dl2_table: LSTDL2EventTable,
    directory_template: str,
    irf_type: Literal["full-enclosure", "point"],
) -> list[MCNode]:
    """The MC nodes of one real DL2 file's provenance (its MC group)."""
    if dl2_table.nsb_level is None:
        raise ValueError("DL2 table does not contain an NSB level")
    if dl2_table.dec_line is None:
        raise ValueError("DL2 table does not contain a declination line")

    campaign = _mc_campaign_directory(directory_template, dl2_table.nsb_level)
    mc_type = "Gamma" if irf_type == "point" else "GammaDiffuse"
    return [
        MCNode(
            nsb_level=attribute.nsb,
            dec_line=attribute.dec,
            zenith=attribute.theta,
            azimuth=attribute.azimuth,
            dl2_path=attribute.path,
            irf_type=irf_type,
        )
        for attribute in find_mc_dl2_files(campaign)
        if attribute.dec == dl2_table.dec_line and attribute.mc_type == mc_type
    ]


class DL3Producer(DLProducer):
    level: ClassVar[Level] = "dl3"
    config: DL3ProducerConfig

    def __init__(
        self,
        config: DL3ProducerConfig,
        output_root: Path,
        source: SourceConfig | None = None,
        dl2_directory: str | Path | None = None,
    ):
        super().__init__(config, output_root)
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

    def _link_relpath(self, attrs: DLAttributes, zenith_bin: str) -> Path:
        """The variant tree mirrors the server's own path order with the
        zenith bin prepended: <zd_bin>/<irf_type>/<mc_type>/<cut_label>/<file>."""
        if not isinstance(attrs, DL3Attributes):
            raise TypeError(f"DL3 linking expects DL3 attributes, got {type(attrs).__name__}")
        cut_label = f"gheff{attrs.gh_eff}_thetacont{attrs.theta_containment}"
        return Path(zenith_bin) / attrs.irf_type / attrs.mc_data_type / cut_label / attrs.path.name

    def build(self, selection: RunSelection) -> Counter:
        """Build every configured reduction in its own IRF and DL3 namespace."""
        build_options = self.config.build
        if build_options is None or not build_options.reductions:  # pragma: no cover - guarded by config parsing
            raise ValueError("dl3: building requires at least one [[dl3.build.reduction]] table")
        if self.source is None:
            raise ValueError("dl3: building requires the analysis source (SourceConfig with name, ra, dec)")

        counts: Counter = Counter()
        if not self.dl2_directory.is_dir():
            warnings.warn(f"dl3: no DL2 directory at {self.dl2_directory}; skipping", stacklevel=2)
            return counts

        for reduction in build_options.reductions:
            self._build_reduction(reduction, self.source, counts)
        return counts

    def _build_reduction(
        self,
        reduction: DL3ReductionOptions,
        source: SourceConfig,
        counts: Counter,
    ) -> None:
        build_options = self.config.build
        if build_options is None:  # pragma: no cover - guarded by build()
            raise ValueError("dl3: building requires [dl3.build]")

        irf_config = IRFConfig(reduction.event_filter, point_like=reduction.point_like)
        event_filter = irf_config.event_filter
        irfs_root = (build_options.irf_directory or self.output_root / "irfs") / reduction.name
        generator = IRFGenerator(irfs_root, irf_config)

        # One IRF per MC node, deduplicated across DL2 files of the same group.
        nodes: dict[Path, IRFNode] = {}
        for dl2_file in sorted(self.dl2_directory.glob("*/*.h5")):
            table = LSTDL2EventTable(dl2_file)
            mc_nodes = _mc_nodes_for_observation(
                table,
                build_options.mc_dl2_directory,
                irf_config.irf_type,
            )
            for mc_node in mc_nodes:
                node = IRFNode.from_mc_node(mc_node, gh_type=event_filter.gh_type, gh_value=event_filter.gh_value)
                nodes.setdefault(node.relative_path, node)
        for node in nodes.values():
            generator.make_irf_node(node)
        counts["dl3:irf_nodes"] += len(nodes)

        reducer = DL3Reducer(
            irfs_root=generator.base_path,
            source_name=source.name,
            source_ra=source.ra,
            source_dec=source.dec,
            irf_config=irf_config,
            interp_method=build_options.interp_method,
            use_nearest_irf_node=build_options.use_nearest_irf_node,
            overwrite=build_options.overwrite,
        )
        for dl2_dir in sorted(self.dl2_directory.iterdir()):
            if not dl2_dir.is_dir():
                continue
            dl2_files = sorted(dl2_dir.glob("dl2_*.h5"))
            if not dl2_files:
                counts["dl3:no_dl2"] += 1
                continue
            reduced = reducer.reduce(dl2_files, self.output_dir / reduction.name / dl2_dir.name)
            counts["dl3:reduced"] += len(reduced)
            counts["dl3:failed"] += len(dl2_files) - len(reduced)
