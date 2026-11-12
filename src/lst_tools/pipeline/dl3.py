"""Selected observations and the named reductions that produce their DL3 files."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lst_tools.dl2 import LSTDL2EventTable
from lst_tools.dl3 import DL3Reducer
from lst_tools.finder import MC_DL2_PATH, MCDL2Attribute, find_mc_dl2_files, parse_run_id
from lst_tools.irf import IRFConfig, IRFGenerator, IRFNode, MCNode, irf_group_path

from .configs import NSB_LEVEL_VARIABLE, DL3BuildOptions, DL3ReductionOptions, SourceConfig
from .selection import RunSelection


@dataclass(frozen=True)
class DL2Observation:
    path: Path

    @classmethod
    def from_selection(cls, selection: RunSelection, directory: Path) -> tuple["DL2Observation", ...]:
        files_by_run: dict[int, list[Path]] = {}
        for path in sorted(directory.glob("dl2_*.h5")):
            if not path.is_file():
                continue
            try:
                run_number = parse_run_id(path.name)
            except ValueError:
                continue
            files_by_run.setdefault(run_number, []).append(path)

        observations = []
        missing = []
        for run in selection:
            matches = files_by_run.get(run.run_number, [])
            if not matches:
                missing.append(f"run {run.run_number} ({run.date}) in {directory}")
            elif len(matches) > 1:
                raise ValueError(f"dl3: multiple DL2 files for selected run {run.run_number}: {matches}")
            else:
                observations.append(cls(matches[0].absolute()))
        if missing:
            raise FileNotFoundError("dl3: missing DL2 files for selected runs:\n" + "\n".join(missing))
        return tuple(observations)

    def mc_group(self) -> "MCGroup":
        table = LSTDL2EventTable(self.path)
        if table.nsb_level is None or table.dec_line is None:
            raise ValueError(f"Could not derive the MC group (NSB tuning, declination line) from {self.path}")
        return MCGroup(table.nsb_level, table.dec_line)


@dataclass(frozen=True)
class MCGroup:
    nsb_level: float
    dec_line: int


class MCCatalog:
    def __init__(self, directory_template: str):
        self.directory_template = directory_template
        self._campaigns: dict[Path, list[MCDL2Attribute]] = {}

    def nodes_for(self, group: MCGroup, irf_type: Literal["full-enclosure", "point"]) -> tuple[MCNode, ...]:
        campaign = Path(self.directory_template.replace(NSB_LEVEL_VARIABLE, f"{group.nsb_level:.2f}")).expanduser()
        if not campaign.is_absolute():
            campaign = MC_DL2_PATH / campaign
        if campaign not in self._campaigns:
            self._campaigns[campaign] = list(find_mc_dl2_files(campaign))
        mc_type = "Gamma" if irf_type == "point" else "GammaDiffuse"
        nodes = tuple(
            MCNode(attribute.nsb, attribute.dec, attribute.theta, attribute.azimuth, attribute.path, irf_type)
            for attribute in self._campaigns[campaign]
            if attribute.nsb == group.nsb_level and attribute.dec == group.dec_line and attribute.mc_type == mc_type
        )
        if not nodes:
            raise FileNotFoundError(
                f"No {mc_type} MC nodes for NSB {group.nsb_level:g}, declination line {group.dec_line} in {campaign}"
            )
        return nodes


@dataclass(frozen=True)
class IRFSet:
    group: MCGroup
    root: Path
    config: IRFConfig
    nodes: tuple[IRFNode, ...]

    @property
    def directory(self) -> Path:
        return self.root / irf_group_path(
            nsb_level=self.group.nsb_level,
            dec_line=self.group.dec_line,
            irf_type=self.config.irf_type,
            cut_label=self.config.event_filter.cut_label,
        )

    def prepare(self) -> list[Path]:
        return IRFGenerator(self.root, self.config).make_irf_nodes(self.nodes)


@dataclass(frozen=True)
class DL3Reduction:
    name: str
    source: SourceConfig
    irf_config: IRFConfig
    irfs_root: Path
    output_dir: Path
    observations: tuple[DL2Observation, ...]
    interp_method: str
    use_nearest_irf_node: bool
    overwrite: bool

    @classmethod
    def from_options(
        cls,
        reduction: DL3ReductionOptions,
        options: DL3BuildOptions,
        source: SourceConfig,
        observations: tuple[DL2Observation, ...],
        irfs_root: Path,
        output_dir: Path,
    ) -> "DL3Reduction":
        return cls(
            name=reduction.name,
            source=source,
            irf_config=IRFConfig(reduction.event_filter, point_like=reduction.point_like),
            irfs_root=(irfs_root / reduction.name).absolute(),
            output_dir=(output_dir / reduction.name).absolute(),
            observations=observations,
            interp_method=options.interp_method,
            use_nearest_irf_node=options.use_nearest_irf_node,
            overwrite=options.overwrite,
        )

    def required_irf_sets(self, groups: tuple[MCGroup, ...], catalog: MCCatalog) -> tuple[IRFSet, ...]:
        event_filter = self.irf_config.event_filter
        sets = []
        for group in groups:
            nodes = {}
            for mc_node in catalog.nodes_for(group, self.irf_config.irf_type):
                node = IRFNode.from_mc_node(mc_node, gh_type=event_filter.gh_type, gh_value=event_filter.gh_value)
                nodes.setdefault(node.relative_path, node)
            sets.append(IRFSet(group, self.irfs_root, self.irf_config, tuple(nodes.values())))
        return tuple(sets)

    def reduce_observations(self) -> Counter:
        reducer = DL3Reducer(
            irfs_root=self.irfs_root,
            source_name=self.source.name,
            source_ra=self.source.ra,
            source_dec=self.source.dec,
            irf_config=self.irf_config,
            interp_method=self.interp_method,
            use_nearest_irf_node=self.use_nearest_irf_node,
            overwrite=self.overwrite,
        )
        counts: Counter = Counter()
        if not self.observations:
            return counts
        files = [observation.path for observation in self.observations]
        reduced = reducer.reduce(files, self.output_dir)
        counts["dl3:reduced"] += len(reduced)
        counts["dl3:failed"] += len(files) - len(reduced)
        return counts


@dataclass(frozen=True)
class DL3Build:
    observations: tuple[DL2Observation, ...]
    reductions: tuple[DL3Reduction, ...]
    mc_dl2_directory: str

    def run(self, max_workers: int | None = None) -> Counter:
        from .dl3_execution import DL3Executor

        self.validate_inputs()
        if not self.observations:
            return Counter()
        groups = tuple(dict.fromkeys(observation.mc_group() for observation in self.observations))
        catalog = MCCatalog(self.mc_dl2_directory)
        irf_sets = tuple(
            irf_set for reduction in self.reductions for irf_set in reduction.required_irf_sets(groups, catalog)
        )
        executor = DL3Executor(max_workers)
        executor.prepare_irfs(irf_sets)
        counts = executor.reduce_observations(self.reductions)
        counts["dl3:irf_nodes"] = sum(len(irf_set.nodes) for irf_set in irf_sets)
        return counts

    def validate_inputs(self) -> None:
        missing = [str(observation.path) for observation in self.observations if not observation.path.is_file()]
        if missing:
            raise FileNotFoundError("dl3: selected DL2 files no longer exist:\n" + "\n".join(missing))
