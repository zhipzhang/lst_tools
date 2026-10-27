"""
Typed producer configurations, parsed once from the pipeline TOML config.

One ``DLxProducerConfig`` per data level, each holding the ``source``
mode ("link" or "build") and the mode-specific options in their own
dataclass — so a key can only appear where it is meaningful, and
"key under the wrong mode" is structurally impossible rather than
something validation has to catch after the fact.

Options given for the *other* mode are not an error: they are ignored
with a warning, so a config can carry both blocks while flipping
``source`` during testing.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, Protocol, Self, TypeVar

from lst_tools.event_filter import EventFilter

Source = Literal["link", "build"]
IRFType = Literal["full-enclosure", "point"]

LinkOptionsT = TypeVar("LinkOptionsT")
BuildOptionsT = TypeVar("BuildOptionsT")

# Placeholder in the RF model directory template, substituted per run when
# DL2 is built: the NSB tuning comes from the run's DL1 (nights differ, so
# no single real model directory can be named in the config).
NSB_LEVEL_VARIABLE = "$NSB_LEVEL"


class ConfigError(ValueError):
    """Raised when a pipeline config section is malformed."""


def _unknown_keys(section: dict, known: set[str], where: str) -> None:
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"Unknown keys in [{where}]: {sorted(unknown)}")


def _parse_source(section: dict, where: str) -> Source:
    source = section.get("source")
    if source not in ("link", "build"):
        raise ConfigError(f'[{where}] requires source = "build" or "link", got {source!r}')
    return source


def _optional_path(value) -> Path | None:
    return Path(value).expanduser() if value is not None else None


class _ParsesOptions(Protocol):
    """Structural type of the per-mode options classes' parser."""

    @classmethod
    def parse(cls, section: dict) -> Any: ...


@dataclass(frozen=True)
class _LevelProducerConfig(Generic[LinkOptionsT, BuildOptionsT]):
    """Shared parsing for the DL2/DL3 producer configs.

    Subclasses declare their level name and options classes; everything
    else — section presence, ``enabled``, ``source`` dispatch, wrong-mode
    warnings — happens here once.
    """

    _level: ClassVar[str]
    _link_options_cls: ClassVar[type[_ParsesOptions]]
    _build_options_cls: ClassVar[type[_ParsesOptions]]

    enabled: bool = False  # a missing [dlX] section means the level is skipped
    source: Source = "link"
    link: LinkOptionsT | None = None
    build: BuildOptionsT | None = None

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            return cls(enabled=False)
        _unknown_keys(section, {"enabled", "source", "build", "link"}, cls._level)
        if not section.get("enabled", True):
            source = section.get("source", "link")
            if source not in ("link", "build"):
                raise ConfigError(f'[{cls._level}] requires source = "build" or "link", got {source!r}')
            return cls(enabled=False, source=source)
        source = _parse_source(section, cls._level)

        # Only the active mode's options are parsed; the other mode's
        # section, if present, is ignored with a warning (so a config may
        # carry both blocks while flipping ``source`` during testing).
        if source == "link":
            active_options = cls._parse_mode_options(section, "link")
            if "build" in section:
                warnings.warn(
                    f'[{cls._level}.build] options are ignored because source = "link"',
                    stacklevel=2,
                )
            return cls(enabled=True, source=source, link=active_options)

        active_options = cls._parse_mode_options(section, "build")
        if active_options is None:
            raise ConfigError(f'[{cls._level}.build] is required when source = "build"')
        if "link" in section:
            warnings.warn(
                f'[{cls._level}.link] options are ignored because source = "build"',
                stacklevel=2,
            )
        return cls(enabled=True, source=source, build=active_options)

    @classmethod
    def _parse_mode_options(cls, section: dict, mode: Literal["link", "build"]):
        sub_section = section.get(mode)
        if sub_section is None:
            return None
        options_cls = cls._link_options_cls if mode == "link" else cls._build_options_cls
        return options_cls.parse(sub_section)


@dataclass(frozen=True)
class DL1BuildOptions:
    """Reserved until DL1Producer.build is implemented."""


@dataclass(frozen=True)
class DL1LinkOptions:
    output_directory: Path | None = None  # default <output_root>/dl1

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        section = section or {}
        _unknown_keys(section, {"output_directory"}, "dl1.link")
        return cls(output_directory=_optional_path(section.get("output_directory")))


@dataclass(frozen=True)
class DL1ProducerConfig:
    """DL1 is linked-only today; the class exists so the build mode can
    land later without a config redesign."""

    enabled: bool = True
    source: Literal["link"] = "link"
    link: DL1LinkOptions | None = None
    build: DL1BuildOptions | None = None

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        section = section or {}
        _unknown_keys(section, {"enabled", "source", "build", "link"}, "dl1")
        source = section.get("source", "link")
        if source != "link":
            raise ConfigError(f"[dl1] source must be 'link' (building DL1 is not supported yet), got {source!r}")
        return cls(
            enabled=section.get("enabled", True),
            source="link",
            link=DL1LinkOptions.parse(section["link"]) if "link" in section else None,
        )


@dataclass(frozen=True)
class DL2BuildOptions:
    output_directory: Path | None = None  # default <output_root>/dl2
    # RF model directory template, not necessarily a real path: it may carry
    # a $NSB_LEVEL placeholder, substituted per run at build time with the
    # run's NSB level (from its DL1 file) — the NSB tuning is a per-night
    # property, so no single real directory can be named here.
    rf_directory: str | None = None
    overwrite: bool = False

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"output_directory", "rf_directory", "overwrite"}, "dl2.build")
        rf_directory = section.get("rf_directory")
        if rf_directory is None:
            raise ConfigError("[dl2.build] requires rf_directory (the RF model directory template used to build DL2)")
        return cls(
            output_directory=_optional_path(section.get("output_directory")),
            rf_directory=str(rf_directory),
            overwrite=section.get("overwrite", False),
        )


@dataclass(frozen=True)
class DL2LinkOptions:
    """Newest-version selection is the base default; only the output
    directory is configurable."""

    output_directory: Path | None = None  # default <output_root>/dl2

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        section = section or {}
        _unknown_keys(section, {"output_directory"}, "dl2.link")
        return cls(output_directory=_optional_path(section.get("output_directory")))


@dataclass(frozen=True)
class DL2ProducerConfig(_LevelProducerConfig[DL2LinkOptions, DL2BuildOptions]):
    _level: ClassVar[str] = "dl2"
    _link_options_cls: ClassVar[type[_ParsesOptions]] = DL2LinkOptions
    _build_options_cls: ClassVar[type[_ParsesOptions]] = DL2BuildOptions


@dataclass(frozen=True)
class DL3BuildOptions:
    output_directory: Path | None = None  # default <output_root>/dl3
    irf_directory: Path | None = None  # default <output_root>/irfs
    interp_method: str = "linear"
    use_nearest_irf_node: bool = False
    overwrite: bool = False
    event_filter: EventFilter | None = None  # from [dl3.build.irf]
    point_like: bool = False
    mc_dl2_directory: str = "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(
            section,
            {"output_directory", "irf_directory", "interp_method", "use_nearest_irf_node", "overwrite", "irf"},
            "dl3.build",
        )
        if "irf" not in section:
            raise ConfigError(
                '[dl3.build.irf] is required when source = "build":'
                " the IRF generation and the DL3 reduction share its event selection"
            )
        irf_section = section["irf"]
        _unknown_keys(
            irf_section,
            {"intensity_cut", "leakage_cut", "gh_cut", "gh_efficiency", "point_like", "mc_dl2_directory"},
            "dl3.build.irf",
        )
        try:
            event_filter = EventFilter.from_cuts(
                intensity_cut=irf_section["intensity_cut"],
                leakage_cut=irf_section["leakage_cut"],
                gh_cut=irf_section.get("gh_cut"),
                gh_efficiency=irf_section.get("gh_efficiency"),
            )
        except KeyError as missing:
            raise ConfigError(f"[dl3.build.irf] is missing required key {missing}") from missing
        except ValueError as invalid:
            raise ConfigError(f"[dl3.build.irf] is invalid: {invalid}") from invalid
        return cls(
            output_directory=_optional_path(section.get("output_directory")),
            irf_directory=_optional_path(section.get("irf_directory")),
            interp_method=section.get("interp_method", "linear"),
            use_nearest_irf_node=section.get("use_nearest_irf_node", False),
            overwrite=section.get("overwrite", False),
            event_filter=event_filter,
            point_like=irf_section.get("point_like", False),
            mc_dl2_directory=irf_section.get(
                "mc_dl2_directory", "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_$NSB_LEVEL"
            ),
        )


@dataclass(frozen=True)
class DL3LinkOptions:
    output_directory: Path | None = None  # default <output_root>/dl3
    irf_type: IRFType | None = None  # None links both irf types

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        section = section or {}
        _unknown_keys(section, {"output_directory", "irf_type"}, "dl3.link")
        irf_type = section.get("irf_type")
        if irf_type is not None and irf_type not in ("full-enclosure", "point"):
            raise ConfigError(f"[dl3.link] irf_type must be 'full-enclosure' or 'point', got {irf_type!r}")
        return cls(
            output_directory=_optional_path(section.get("output_directory")),
            irf_type=irf_type,
        )


@dataclass(frozen=True)
class DL3ProducerConfig(_LevelProducerConfig[DL3LinkOptions, DL3BuildOptions]):
    _level: ClassVar[str] = "dl3"
    _link_options_cls: ClassVar[type[_ParsesOptions]] = DL3LinkOptions
    _build_options_cls: ClassVar[type[_ParsesOptions]] = DL3BuildOptions


@dataclass(frozen=True)
class SourceConfig:
    """The analysis source: needed by the DL3 build (reduction) and run selection."""

    name: str
    ra: float
    dec: float

    @classmethod
    def parse(cls, section: dict | None) -> Self:
        if section is None:
            raise ConfigError("[source] is required (name, ra, dec)")
        try:
            return cls(name=section["name"], ra=float(section["ra"]), dec=float(section["dec"]))
        except KeyError as missing:
            raise ConfigError(f"[source] is missing required key {missing}") from missing


@dataclass(frozen=True)
class PipelineConfig:
    """The three producer configs, parsed from the whole pipeline config dict."""

    dl1: DL1ProducerConfig
    dl2: DL2ProducerConfig
    dl3: DL3ProducerConfig

    @classmethod
    def parse(cls, raw_config: dict) -> "PipelineConfig":
        """Parse the ``[dl1]``/``[dl2]``/``[dl3]`` sections of the raw TOML config.

        Raises ConfigError for malformed sections and for the removed
        top-level ``[irf]`` section (it moved under ``[dl3.build.irf]``).
        """
        if "irf" in raw_config:
            raise ConfigError(
                "The top-level [irf] section moved under [dl3.build.irf];"
                " it configures our IRF production and is meaningless when linking"
            )
        return cls(
            dl1=DL1ProducerConfig.parse(raw_config.get("dl1")),
            dl2=DL2ProducerConfig.parse(raw_config.get("dl2")),
            dl3=DL3ProducerConfig.parse(raw_config.get("dl3")),
        )
