"""Typed configuration for standalone DL3-to-DL4 dataset generation."""

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import astropy.units as u
from gammapy.maps import MapAxis
from regions import Regions, SkyRegion

DatasetType = Literal["spectrum", "map"]
IRFName = Literal["aeff", "edisp", "psf", "rad_max"]

_ALLOWED_IRFS = {"aeff", "edisp", "psf", "rad_max", "bkg"}
_SAFE_MASK_METHODS = {"aeff-default", "aeff-max", "edisp-bias", "offset-max", "bkg-peak"}


class DL4ConfigError(ValueError):
    """Raised when a DL4 dataset configuration is malformed."""


def _unknown_keys(section: dict, known: set[str], where: str) -> None:
    unknown = set(section) - known
    if unknown:
        raise DL4ConfigError(f"Unknown keys in [{where}]: {sorted(unknown)}")


def _required(section: dict, key: str, where: str):
    if key not in section:
        raise DL4ConfigError(f"[{where}] requires {key!r}")
    return section[key]


def _table(section: dict | None, key: str, where: str, *, required: bool = False) -> dict:
    if section is None or key not in section:
        if required:
            raise DL4ConfigError(f"[{where}.{key}] is required")
        return {}
    value = section[key]
    if not isinstance(value, dict):
        raise DL4ConfigError(f"[{where}.{key}] must be a TOML table")
    return value


def _bool(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise DL4ConfigError(f"{where} must be a boolean")
    return value


def _float(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DL4ConfigError(f"{where} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise DL4ConfigError(f"{where} must be finite")
    return value


def _positive_float(value, where: str) -> float:
    value = _float(value, where)
    if value <= 0:
        raise DL4ConfigError(f"{where} must be greater than zero")
    return value


def _int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DL4ConfigError(f"{where} must be an integer")
    return value


def _strings(value, where: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DL4ConfigError(f"{where} must be an array of non-empty strings")
    if not allow_empty and not value:
        raise DL4ConfigError(f"{where} must not be empty")
    return tuple(value)


def _path(value, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DL4ConfigError(f"{where} must be a non-empty path")
    return Path(value).expanduser()


def _coordinates(ra, dec, where: str) -> tuple[float, float]:
    ra = _float(ra, f"{where} RA")
    dec = _float(dec, f"{where} Dec")
    if not 0 <= ra < 360:
        raise DL4ConfigError(f"{where} RA must be in [0, 360) degrees")
    if not -90 <= dec <= 90:
        raise DL4ConfigError(f"{where} Dec must be in [-90, 90] degrees")
    return ra, dec


@dataclass(frozen=True)
class DL4DataConfig:
    directory: Path
    obs_ids: tuple[int, ...]
    required_irf: tuple[str, ...]
    skip_missing: bool = False

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"directory", "obs_ids", "required_irf", "skip_missing"}, "data")
        obs_ids = section.get("obs_ids", [])
        if not isinstance(obs_ids, list):
            raise DL4ConfigError("data.obs_ids must be an array of observation IDs")
        parsed_ids = tuple(_int(value, "data.obs_ids entries") for value in obs_ids)
        if any(value < 0 for value in parsed_ids) or len(set(parsed_ids)) != len(parsed_ids):
            raise DL4ConfigError("data.obs_ids must contain unique non-negative observation IDs")
        required_irf = _strings(
            _required(section, "required_irf", "data"),
            "data.required_irf",
            allow_empty=False,
        )
        unknown_irfs = set(required_irf) - _ALLOWED_IRFS
        if unknown_irfs:
            raise DL4ConfigError(f"Unknown required_irf entries: {sorted(unknown_irfs)}")
        if len(set(required_irf)) != len(required_irf):
            raise DL4ConfigError("data.required_irf entries must be unique")
        if "bkg" in required_irf:
            raise DL4ConfigError("DL4 generation does not use a DL3 background IRF; remove 'bkg'")
        return cls(
            directory=_path(_required(section, "directory", "data"), "data.directory"),
            obs_ids=parsed_ids,
            required_irf=required_irf,
            skip_missing=_bool(section.get("skip_missing", False), "data.skip_missing"),
        )


@dataclass(frozen=True)
class DL4SourceConfig:
    name: str
    ra_deg: float
    dec_deg: float

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"name", "ra_deg", "dec_deg"}, "source")
        name = _required(section, "name", "source")
        if not isinstance(name, str) or not name:
            raise DL4ConfigError("source.name must be a non-empty string")
        ra_deg, dec_deg = _coordinates(
            _required(section, "ra_deg", "source"),
            _required(section, "dec_deg", "source"),
            "source",
        )
        return cls(name=name, ra_deg=ra_deg, dec_deg=dec_deg)


@dataclass(frozen=True)
class DL4OutputConfig:
    directory: Path
    overwrite: bool = False

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"directory", "overwrite"}, "output")
        return cls(
            directory=_path(_required(section, "directory", "output"), "output.directory"),
            overwrite=_bool(section.get("overwrite", False), "output.overwrite"),
        )


@dataclass(frozen=True)
class EnergyAxisConfig:
    min: float
    max: float
    nbin: int
    unit: str = "TeV"
    per_decade: bool = False
    strict_bounds: bool = True

    @classmethod
    def parse(cls, section: dict, where: str) -> Self:
        _unknown_keys(section, {"min", "max", "nbin", "unit", "per_decade", "strict_bounds"}, where)
        minimum = _positive_float(_required(section, "min", where), f"{where}.min")
        maximum = _positive_float(_required(section, "max", where), f"{where}.max")
        if maximum <= minimum:
            raise DL4ConfigError(f"{where}.max must be greater than {where}.min")
        nbin = _int(_required(section, "nbin", where), f"{where}.nbin")
        if nbin < 1:
            raise DL4ConfigError(f"{where}.nbin must be positive")
        unit = section.get("unit", "TeV")
        if not isinstance(unit, str):
            raise DL4ConfigError(f"{where}.unit must be a string")
        try:
            physical_type = u.Unit(unit).physical_type
        except ValueError as error:
            raise DL4ConfigError(f"Invalid energy unit {unit!r} in [{where}]") from error
        if physical_type != "energy":
            raise DL4ConfigError(f"{where}.unit must be an energy unit")
        return cls(
            min=minimum,
            max=maximum,
            nbin=nbin,
            unit=unit,
            per_decade=_bool(section.get("per_decade", False), f"{where}.per_decade"),
            strict_bounds=_bool(section.get("strict_bounds", True), f"{where}.strict_bounds"),
        )

    def to_axis(self, name: Literal["energy", "energy_true"]) -> MapAxis:
        return MapAxis.from_energy_bounds(
            energy_min=self.min,
            energy_max=self.max,
            nbin=self.nbin,
            unit=self.unit,
            per_decade=self.per_decade,
            strict_bounds=self.strict_bounds,
            name=name,
        )


@dataclass(frozen=True)
class EnergyConfig:
    reconstructed: EnergyAxisConfig
    true: EnergyAxisConfig

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"min", "max", "nbin", "unit", "per_decade", "strict_bounds", "true"}, "dataset.energy")
        reconstructed_section = {key: value for key, value in section.items() if key != "true"}
        return cls(
            reconstructed=EnergyAxisConfig.parse(reconstructed_section, "dataset.energy"),
            true=EnergyAxisConfig.parse(
                _table(section, "true", "dataset.energy", required=True), "dataset.energy.true"
            ),
        )


@dataclass(frozen=True)
class SafeMaskConfig:
    enabled: bool = True
    methods: tuple[str, ...] = ("aeff-default",)
    aeff_percent: float = 10.0
    bias_percent: float = 10.0
    offset_max_deg: float = 3.0
    irfs: Literal["DL4", "DL3"] = "DL4"
    position_ra_deg: float | None = None
    position_dec_deg: float | None = None
    fixed_offset_deg: float | None = None

    @classmethod
    def parse(cls, section: dict) -> Self:
        known = {
            "enabled",
            "methods",
            "aeff_percent",
            "bias_percent",
            "offset_max_deg",
            "irfs",
            "position_ra_deg",
            "position_dec_deg",
            "fixed_offset_deg",
        }
        _unknown_keys(section, known, "dataset.safe_mask")
        methods = _strings(section.get("methods", ["aeff-default"]), "dataset.safe_mask.methods", allow_empty=False)
        unknown_methods = set(methods) - _SAFE_MASK_METHODS
        if unknown_methods:
            raise DL4ConfigError(f"Unknown safe-mask methods: {sorted(unknown_methods)}")
        irfs = section.get("irfs", "DL4")
        if irfs not in ("DL3", "DL4"):
            raise DL4ConfigError("dataset.safe_mask.irfs must be 'DL3' or 'DL4'")
        position_values = (section.get("position_ra_deg"), section.get("position_dec_deg"))
        if (position_values[0] is None) != (position_values[1] is None):
            raise DL4ConfigError("Safe-mask position requires both RA and Dec")
        position = None
        if position_values[0] is not None:
            position = _coordinates(*position_values, "dataset.safe_mask.position")
        fixed_offset = section.get("fixed_offset_deg")
        if position is not None and fixed_offset is not None:
            raise DL4ConfigError("Safe-mask position and fixed_offset are mutually exclusive")
        return cls(
            enabled=_bool(section.get("enabled", True), "dataset.safe_mask.enabled"),
            methods=methods,
            aeff_percent=_positive_float(section.get("aeff_percent", 10.0), "dataset.safe_mask.aeff_percent"),
            bias_percent=_positive_float(section.get("bias_percent", 10.0), "dataset.safe_mask.bias_percent"),
            offset_max_deg=_positive_float(section.get("offset_max_deg", 3.0), "dataset.safe_mask.offset_max_deg"),
            irfs=irfs,
            position_ra_deg=position[0] if position else None,
            position_dec_deg=position[1] if position else None,
            fixed_offset_deg=(
                _positive_float(fixed_offset, "dataset.safe_mask.fixed_offset_deg")
                if fixed_offset is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SpectrumMakerConfig:
    selection: tuple[str, ...] = ("counts", "exposure", "edisp")
    containment_correction: bool = False
    use_region_center: bool = True

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(
            section,
            {"selection", "containment_correction", "use_region_center"},
            "dataset.spectrum.maker",
        )
        selection = _strings(
            section.get("selection", ["counts", "exposure", "edisp"]),
            "dataset.spectrum.maker.selection",
            allow_empty=False,
        )
        invalid = set(selection) - {"counts", "exposure", "edisp"}
        if invalid:
            raise DL4ConfigError(f"Invalid spectrum maker selections: {sorted(invalid)}")
        if "counts" not in selection:
            raise DL4ConfigError("Spectrum maker selection must include 'counts'")
        return cls(
            selection=selection,
            containment_correction=_bool(
                section.get("containment_correction", False),
                "dataset.spectrum.maker.containment_correction",
            ),
            use_region_center=_bool(
                section.get("use_region_center", True),
                "dataset.spectrum.maker.use_region_center",
            ),
        )


@dataclass(frozen=True)
class ExclusionConfig:
    regions: tuple[str, ...] = ()
    mask_width_deg: float = 10.0
    mask_binsz_deg: float = 0.05

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"regions", "mask_width_deg", "mask_binsz_deg"}, "dataset.spectrum.background.exclusion")
        regions = _strings(section.get("regions", []), "dataset.spectrum.background.exclusion.regions")
        for region_string in regions:
            try:
                parsed = Regions.parse(region_string, format="ds9")
            except Exception as error:
                raise DL4ConfigError(f"Invalid DS9 exclusion region {region_string!r}: {error}") from error
            if not parsed or any(not isinstance(region, SkyRegion) for region in parsed):
                raise DL4ConfigError(f"Invalid DS9 exclusion region {region_string!r}: no valid sky regions")
        return cls(
            regions=regions,
            mask_width_deg=_positive_float(
                section.get("mask_width_deg", 10.0),
                "dataset.spectrum.background.exclusion.mask_width_deg",
            ),
            mask_binsz_deg=_positive_float(
                section.get("mask_binsz_deg", 0.05),
                "dataset.spectrum.background.exclusion.mask_binsz_deg",
            ),
        )

    def sky_regions(self) -> list[SkyRegion]:
        return [region for text in self.regions for region in Regions.parse(text, format="ds9")]


@dataclass(frozen=True)
class SpectrumBackgroundConfig:
    n_off_regions: int = 3
    binsz_deg: float = 0.01
    exclusion: ExclusionConfig = ExclusionConfig()

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"n_off_regions", "binsz_deg", "exclusion"}, "dataset.spectrum.background")
        n_off_regions = _int(section.get("n_off_regions", 3), "dataset.spectrum.background.n_off_regions")
        if n_off_regions < 1:
            raise DL4ConfigError("dataset.spectrum.background.n_off_regions must be positive")
        return cls(
            n_off_regions=n_off_regions,
            binsz_deg=_positive_float(section.get("binsz_deg", 0.01), "dataset.spectrum.background.binsz_deg"),
            exclusion=ExclusionConfig.parse(_table(section, "exclusion", "dataset.spectrum.background")),
        )


@dataclass(frozen=True)
class SpectrumConfig:
    maker: SpectrumMakerConfig = SpectrumMakerConfig()
    background: SpectrumBackgroundConfig = SpectrumBackgroundConfig()

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"maker", "background"}, "dataset.spectrum")
        return cls(
            maker=SpectrumMakerConfig.parse(_table(section, "maker", "dataset.spectrum")),
            background=SpectrumBackgroundConfig.parse(_table(section, "background", "dataset.spectrum")),
        )


@dataclass(frozen=True)
class MapMakerConfig:
    selection: tuple[str, ...] = ("counts", "exposure", "edisp", "psf")

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"selection"}, "dataset.map.maker")
        selection = _strings(
            section.get("selection", ["counts", "exposure", "edisp", "psf"]),
            "dataset.map.maker.selection",
            allow_empty=False,
        )
        invalid = set(selection) - {"counts", "exposure", "edisp", "psf"}
        if invalid:
            raise DL4ConfigError(f"Invalid map maker selections: {sorted(invalid)}")
        if "counts" not in selection:
            raise DL4ConfigError("Map maker selection must include 'counts'")
        return cls(selection=selection)


@dataclass(frozen=True)
class MapGeometryConfig:
    center_ra_deg: float
    center_dec_deg: float
    width_deg: tuple[float, float]
    pixel_size_deg: float
    frame: str = "icrs"
    projection: str = "CAR"

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(
            section,
            {"center_ra_deg", "center_dec_deg", "width_deg", "pixel_size_deg", "frame", "projection"},
            "dataset.map.geometry",
        )
        center_ra, center_dec = _coordinates(
            _required(section, "center_ra_deg", "dataset.map.geometry"),
            _required(section, "center_dec_deg", "dataset.map.geometry"),
            "dataset.map.geometry",
        )
        width = section.get("width_deg")
        if not isinstance(width, list) or len(width) != 2:
            raise DL4ConfigError("dataset.map.geometry.width_deg must contain two values")
        width = (
            _positive_float(width[0], "dataset.map.geometry.width_deg[0]"),
            _positive_float(width[1], "dataset.map.geometry.width_deg[1]"),
        )
        frame = section.get("frame", "icrs")
        if frame not in ("icrs", "galactic"):
            raise DL4ConfigError("dataset.map.geometry.frame must be 'icrs' or 'galactic'")
        projection = section.get("projection", "CAR")
        if not isinstance(projection, str) or not projection:
            raise DL4ConfigError("dataset.map.geometry.projection must be a non-empty string")
        return cls(
            center_ra_deg=center_ra,
            center_dec_deg=center_dec,
            width_deg=width,
            pixel_size_deg=_positive_float(
                _required(section, "pixel_size_deg", "dataset.map.geometry"),
                "dataset.map.geometry.pixel_size_deg",
            ),
            frame=frame,
            projection=projection,
        )


@dataclass(frozen=True)
class MapIRFConfig:
    pixel_size_deg: float = 0.2

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"pixel_size_deg"}, "dataset.map.irf")
        return cls(
            pixel_size_deg=_positive_float(
                section.get("pixel_size_deg", 0.2),
                "dataset.map.irf.pixel_size_deg",
            )
        )


@dataclass(frozen=True)
class MapConfig:
    maker: MapMakerConfig
    geometry: MapGeometryConfig
    irf: MapIRFConfig = MapIRFConfig()

    @classmethod
    def parse(cls, section: dict) -> Self:
        _unknown_keys(section, {"maker", "geometry", "irf"}, "dataset.map")
        return cls(
            maker=MapMakerConfig.parse(_table(section, "maker", "dataset.map")),
            geometry=MapGeometryConfig.parse(_table(section, "geometry", "dataset.map", required=True)),
            irf=MapIRFConfig.parse(_table(section, "irf", "dataset.map")),
        )


@dataclass(frozen=True)
class DL4DatasetProducerConfig:
    type: DatasetType
    energy: EnergyConfig
    safe_mask: SafeMaskConfig
    spectrum: SpectrumConfig | None
    map: MapConfig | None

    @classmethod
    def parse(cls, section: dict, required_irf: tuple[str, ...]) -> Self:
        _unknown_keys(section, {"type", "energy", "safe_mask", "spectrum", "map"}, "dataset")
        dataset_type = _required(section, "type", "dataset")
        if dataset_type not in ("spectrum", "map"):
            raise DL4ConfigError("dataset.type must be 'spectrum' or 'map'")
        energy = EnergyConfig.parse(_table(section, "energy", "dataset", required=True))
        safe_mask = SafeMaskConfig.parse(_table(section, "safe_mask", "dataset"))
        inactive = "map" if dataset_type == "spectrum" else "spectrum"
        if inactive in section:
            warnings.warn(
                f"[dataset.{inactive}] options are ignored because dataset.type = {dataset_type!r}", stacklevel=2
            )

        spectrum = None
        map_config = None
        if dataset_type == "spectrum":
            required = {"aeff", "edisp", "rad_max"}
            missing = required - set(required_irf)
            if missing:
                raise DL4ConfigError(
                    f"Spectrum generation requires IRFs: {sorted(required)}; missing {sorted(missing)}"
                )
            spectrum = SpectrumConfig.parse(_table(section, "spectrum", "dataset"))
            if spectrum.maker.containment_correction and "psf" not in required_irf:
                raise DL4ConfigError("Spectrum containment correction requires 'psf' in data.required_irf")
        else:
            required = {"aeff", "edisp", "psf"}
            missing = required - set(required_irf)
            if missing:
                raise DL4ConfigError(f"Map generation requires IRFs: {sorted(required)}; missing {sorted(missing)}")
            if safe_mask.enabled and "bkg-peak" in safe_mask.methods:
                raise DL4ConfigError("The bkg-peak safe mask is unavailable without a background model")
            map_config = MapConfig.parse(_table(section, "map", "dataset", required=True))

        return cls(
            type=dataset_type,
            energy=energy,
            safe_mask=safe_mask,
            spectrum=spectrum,
            map=map_config,
        )


@dataclass(frozen=True)
class DL4DatasetConfig:
    data: DL4DataConfig
    source: DL4SourceConfig
    output: DL4OutputConfig
    dataset: DL4DatasetProducerConfig

    @classmethod
    def parse(cls, raw_config: dict) -> Self:
        _unknown_keys(raw_config, {"data", "source", "output", "dataset"}, "root")
        data = DL4DataConfig.parse(_table(raw_config, "data", "root", required=True))
        return cls(
            data=data,
            source=DL4SourceConfig.parse(_table(raw_config, "source", "root", required=True)),
            output=DL4OutputConfig.parse(_table(raw_config, "output", "root", required=True)),
            dataset=DL4DatasetProducerConfig.parse(
                _table(raw_config, "dataset", "root", required=True),
                data.required_irf,
            ),
        )
