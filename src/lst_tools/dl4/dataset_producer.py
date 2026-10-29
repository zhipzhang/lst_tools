"""Generate one Gammapy DL4 dataset per DL3 observation."""

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import Observation
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.makers import (
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
    WobbleRegionsFinder,
)
from gammapy.maps import RegionGeom, WcsGeom
from regions import PointSkyRegion

from lst_tools.dl3 import Dl3DataStore

from .config import DL4DatasetConfig

ObservationStatus = Literal["built", "existing", "failed"]


@dataclass(frozen=True)
class DL4ObservationResult:
    obs_id: int
    status: ObservationStatus
    output: Path | None = None
    error: str | None = None


@dataclass(frozen=True)
class DL4ProductionSummary:
    results: tuple[DL4ObservationResult, ...]
    counts: Counter


class DL4DatasetProducer:
    """Build independent spectrum or map datasets from a DL3 data store."""

    def __init__(self, config: DL4DatasetConfig, data_store=None) -> None:
        self.config = config
        self.data_store = data_store if data_store is not None else Dl3DataStore(config.data.directory)
        self.energy_axis = config.dataset.energy.reconstructed.to_axis("energy")
        self.energy_axis_true = config.dataset.energy.true.to_axis("energy_true")
        self.source_position = SkyCoord(config.source.ra_deg, config.source.dec_deg, unit="deg", frame="icrs")

    def produce(self) -> DL4ProductionSummary:
        data = self.config.data
        observations = self.data_store.get_observations(
            obs_id=list(data.obs_ids) if data.obs_ids else None,
            skip_missing=data.skip_missing,
            required_irf=list(data.required_irf),
            require_events=True,
        )
        if not observations:
            raise RuntimeError("No DL3 observations were selected for DL4 generation")

        self.config.output.directory.mkdir(parents=True, exist_ok=True)
        results = []
        counts: Counter = Counter()
        for observation in observations:
            obs_id = int(observation.obs_id)
            output = None
            try:
                output = self.output_file(observation)
                if output.exists() and not self.config.output.overwrite:
                    status: ObservationStatus = "existing"
                else:
                    self.build_observation(observation, output)
                    status = "built"
                results.append(DL4ObservationResult(obs_id=obs_id, status=status, output=output))
            except Exception as error:  # noqa: BLE001 - preserve successful observations in a batch
                warnings.warn(f"DL4 generation failed for observation {obs_id}: {error}", stacklevel=2)
                status = "failed"
                results.append(DL4ObservationResult(obs_id=obs_id, status=status, output=output, error=str(error)))
            counts[f"dl4:{status}"] += 1
        return DL4ProductionSummary(results=tuple(results), counts=counts)

    def output_file(self, observation: Observation) -> Path:
        location = self.data_store.hdu_table.hdu_location(obs_id=int(observation.obs_id), hdu_type="events")
        if location is None or location.file_name is None:
            raise ValueError(f"No EVENTS filename found for observation {observation.obs_id}")
        filename = Path(location.file_name).name
        if filename.endswith(".fits.gz"):
            stem = filename.removesuffix(".fits.gz")
        elif filename.endswith(".fits"):
            stem = filename.removesuffix(".fits")
        else:
            raise ValueError(f"DL3 EVENTS filename must end in .fits or .fits.gz: {filename}")
        if not stem.startswith("dl3_"):
            raise ValueError(f"Expected a DL3 filename starting with 'dl3_': {filename}")
        stem = f"dl4_{stem.removeprefix('dl3_')}_{self.config.dataset.type}"
        return self.config.output.directory / f"{stem}.fits"

    def build_observation(self, observation: Observation, output: Path | None = None) -> Path:
        output = output if output is not None else self.output_file(observation)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.config.dataset.type == "spectrum":
            dataset = self._build_spectrum(observation)
            dataset.write(output, format="gadf", overwrite=self.config.output.overwrite)
        else:
            dataset = self._build_map(observation)
            dataset.write(output, overwrite=self.config.output.overwrite)
        return output

    def _build_spectrum(self, observation: Observation):
        options = self.config.dataset.spectrum
        if options is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("Spectrum options are required")
        geom = RegionGeom.create(
            region=PointSkyRegion(center=self.source_position),
            axes=[self.energy_axis],
        )
        empty = SpectrumDataset.create(
            geom=geom,
            energy_axis_true=self.energy_axis_true,
            name=str(observation.obs_id),
        )
        maker = SpectrumDatasetMaker(
            selection=list(options.maker.selection),
            containment_correction=options.maker.containment_correction,
            use_region_center=options.maker.use_region_center,
        )
        dataset = maker.run(empty, observation)
        background_maker = ReflectedRegionsBackgroundMaker(
            region_finder=WobbleRegionsFinder(
                n_off_regions=options.background.n_off_regions,
                binsz=options.background.binsz_deg * u.deg,
            ),
            exclusion_mask=self._exclusion_mask(),
        )
        dataset = background_maker.run(dataset, observation)
        return self._apply_safe_mask(dataset, observation)

    def _build_map(self, observation: Observation):
        options = self.config.dataset.map
        if options is None:  # pragma: no cover - guarded by config parsing
            raise ValueError("Map options are required")
        geometry = options.geometry
        geom = WcsGeom.create(
            skydir=(geometry.center_ra_deg, geometry.center_dec_deg),
            width=geometry.width_deg,
            binsz=geometry.pixel_size_deg,
            frame=geometry.frame,
            proj=geometry.projection,
            axes=[self.energy_axis],
        )
        empty = MapDataset.create(
            geom=geom,
            energy_axis_true=self.energy_axis_true,
            binsz_irf=options.irf.pixel_size_deg * u.deg,
            name=str(observation.obs_id),
        )
        maker = MapDatasetMaker(selection=list(options.maker.selection))
        dataset = maker.run(empty, observation)
        return self._apply_safe_mask(dataset, observation)

    def _apply_safe_mask(self, dataset, observation: Observation):
        options = self.config.dataset.safe_mask
        if not options.enabled:
            return dataset
        position = None
        if options.position_ra_deg is not None:
            position = SkyCoord(options.position_ra_deg, options.position_dec_deg, unit="deg", frame="icrs")
        fixed_offset = options.fixed_offset_deg * u.deg if options.fixed_offset_deg is not None else None
        maker = SafeMaskMaker(
            methods=list(options.methods),
            aeff_percent=options.aeff_percent,
            bias_percent=options.bias_percent,
            position=position,
            fixed_offset=fixed_offset,
            offset_max=options.offset_max_deg * u.deg,
            irfs=options.irfs,
        )
        return maker.run(dataset, observation)

    def _exclusion_mask(self):
        options = self.config.dataset.spectrum
        if options is None or not options.background.exclusion.regions:
            return None
        exclusion = options.background.exclusion
        geom = WcsGeom.create(
            skydir=self.source_position,
            width=exclusion.mask_width_deg * u.deg,
            binsz=exclusion.mask_binsz_deg * u.deg,
            frame="icrs",
        )
        return ~geom.region_mask(exclusion.sky_regions())
