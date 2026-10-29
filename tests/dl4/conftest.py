import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from gammapy.data import (
    GTI,
    EventList,
    FixedPointingInfo,
    HDUIndexTable,
    Observation,
    ObservationTable,
)
from gammapy.irf import PSF3D, EffectiveAreaTable2D, EnergyDispersion2D, RadMax2D
from gammapy.maps import MapAxis

SOURCE = SkyCoord(83.632, 22.0145, unit="deg", frame="icrs")
REFERENCE_TIME = Time("2024-01-01T00:00:00", scale="tt")


def make_observation(obs_id=2033, *, point_like=True):
    energy_true = MapAxis.from_energy_bounds(0.02, 50, nbin=8, unit="TeV", name="energy_true")
    energy = MapAxis.from_energy_bounds(0.03, 30, nbin=5, unit="TeV", name="energy")
    offset = MapAxis.from_edges([0, 1, 2.5], name="offset", unit="deg")
    migra = MapAxis.from_edges(np.linspace(0, 3, 61), name="migra")
    rad = MapAxis.from_edges(np.linspace(0, 1, 101), name="rad", unit="deg")

    aeff = EffectiveAreaTable2D(
        axes=[energy_true, offset],
        data=np.full((energy_true.nbin, offset.nbin), 1e6),
        unit="m2",
        is_pointlike=point_like,
    )
    edisp = EnergyDispersion2D.from_gauss(energy_true, migra, offset, bias=0, sigma=0.1)
    edisp = EnergyDispersion2D(
        axes=edisp.axes,
        data=edisp.data,
        is_pointlike=point_like,
    )
    psf_values = np.exp(-0.5 * (rad.center.to_value("deg") / 0.1) ** 2)
    psf = PSF3D(
        axes=[energy_true, offset, rad],
        data=np.broadcast_to(psf_values, (energy_true.nbin, offset.nbin, rad.nbin)),
        unit="deg-2",
    )
    psf.normalize()
    rad_max = None
    if point_like:
        rad_max = RadMax2D(
            axes=[energy, offset],
            data=np.full((energy.nbin, offset.nbin), 0.15),
            unit="deg",
            is_pointlike=True,
        )

    ras = np.concatenate([np.full(10, SOURCE.ra.deg), np.linspace(82.8, 84.5, 40)])
    decs = np.full(50, SOURCE.dec.deg)
    events_table = Table(
        {
            "RA": ras * u.deg,
            "DEC": decs * u.deg,
            "ENERGY": np.full(50, 1.0) * u.TeV,
            "TIME": REFERENCE_TIME + np.arange(50) * u.s,
        }
    )
    events_table.meta.update(
        OBS_ID=obs_id,
        RA_PNT=84.132,
        DEC_PNT=22.0145,
        TSTART=0.0,
        TSTOP=1000.0,
        ONTIME=1000.0,
        LIVETIME=900.0,
        DEADC=0.9,
        TELESCOP="LST-1",
        INSTRUME="LST",
        TIMESYS="tt",
        TIMEUNIT="s",
        MJDREFI=int(REFERENCE_TIME.mjd),
        MJDREFF=REFERENCE_TIME.mjd % 1,
        GEOLON=-17.89,
        GEOLAT=28.76,
        GEOALT=2184.0,
    )
    gti = GTI(
        Table(
            {
                "START": REFERENCE_TIME + [0.0] * u.s,
                "STOP": REFERENCE_TIME + [1000.0] * u.s,
            }
        ),
        reference_time=REFERENCE_TIME,
    )
    return Observation(
        obs_id=obs_id,
        events=EventList(events_table),
        gti=gti,
        aeff=aeff,
        edisp=edisp,
        psf=psf,
        rad_max=rad_max,
        pointing=FixedPointingInfo(fixed_icrs=SkyCoord(84.132, 22.0145, unit="deg", frame="icrs")),
    )


def write_dl3_store(directory, observations):
    directory.mkdir(parents=True, exist_ok=True)
    hdu_rows = []
    for observation in observations:
        filename = f"dl3_LST-1.Run{observation.obs_id:05d}.fits"
        path = directory / filename
        hdus = [
            fits.PrimaryHDU(),
            fits.BinTableHDU(observation.events.table, name="EVENTS"),
            observation.gti.to_table_hdu(),
        ]
        components = [
            ("aeff", "aeff_2d", "EFFECTIVE AREA", observation.aeff),
            ("edisp", "edisp_2d", "EDISP", observation.edisp),
            ("psf", "psf_table", "PSF", observation.psf),
        ]
        if observation.rad_max is not None:
            components.append(("rad_max", "rad_max_2d", "RAD_MAX", observation.rad_max))
        for _, _, hdu_name, component in components:
            hdu = component.to_table_hdu()
            hdu.name = hdu_name
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path)

        hdu_rows.append((observation.obs_id, "events", "events", ".", filename, "EVENTS"))
        hdu_rows.append((observation.obs_id, "gti", "gti", ".", filename, "GTI"))
        for hdu_type, hdu_class, hdu_name, _ in components:
            hdu_rows.append((observation.obs_id, hdu_type, hdu_class, ".", filename, hdu_name))

    HDUIndexTable(
        rows=hdu_rows,
        names=["OBS_ID", "HDU_TYPE", "HDU_CLASS", "FILE_DIR", "FILE_NAME", "HDU_NAME"],
    ).write(directory / "hdu-index.fits.gz", format="fits")
    ObservationTable({"OBS_ID": [observation.obs_id for observation in observations]}).write(
        directory / "obs-index.fits.gz",
        format="fits",
    )
    return directory


@pytest.fixture
def observation_factory():
    return make_observation


@pytest.fixture
def dl3_store_factory():
    return write_dl3_store


@pytest.fixture
def raw_config_factory(tmp_path):
    def factory(dataset_type="spectrum", *, dl3_directory=None):
        required_irf = ["aeff", "edisp", "rad_max"] if dataset_type == "spectrum" else ["aeff", "edisp", "psf"]
        dataset = {
            "type": dataset_type,
            "energy": {
                "min": 0.03,
                "max": 30.0,
                "nbin": 5,
                "unit": "TeV",
                "true": {"min": 0.02, "max": 50.0, "nbin": 8, "unit": "TeV"},
            },
            "safe_mask": {"enabled": False},
        }
        if dataset_type == "spectrum":
            dataset["spectrum"] = {
                "maker": {"selection": ["counts", "exposure", "edisp"]},
                "background": {"n_off_regions": 3, "binsz_deg": 0.01},
            }
        else:
            dataset["map"] = {
                "maker": {"selection": ["counts", "exposure", "edisp", "psf"]},
                "geometry": {
                    "center_ra_deg": SOURCE.ra.deg,
                    "center_dec_deg": SOURCE.dec.deg,
                    "width_deg": [2.0, 2.0],
                    "pixel_size_deg": 0.1,
                },
                "irf": {"pixel_size_deg": 0.2},
            }
        return {
            "data": {
                "directory": str(dl3_directory or tmp_path / "dl3"),
                "obs_ids": [],
                "required_irf": required_irf,
                "skip_missing": False,
            },
            "source": {"name": "Crab Nebula", "ra_deg": SOURCE.ra.deg, "dec_deg": SOURCE.dec.deg},
            "output": {"directory": str(tmp_path / "dl4"), "overwrite": False},
            "dataset": dataset,
        }

    return factory
