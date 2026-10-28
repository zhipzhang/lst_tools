import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from gammapy.irf import Background3D
from gammapy.maps import RegionGeom
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame
from regions import CircleSkyRegion

from lst_tools.bkg import BkgMaker, CameraImage, ObservationFrame, SkyOffsetEvents


def make_image(frame, *, livetime=10 * u.s, energy=(0.5, 5), x=(-0.5, 0.5), y=(-0.5, 0.5)):
    events = SkyOffsetEvents(x=x, y=y, energy=energy, livetime=livetime, frame=frame)
    return CameraImage.from_events(events, [-1, 0, 1], [-1, 0, 1], [0.1, 1, 10])


def exclusion_region(frame):
    center = SkyCoord(0.5, 0.5, unit=u.deg, frame=frame.fov_frame).icrs
    return RegionGeom.from_regions(CircleSkyRegion(center=center, radius=0.4 * u.deg))


def test_accumulates_counts_and_exposure_from_multiple_images(observation_frame):
    maker = BkgMaker()
    images = [make_image(observation_frame), make_image(observation_frame, livetime=20 * u.s)]

    counts, exposure_map = maker._accumulate(images, None)

    assert counts.sum() == 4
    assert exposure_map.shape == (8, 8)
    np.testing.assert_allclose(exposure_map, 30)


def test_exclusion_region_masks_count_and_exposure_pixels(observation_frame):
    maker = BkgMaker(exposure_oversampling=2)
    image = make_image(observation_frame)

    counts, exposure_map = maker._accumulate([image], exclusion_region(observation_frame))

    assert counts[1, 1, :].sum() == 0
    assert counts.sum() == 1
    assert np.count_nonzero(exposure_map == 0) > 0
    assert np.count_nonzero(exposure_map == 10) > 0


def test_convolve_uses_the_same_kernel_for_counts_and_exposure():
    counts = np.zeros((3, 3, 1))
    counts[1, 1, 0] = 2
    exposure = counts * 10

    convolved_counts, convolved_exposure = BkgMaker().convolve(counts, exposure, (1, 1), 1 * u.deg)

    np.testing.assert_allclose(convolved_exposure, convolved_counts * 10)
    assert convolved_counts[1, 1, 0] == 2
    assert convolved_counts[0, 1, 0] == 2
    assert convolved_counts[0, 0, 0] == 0


def test_make_icrs_background_map_returns_normalized_gammapy_background(observation_frame):
    image = make_image(observation_frame)
    maker = BkgMaker()

    background = maker.make_icrs_background_map([image])

    assert isinstance(background, Background3D)
    assert background.axes.names == ["energy", "fov_lon", "fov_lat"]
    expected = 1 / (10 * 900_000 * image.pixel_solid_angle[0, 0])
    assert background.data[0, 0, 0] == pytest.approx(expected)
    assert background.unit == u.Unit("s-1 MeV-1 sr-1")


def test_make_icrs_background_map_writes_fits(observation_frame, tmp_path):
    filename = tmp_path / "background.fits"
    background = BkgMaker().make_icrs_background_map([make_image(observation_frame)], filename=filename)

    restored = Background3D.read(filename)
    np.testing.assert_allclose(restored.data, background.data)


def test_zero_exposure_cannot_be_exported(observation_frame):
    image = CameraImage([-1, 0, 1], [-1, 0, 1], [0.1, 1, 10], frame=observation_frame)

    with pytest.raises(ValueError, match="zero exposure"):
        BkgMaker().make_icrs_background_map([image])


def test_make_icrs_background_map_requires_icrs_frames():
    altaz = AltAz(obstime=Time(1_700_000_000, format="unix"))
    origin = SkyCoord(alt=70 * u.deg, az=10 * u.deg, frame=altaz)
    frame = ObservationFrame(FoVAltAzFrame(origin=origin), Time(1_700_000_000, format="unix"), 70 * u.deg, 10 * u.deg)

    with pytest.raises(ValueError, match="FoVICRSFrame"):
        BkgMaker().make_icrs_background_map([make_image(frame)])


def test_make_icrs_background_map_requires_matching_axes(observation_frame):
    first = make_image(observation_frame)
    second = CameraImage([-2, 0, 2], [-1, 0, 1], [0.1, 1, 10], frame=observation_frame)

    with pytest.raises(ValueError, match="identical axes"):
        BkgMaker().make_icrs_background_map([first, second])


def test_make_icrs_background_map_requires_nonempty_images():
    with pytest.raises(ValueError, match="At least one"):
        BkgMaker().make_icrs_background_map([])
