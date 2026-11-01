import astropy.units as u
import hist
import numpy as np
import pytest
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import RegionGeom
from gammapy.utils.coordinates import FoVAltAzFrame
from regions import CircleSkyRegion

from lst_tools.bkg import BkgMaker, CameraImage, ObservationFrame, SkyOffsetEvents
from lst_tools.location import LST_LOCATION


def make_image(
    frame,
    *,
    livetime=10 * u.s,
    energy=(0.5, 5),
    x=(-0.5, 0.5),
    y=(-0.5, 0.5),
    x_edges=(-1, 0, 1),
    y_edges=(-1, 0, 1),
):
    events = SkyOffsetEvents(x=x, y=y, energy=energy, livetime=livetime, frame=frame)
    return CameraImage.from_events(events, x_edges, y_edges, [0.1, 1, 10])


def exclusion_region(frame):
    center = SkyCoord(0.5, 0.5, unit=u.deg, frame=frame.fov_frame).icrs
    return RegionGeom.from_regions(CircleSkyRegion(center=center, radius=0.4 * u.deg))


def make_altaz_frame(*, vector=False):
    offsets = [0, 60] if vector else 0
    times = Time(1_700_000_000 + np.asarray(offsets), format="unix")
    pointing = SkyCoord(ra=83.5, dec=20.3, unit=u.deg, frame="icrs")
    origin = pointing.transform_to(AltAz(obstime=times, location=LST_LOCATION))
    mean_time = times.mean()
    mean_pointing = pointing.transform_to(AltAz(obstime=mean_time, location=LST_LOCATION))
    return ObservationFrame(
        fov_frame=FoVAltAzFrame(origin=origin, location=LST_LOCATION, obstime=times),
        mean_time=mean_time,
        pointing_alt=mean_pointing.alt,
        pointing_az=mean_pointing.az,
    )


def test_make_counts_and_exposure_maps_accumulate_multiple_images(observation_frame):
    maker = BkgMaker()
    images = (make_image(observation_frame, livetime=lt) for lt in (10 * u.s, 20 * u.s))

    counts = maker.make_counts_map([make_image(observation_frame), make_image(observation_frame, livetime=20 * u.s)])
    exposure = maker.make_exposure_map(images)

    assert isinstance(counts, hist.Hist)
    assert isinstance(exposure, hist.Hist)
    assert counts.axes.name == ("x", "y", "energy")
    assert exposure.axes.name == ("x", "y")
    assert counts.values().sum() == pytest.approx(4)
    assert exposure.values().shape == (2, 2)
    np.testing.assert_allclose(exposure.values(), 30)


def test_make_counts_and_exposure_maps_accept_single_image(observation_frame):
    maker = BkgMaker()
    image = make_image(observation_frame)

    counts = maker.make_counts_map(image)
    exposure = maker.make_exposure_map(image)

    np.testing.assert_allclose(counts.values(), maker.make_counts_map([image]).values())
    np.testing.assert_allclose(exposure.values(), maker.make_exposure_map([image]).values())


def test_exclusion_region_masks_count_and_exposure_pixels(observation_frame):
    fine_edges = np.linspace(-1, 1, 5)
    maker = BkgMaker(downsampling=2)
    image = make_image(observation_frame, x_edges=fine_edges, y_edges=fine_edges)
    exclusion = exclusion_region(observation_frame)

    counts = maker.make_counts_map([image], exclusion)
    exposure = maker.make_exposure_map([image], exclusion)

    assert counts.values()[1, 1, :].sum() == 0
    assert counts.values().sum() == pytest.approx(1)
    assert np.count_nonzero(exposure.values() == 0) > 0
    assert np.count_nonzero(exposure.values() == 10) > 0


def test_downsampling_masks_real_fine_events_and_averages_exposure():
    frame = make_altaz_frame()
    fine_edges = np.linspace(-1, 1, 9)
    # Two events inside the upper-right coarse pixel [0, 1] x [0, 1]:
    # (0.5, 0.5) falls inside the exclusion circle; (0.125, 0.125) is unexcluded.
    image = make_image(
        frame,
        x=(0.5, 0.125),
        y=(0.5, 0.125),
        energy=(0.5, 5),
        x_edges=fine_edges,
        y_edges=fine_edges,
    )
    center = SkyCoord(0.5, 0.5, unit=u.deg, frame=frame.fov_frame).icrs
    exclusion = RegionGeom.from_regions(CircleSkyRegion(center=center, radius=0.2 * u.deg))
    maker = BkgMaker(downsampling=4)

    counts = maker.make_counts_map(image, exclusion, downsample=False)
    coarse_counts = maker.make_counts_map(image, exclusion)
    exposure = maker.make_exposure_map(image, exclusion, downsample=False)
    coarse_exposure = maker.make_exposure_map(image, exclusion)

    assert counts.values().shape == (8, 8, 2)
    assert coarse_counts.values().shape == (2, 2, 2)
    assert exposure.values().shape == (8, 8)
    assert coarse_exposure.values().shape == (2, 2)
    assert np.count_nonzero(exposure.values() == 0) == 4
    # Only the genuinely unexcluded event survives (exact integer 1, no leaked source fraction).
    assert coarse_counts.values()[1, 1, 0] == 0
    assert coarse_counts.values()[1, 1, 1] == 1
    assert coarse_exposure.values()[1, 1] == pytest.approx(7.5)


def test_downsampling_requires_divisible_uniform_spatial_bins(observation_frame):
    non_divisible = make_image(observation_frame, x_edges=np.linspace(-1, 1, 4), y_edges=np.linspace(-1, 1, 4))
    non_uniform = make_image(observation_frame, x_edges=[-1, -0.5, 0, 0.2, 1], y_edges=np.linspace(-1, 1, 5))
    maker = BkgMaker(downsampling=2)

    with pytest.raises(ValueError, match="divisible by downsampling"):
        maker.make_counts_map(non_divisible)
    with pytest.raises(ValueError, match="Uniform spatial binning"):
        maker.make_counts_map(non_uniform)


def test_convolve_uses_the_same_kernel_for_counts_and_exposure():
    x_axis = hist.axis.Variable([0, 1, 2, 3], name="x", underflow=False, overflow=False)
    y_axis = hist.axis.Variable([0, 1, 2, 3], name="y", underflow=False, overflow=False)
    e_axis = hist.axis.Variable([0.1, 1], name="energy", underflow=False, overflow=False)

    counts = hist.Hist(x_axis, y_axis, e_axis)
    counts.values()[1, 1, 0] = 2
    exposure = hist.Hist(x_axis, y_axis)
    exposure.values()[1, 1] = 20

    convolved_counts, convolved_exposure = BkgMaker().convolve(counts, exposure, 1 * u.deg)

    assert isinstance(convolved_counts, hist.Hist)
    assert isinstance(convolved_exposure, hist.Hist)
    assert convolved_exposure.axes.name == ("x", "y", "energy")
    np.testing.assert_allclose(convolved_exposure.values(), convolved_counts.values() * 10)
    assert convolved_counts.values()[1, 1, 0] == 2
    assert convolved_counts.values()[0, 1, 0] == 2
    assert convolved_counts.values()[0, 0, 0] == 0


def test_public_maps_support_convolution_and_reject_unrebinned_convolution(observation_frame):
    maker = BkgMaker()
    image = make_image(observation_frame)

    counts = maker.make_counts_map([image], kernel_radius=1 * u.deg)
    exposure = maker.make_exposure_map((img for img in [image]), kernel_radius=1 * u.deg)
    expected_counts, expected_exposure = maker.convolve(
        maker.make_counts_map([image]),
        maker.make_exposure_map([image]),
        1 * u.deg,
    )

    np.testing.assert_allclose(counts.values(), expected_counts.values())
    np.testing.assert_allclose(exposure.values(), expected_exposure.values())

    with pytest.raises(ValueError, match="downsample=False"):
        maker.make_counts_map([image], kernel_radius=1 * u.deg, downsample=False)
    with pytest.raises(ValueError, match="downsample=False"):
        maker.make_exposure_map([image], kernel_radius=1 * u.deg, downsample=False)


def test_public_maps_reject_mixed_fov_frames(observation_frame):
    icrs_image = make_image(observation_frame)
    altaz_image = make_image(make_altaz_frame())

    with pytest.raises(TypeError, match="same FoV frame"):
        BkgMaker().make_counts_map([icrs_image, altaz_image])


def test_make_icrs_background_map_returns_normalized_gammapy_background(observation_frame):
    image = make_image(observation_frame)
    maker = BkgMaker()

    background = maker.make_icrs_background_map([image])

    assert isinstance(background, Background3D)
    assert background.axes.names == ["energy", "fov_lon", "fov_lat"]
    assert background.axes["energy"].interp == "log"
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

    with pytest.raises(TypeError, match="FoVICRSFrame"):
        BkgMaker().make_icrs_background_map([make_image(frame)])


def test_make_icrs_background_map_requires_matching_axes(observation_frame):
    first = make_image(observation_frame)
    second = CameraImage([-2, 0, 2], [-1, 0, 1], [0.1, 1, 10], frame=observation_frame)

    with pytest.raises(ValueError, match="identical axes"):
        BkgMaker().make_icrs_background_map([first, second])


def test_make_icrs_background_map_requires_nonempty_images():
    with pytest.raises(ValueError, match="At least one"):
        BkgMaker().make_icrs_background_map([])


def test_make_altaz_background_map_returns_normalized_altaz_background():
    image = make_image(make_altaz_frame())

    background = BkgMaker().make_altaz_background_map([image])

    expected = 1 / (10 * 900_000 * image.pixel_solid_angle[0, 0])
    assert background.data[0, 0, 0] == pytest.approx(expected)
    assert background.fov_alignment == FoVAlignment.ALTAZ


def test_altaz_exclusion_masks_counts_and_exposure():
    frame = make_altaz_frame()
    fine_edges = np.linspace(-1, 1, 5)
    image = make_image(frame, x_edges=fine_edges, y_edges=fine_edges)
    exclusion = exclusion_region(frame)
    maker = BkgMaker(downsampling=2)

    counts = maker.make_counts_map([image], exclusion)
    exposure = maker.make_exposure_map([image], exclusion)

    assert counts.values()[1, 1, :].sum() == 0
    assert counts.values().sum() == pytest.approx(1)
    assert np.count_nonzero(exposure.values() == 0) > 0
    assert np.count_nonzero(exposure.values() == 10) > 0


def test_altaz_exclusion_requires_scalar_pointing():
    vector_image = make_image(make_altaz_frame(vector=True))
    exclusion = exclusion_region(make_altaz_frame())
    maker = BkgMaker()

    background = maker.make_altaz_background_map([vector_image])
    assert background.fov_alignment == FoVAlignment.ALTAZ
    assert maker.make_exposure_map([vector_image]).values()[0, 0] == pytest.approx(10)

    with pytest.raises(ValueError, match="scalar-pointing time slices"):
        maker.make_altaz_background_map([vector_image], exclusion_region=exclusion)
    with pytest.raises(ValueError, match="scalar-pointing time slices"):
        maker.make_counts_map([vector_image], exclusion_region=exclusion)
    with pytest.raises(ValueError, match="scalar-pointing time slices"):
        maker.make_exposure_map([vector_image], exclusion_region=exclusion)


def test_make_altaz_background_map_requires_altaz_frames(observation_frame):
    with pytest.raises(TypeError, match="FoVAltAzFrame"):
        BkgMaker().make_altaz_background_map([make_image(observation_frame)])


def test_time_sliced_altaz_background_end_to_end():
    events = SkyOffsetEvents(
        x=[-0.5, 0.125],
        y=[-0.5, 0.125],
        energy=[0.5, 5],
        livetime=10 * u.s,
        frame=make_altaz_frame(vector=True),
    )
    fine_edges = np.linspace(-1, 1, 9)
    slices = events.time_slices()
    first_slice_image = CameraImage.from_events(slices[0], fine_edges, fine_edges, [0.1, 1, 10])
    center = SkyCoord(0.5, 0.5, unit=u.deg, frame=first_slice_image.frame.fov_frame).icrs
    exclusion = RegionGeom.from_regions(CircleSkyRegion(center=center, radius=0.2 * u.deg))

    maker = BkgMaker(downsampling=4)
    background = maker.make_altaz_background_map(
        (CameraImage.from_events(s, fine_edges, fine_edges, [0.1, 1, 10]) for s in slices),
        exclusion_region=exclusion,
    )
    unmasked = maker.make_altaz_background_map(
        CameraImage.from_events(s, fine_edges, fine_edges, [0.1, 1, 10]) for s in slices
    )
    exposure = maker.make_exposure_map(
        (CameraImage.from_events(s, fine_edges, fine_edges, [0.1, 1, 10]) for s in slices),
        exclusion_region=exclusion,
    )

    assert background.fov_alignment == FoVAlignment.ALTAZ
    assert background.data.shape == (2, 2, 2)
    assert np.all(np.isfinite(background.data))
    assert background.data[1, 1, 1] > unmasked.data[1, 1, 1]
    assert background.data[0, 0, 0] == pytest.approx(unmasked.data[0, 0, 0])
    assert exposure.values()[1, 1] < 10
    assert exposure.values()[0, 0] == pytest.approx(10)
