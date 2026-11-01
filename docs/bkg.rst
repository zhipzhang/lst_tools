Camera-coordinate background models
===================================

The :mod:`lst_tools.bkg` module builds background models from LST event
samples in telescope sky-offset coordinates. The workflow is:

1. Convert LST DL2/DL3 events into :class:`~lst_tools.bkg.SkyOffsetEvents`.
2. Bin events into per-observation :class:`~lst_tools.bkg.CameraImage` objects.
3. Combine images or events into a Gammapy-compatible background IRF.

All coordinates are stored in degrees and energies in TeV internally, but the
public constructors accept any compatible :class:`astropy.units.Quantity`.

Event-level representation
--------------------------

A :class:`~lst_tools.bkg.SkyOffsetEvents` object holds one observation's
reconstructed directions and energies together with the metadata needed to
define its coordinate frame:

.. code-block:: python

   from lst_tools.bkg import ObservationFrame, SkyOffsetEvents
   from astropy.coordinates import SkyCoord
   from astropy.time import Time
   import astropy.units as u
   from gammapy.utils.coordinates import FoVICRSFrame

   origin = SkyCoord(ra=83.5, dec=22.0, unit=u.deg, frame="icrs")
   frame = ObservationFrame(
       fov_frame=FoVICRSFrame(origin=origin),
       mean_time=Time(1_700_000_000, format="unix"),
       pointing_alt=70 * u.deg,
       pointing_az=10 * u.deg,
   )

   events = SkyOffsetEvents(
       x=[-0.5, 0.25, 1.0] * u.deg,
       y=[0.0, 0.5, -0.25] * u.deg,
       energy=[0.2, 1.0, 4.0] * u.TeV,
       livetime=10 * u.s,
       frame=frame,
   )

``SkyOffsetEvents`` intentionally does not combine observations or manage
exclusion regions; those belong to the background-making layer described
below. Following the GADF/Gammapy background-IRF convention, ``livetime``
stores the elapsed observation duration (ontime, including dead time) so that
Gammapy's ``make_map_background_irf`` integration over ``ontime`` reproduces
observed background counts directly.

Adapters from LST data
----------------------

For LST DL2 files, use :func:`~lst_tools.bkg.sky_offset_events_from_lstdl2`:

.. code-block:: python

   from lst_tools.dl2 import LSTDL2EventTable
   from lst_tools.bkg import sky_offset_events_from_lstdl2

   dl2 = LSTDL2EventTable.from_paths(["dl2_Run01234.h5"])
   events = sky_offset_events_from_lstdl2(dl2)

An optional ``event_filter`` receives the full DL2 DataFrame and returns the
rows to retain. The complete run elapsed time (``dl2.t_elapsed``) is preserved,
so the filter models event selection without changing exposure:

.. code-block:: python

   events = sky_offset_events_from_lstdl2(
       dl2,
       event_filter=lambda data: data.loc[data["reco_energy"] > 1],
   )

DL3 events can be converted with :func:`~lst_tools.bkg.lst_adapter.sky_offset_events_from_dl3`:

.. code-block:: python

   from gammapy.data import DataStore
   from lst_tools.bkg.lst_adapter import sky_offset_events_from_dl3

   obs = DataStore.from_dir("dl3").obs(1234)
   events = sky_offset_events_from_dl3(obs.events)

Camera images
-------------

A :class:`~lst_tools.bkg.CameraImage` is a counts histogram for a single
observation, together with its livetime and frame metadata:

.. code-block:: python

   from lst_tools.bkg import CameraImage

   image = CameraImage.from_events(
       events,
       x_edges=[-2, -1, 0, 1, 2] * u.deg,
       y_edges=[-2, -1, 0, 1, 2] * u.deg,
       e_edges=[0.1, 1, 10] * u.TeV,
   )

``CameraImage`` is deliberately per-observation: it never accumulates events
from multiple observations, so frames and livetimes cannot be mixed
accidentally. Use :meth:`~lst_tools.bkg.CameraImage.to_radial` to rebin a
camera image into energy and offset-radius histograms.

Radial (2D) background models
-----------------------------

:class:`~lst_tools.bkg.Background2DMaker` produces a Gammapy
:class:`gammapy.irf.Background2D` from a single combined sample of
sky-offset events. It histograms counts in reconstructed energy and offset
radius and normalizes by livetime, annular solid angle, and an effective
energy width:

.. code-block:: python

   from lst_tools.bkg import Background2DMaker

   maker = Background2DMaker(
       energy_edges=[0.1, 1, 10] * u.TeV,
       theta_edges=[0, 0.5, 1, 2] * u.deg,
       spectral_index=-2.0,
   )
   background_2d = maker(events)
   background_2d.write("background_2d.fits", overwrite=True)

The ``spectral_index`` is the assumed power-law index used to convert an
integral rate in each energy bin into a differential rate at the logarithmic
bin center. Set it to the spectral shape you want the model to assume.

Full-FoV (3D) background maps
-----------------------------

:class:`~lst_tools.bkg.BkgMaker` combines multiple ICRS or AltAz
:class:`~lst_tools.bkg.CameraImage` objects into a Gammapy
:class:`gammapy.irf.Background3D`. It handles masking, exposure accumulation,
optional spatial downsampling, optional top-hat convolution, and
differential-rate normalization:

.. code-block:: python

   from lst_tools.bkg import BkgMaker
   from gammapy.maps import RegionGeom
   from regions import CircleSkyRegion

   # Bin CameraImage on a fine spatial grid (e.g. 4x finer than the target IRF grid)
   images = [image_from_run_a, image_from_run_b]

   exclusion = RegionGeom.from_regions(
       CircleSkyRegion(center=source_position, radius=0.3 * u.deg)
   )

   maker = BkgMaker(downsampling=4)
   background_3d = maker.make_icrs_background_map(
       images,
       exclusion_region=exclusion,
       kernel_radius=0.2 * u.deg,
   )

Masking and accumulation happen directly on the input ``CameraImage`` grid so
that excluded source events are removed at true event resolution without
leaking across partial coarse pixels. When ``downsampling > 1``, the
accumulated ``counts`` and ``exposure`` histograms are rebinned into
``coarse_counts`` (sum) and ``coarse_exposure`` (mean), optionally convolved
with the same circular top-hat kernel per energy bin, and divided to form a
differential rate in ``s-1 MeV-1 sr-1`` evaluated on a logarithmic energy axis
(``interp="log"``, where linear ``ΔE`` matches an :math:`E^{-2}` spectrum at
logarithmic bin centers). Pixels with zero exposure raise an error so that
unmasked gaps cannot be silently exported.

:meth:`~lst_tools.bkg.BkgMaker.make_counts_map` and
:meth:`~lst_tools.bkg.BkgMaker.make_exposure_map` expose the :class:`hist.Hist`
maps for a single image or any iterable of images. Both return the downsampled
coarse histogram by default; pass ``downsample=False`` to inspect the native
pre-downsampling histogram, or pass ``kernel_radius`` to inspect the
top-hat-smoothed maps:

.. code-block:: python

   coarse_counts = maker.make_counts_map(images, exclusion_region=exclusion)
   coarse_exposure = maker.make_exposure_map(images, exclusion_region=exclusion)

   native_exposure = maker.make_exposure_map(images, exclusion_region=exclusion, downsample=False)

For an AltAz model without exclusion, whole-run images can be passed directly
to :meth:`~lst_tools.bkg.BkgMaker.make_altaz_background_map`. An ICRS exclusion
moves in FoV-AltAz coordinates, so time-varying observations must first be
split into scalar-pointing slices (which can be streamed as a generator):

.. code-block:: python

   events = sky_offset_events_from_lstdl2(dl2, frame="altaz")
   event_slices = events.time_slices(max_fov_rotation=0.2 * u.deg)
   images = (
       CameraImage.from_events(slice_, fine_x_edges, fine_y_edges, e_edges)
       for slice_ in event_slices
   )

   background_altaz = maker.make_altaz_background_map(
       images,
       exclusion_region=exclusion,
       kernel_radius=0.2 * u.deg,
   )

The slicer computes the spherical mean ICRS pointing across the observation,
samples the field-rotation rate over ``[t_min, t_max]`` (using the peak rate so
runs crossing due East or West are not under-sliced), and divides the span into
equal-duration slices. It preserves events and total livetime, including
exposure in slices containing no events. Each image keeps a scalar livetime;
the maker derives pixel-wise effective exposure by masking counts and exposure
with each slice's frame. Fixed scalar-frame inputs, such as the MC adapter
output, do not require slicing.

Visualization
-------------

Two plotting helpers are provided:

.. code-block:: python

   from lst_tools.bkg import plot_energy_slices, plot_radial_acceptance

   fig, axes = plot_energy_slices(image)

   fig, ax = plot_radial_acceptance(
       events,
       energy_edges=[0.1, 1, 10] * u.TeV,
       theta_edges=np.linspace(0, 0.5, 11) * u.deg,
       density=True,
   )

:func:`~lst_tools.bkg.plot_energy_slices` shows one panel per energy bin of a
camera image. :func:`~lst_tools.bkg.plot_radial_acceptance` plots the
solid-angle-corrected radial acceptance of an event sample, optionally as a
probability density or against ``theta²``.
