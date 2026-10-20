"""Histograms of reconstructed events in telescope-centered coordinates."""

from collections.abc import Iterable

import astropy.units as u
import hist
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis
from scipy.ndimage import convolve

from .events import SkyOffsetEvents
from .utils import validate_edges


class CameraImage:
    """Accumulate events and estimate the camera background rate.

    Parameters
    ----------
    x_edges, y_edges : array-like
        Spatial bin edges in degrees.

    e_edges : array-like
        Reconstructed-energy bin edges in TeV.

    exposure_oversampling : int
        Spatial refinement factor for the exposure map.
        For example, 4 means that each counts pixel corresponds
        to 4 x 4 finer exposure pixels.

    Notes
    -----
    The class maintains three quantities:

    - histogram: Event counts (x, y, energy).
    - exposure_map: Fine-resolution livetime map (x, y).
    - exposure: Effective livetime (x, y, energy).

    Events must be accumulated before applying convolution.
    """

    def __init__(
        self,
        x_edges: Iterable[float],
        y_edges: Iterable[float],
        e_edges: Iterable[float],
        exposure_oversampling: int = 4,
    ) -> None:

        self.x_edges = validate_edges(x_edges, "x_edges", u.deg)
        self.y_edges = validate_edges(y_edges, "y_edges", u.deg)
        self.e_edges = validate_edges(e_edges, "e_edges", u.TeV)

        if not isinstance(exposure_oversampling, (int, np.integer)) or exposure_oversampling < 1:
            raise ValueError("exposure_oversampling must be a positive integer")

        self.exposure_oversampling = int(exposure_oversampling)

        # Event counts: (x, y, energy)
        self.histogram = hist.Hist(
            self._make_axis(self.x_edges, "x"),
            self._make_axis(self.y_edges, "y"),
            self._make_axis(self.e_edges, "energy"),
        )

        # Fine-resolution livetime map: (x, y)
        fine_x_edges = self._refine_edges(self.x_edges, self.exposure_oversampling)
        fine_y_edges = self._refine_edges(self.y_edges, self.exposure_oversampling)

        self.exposure_map = hist.Hist(
            self._make_axis(fine_x_edges, "x"),
            self._make_axis(fine_y_edges, "y"),
        )

        # Energy-dependent exposure after convolution.
        # None means that convolution has not been applied.
        self._convolved_exposure = None

        self.live_time = 0 * u.s

    # ---------------------------------------------------------
    # Histogram geometry
    # ---------------------------------------------------------

    @staticmethod
    def _make_axis(edges, name):
        """Create a histogram axis without flow bins."""
        return hist.axis.Variable(
            edges,
            name=name,
            underflow=False,
            overflow=False,
        )

    @staticmethod
    def _refine_edges(edges, factor):
        """Subdivide each spatial bin into `factor` equal bins."""

        edges = np.asarray(edges)

        fractions = np.arange(factor) / factor

        refined = edges[:-1, None] + np.diff(edges)[:, None] * fractions

        return np.concatenate([refined.ravel(), [edges[-1]]])

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Return the counts pixel dimensions (dx, dy) in degrees.

        Requires uniform spacing along each spatial axis.
        """

        dx = np.diff(self.x_edges)
        dy = np.diff(self.y_edges)

        if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
            raise ValueError("Uniform spatial binning is required for convolution.")

        return float(dx[0]), float(dy[0])

    @property
    def pixel_solid_angle(self) -> np.ndarray:
        """Return the solid angle of each counts pixel in sr.

        Shape: (n_x, n_y).

        Uses the spherical longitude-latitude area formula.
        """

        lon_width = np.diff(np.deg2rad(self.x_edges))

        lat_width = np.diff(np.deg2rad(self.y_edges))
        # lat_width = np.diff(np.sin(lat_edges))

        return lon_width[:, None] * lat_width[None, :]

    # ---------------------------------------------------------
    # Exposure
    # ---------------------------------------------------------

    @property
    def exposure(self) -> hist.Hist:
        """Return effective exposure with shape (x, y, energy).

        Before convolution:
            The 2D fine exposure map is downsampled by averaging
            and broadcast along the energy axis.

        After convolution:
            Return the energy-dependent convolved exposure.

        Values are in seconds.
        """

        result = self.histogram.copy()
        result.reset()

        if self._convolved_exposure is not None:
            result.values()[:] = self._convolved_exposure
            return result

        # Downsample fine exposure to counts geometry.
        factor = self.exposure_oversampling

        coarse_exposure = self.exposure_map[
            :: hist.rebin(factor),
            :: hist.rebin(factor),
        ]

        # hist.rebin sums bins, but we need mean livetime.
        exposure_2d = coarse_exposure.values() / factor**2

        # Broadcast 2D exposure over the energy axis.
        result.values()[:] = exposure_2d[:, :, None]

        return result

    # ---------------------------------------------------------
    # Event accumulation
    # ---------------------------------------------------------

    @staticmethod
    def _exclusion_mask(events, x, y):
        """Evaluate source-exclusion regions at given coordinates."""

        if events.exclusion_regions is None:
            return np.zeros(np.shape(x), dtype=bool)

        coordinates = SkyCoord(
            x * u.deg,
            y * u.deg,
            frame=events.frame,
        )

        return np.asarray(
            events.exclusion_regions.contains(coordinates),
            dtype=bool,
        )

    def fill(self, events: SkyOffsetEvents) -> None:
        """Accumulate source-free events and available livetime.

        Counts:
            Accumulate events outside exclusion regions.

        Exposure:
            Add observation livetime to unmasked fine pixels.

        Both operations use the same exclusion regions.
        """

        if not isinstance(events, SkyOffsetEvents):
            raise TypeError("events must be a SkyOffsetEvents object")

        if self._convolved_exposure is not None:
            raise RuntimeError("Cannot fill after convolution. Accumulate all observations before convolving.")

        livetime = events.livetime.to_value(u.s)

        # 1. Accumulate exposure using fine spatial pixels.
        pix_x, pix_y = self.exposure_map.axes.centers

        xx, yy = np.meshgrid(pix_x, pix_y, indexing="ij")

        excluded_pixels = self._exclusion_mask(events, xx, yy)

        self.exposure_map.values()[:] += np.where(
            excluded_pixels,
            0.0,
            livetime,
        )

        # 2. Accumulate only source-free events.
        x = u.Quantity(events.x, u.deg).to_value(u.deg)
        y = u.Quantity(events.y, u.deg).to_value(u.deg)
        energy = u.Quantity(events.energy, u.TeV).to_value(u.TeV)

        excluded_events = self._exclusion_mask(events, x, y)

        keep = ~excluded_events

        self.histogram.fill(
            x=x[keep],
            y=y[keep],
            energy=energy[keep],
        )

        self.live_time += events.livetime

    # ---------------------------------------------------------
    # Spatial convolution
    # ---------------------------------------------------------

    def convolve(self, kernel_radius: Iterable[float]) -> None:
        """Apply energy-dependent circular top-hat convolution.

        Parameters
        ----------
        kernel_radius : array-like or Quantity
            Kernel radius in degrees for each energy bin.
            A scalar applies the same radius to every energy bin.

        Notes
        -----
        The kernel is NOT normalized.

        Both counts and exposure are summed over neighboring
        pixels using exactly the same spatial kernel.

        Outside the map, counts and exposure are treated as zero.

        After convolution, the effective exposure becomes 3D.
        """

        if self._convolved_exposure is not None:
            raise RuntimeError("Convolution has already been applied")

        radii = np.atleast_1d(u.Quantity(kernel_radius, u.deg).to_value(u.deg))

        n_energy = self.histogram.axes[2].size

        if radii.size == 1:
            radii = np.repeat(radii, n_energy)

        if radii.size != n_energy:
            raise ValueError("kernel_radius must be scalar or contain one radius per energy bin")

        if not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("Kernel radii must be finite and nonnegative")

        dx, dy = self.pixel_size

        counts = self.histogram.values().copy()

        # Initial exposure is energy-independent.
        exposure_2d = self.exposure.values()[:, :, 0]

        convolved_counts = np.empty_like(counts)
        convolved_exposure = np.empty_like(counts)

        for i, radius in enumerate(radii):
            # Build a circular kernel in counts-pixel coordinates.
            nx = int(np.ceil(radius / dx))
            ny = int(np.ceil(radius / dy))

            x = np.arange(-nx, nx + 1) * dx
            y = np.arange(-ny, ny + 1) * dy

            xx, yy = np.meshgrid(x, y, indexing="ij")

            kernel = (xx**2 + yy**2 <= radius**2).astype(float)

            # Sum events within the aperture.
            convolved_counts[:, :, i] = convolve(
                counts[:, :, i],
                kernel,
                mode="constant",
                cval=0.0,
            )

            # Sum available livetime over the same aperture.
            convolved_exposure[:, :, i] = convolve(
                exposure_2d,
                kernel,
                mode="constant",
                cval=0.0,
            )

        self.histogram.values()[:] = convolved_counts
        self._convolved_exposure = convolved_exposure

    # ---------------------------------------------------------
    # Background-rate estimation
    # ---------------------------------------------------------

    def to_rate(self) -> hist.Hist:
        """Calculate differential background rate.

        Returns
        -------
        hist.Hist
            Differential background rate with axes (x, y, energy).

        Units
        -----
        s^-1 MeV^-1 sr^-1

        Notes
        -----
        Pixels with zero effective exposure are set to NaN.
        No rate is inferred for those pixels.
        """

        counts = self.histogram.values()
        exposure = self.exposure.values()

        # Energy-bin widths in MeV.
        energy_edges = u.Quantity(self.e_edges, u.TeV).to_value(u.MeV)
        energy_width = np.diff(energy_edges)

        # Counts-pixel solid angles in sr.
        solid_angle = self.pixel_solid_angle

        denominator = exposure * energy_width[None, None, :] * solid_angle[:, :, None]

        rate = np.full_like(counts, np.nan, dtype=float)

        np.divide(
            counts,
            denominator,
            out=rate,
            where=denominator > 0,
        )

        result = self.histogram.copy()
        result.values()[:] = rate

        return result

    # ---------------------------------------------------------
    # Gammapy IRF
    # ---------------------------------------------------------

    def to_bkg_3d(
        self,
        filename: str | None = None,
        *,
        fov_alignment: FoVAlignment = FoVAlignment.RADEC,
    ) -> Background3D:
        """Convert the differential background rate to a Gammapy IRF.

        Parameters
        ----------
        filename : str, optional
            Output FITS filename. If None, no file is written.

        fov_alignment : FoVAlignment
            Field-of-view coordinate alignment.

        Returns
        -------
        Background3D
            Background rate in s^-1 MeV^-1 sr^-1.
        """

        rate = self.to_rate().values()

        if not np.all(np.isfinite(rate)):
            raise ValueError(
                "The background rate contains undefined values. Handle pixels with zero exposure before exporting."
            )

        energy_axis = MapAxis.from_edges(
            self.e_edges,
            name="energy",
            unit="TeV",
        )

        lon_axis = MapAxis.from_edges(
            self.x_edges,
            name="fov_lon",
            unit="deg",
        )

        lat_axis = MapAxis.from_edges(
            self.y_edges,
            name="fov_lat",
            unit="deg",
        )

        # Hist:       (x, y, energy)
        # Background3D: (energy, fov_lon, fov_lat)
        data = np.moveaxis(rate, 2, 0)

        background = Background3D(
            axes=[energy_axis, lon_axis, lat_axis],
            data=data,
            unit="s-1 MeV-1 sr-1",
            fov_alignment=fov_alignment,
        )

        if filename is not None:
            background.write(filename, overwrite=True)

        return background

    # ---------------------------------------------------------
    # Radial histogram
    # ---------------------------------------------------------

    def to_radial(self, theta_edges: Iterable[float]) -> hist.Hist:
        """Convert accumulated counts to a radial histogram.

        Each original spatial bin contributes to the radial bin
        containing its center.

        The output axes are (energy, theta).

        This method must be called before spatial convolution.
        """

        if self._convolved_exposure is not None:
            raise RuntimeError("Radial binning must be performed before convolution.")

        edges = validate_edges(theta_edges, "theta_edges", u.deg)

        x_centers = self.histogram.axes[0].centers
        y_centers = self.histogram.axes[1].centers

        xx, yy = np.meshgrid(
            x_centers,
            y_centers,
            indexing="ij",
        )

        radius = np.hypot(xx, yy)

        counts = self.histogram.values()

        radial_counts = np.stack(
            [
                np.histogram(
                    radius,
                    bins=edges,
                    weights=counts[:, :, i],
                )[0]
                for i in range(counts.shape[2])
            ]
        )

        result = hist.Hist(
            self._make_axis(self.e_edges, "energy"),
            self._make_axis(edges, "theta"),
        )

        result.values()[:] = radial_counts

        return result
