"""Combine camera observations into Gammapy background models."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import astropy.units as u
import hist
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis, RegionGeom
from gammapy.utils.coordinates import FoVAltAzFrame, FoVICRSFrame
from scipy.ndimage import convolve as ndi_convolve

from .camera import CameraImage


class BkgMaker:
    """Mask, accumulate, downsample, smooth, and normalize camera images into background models."""

    def __init__(self, downsampling: int = 1) -> None:
        if isinstance(downsampling, bool) or not isinstance(downsampling, (int, np.integer)) or downsampling < 1:
            raise ValueError("downsampling must be a positive integer")
        self.downsampling = int(downsampling)

    def _validate_reference_axes(self, reference: CameraImage) -> None:
        if self.downsampling == 1:
            return
        x_axis, y_axis = reference.histogram.axes[0], reference.histogram.axes[1]
        if x_axis.size % self.downsampling != 0 or y_axis.size % self.downsampling != 0:
            raise ValueError("CameraImage spatial bin counts must be divisible by downsampling")
        self._spatial_pixel_size(x_axis, y_axis)

    @staticmethod
    def _validate_image(
        image: CameraImage,
        reference: CameraImage,
        frame_type: type[FoVICRSFrame] | type[FoVAltAzFrame],
        required_frame: type[FoVICRSFrame] | type[FoVAltAzFrame] | None,
        exclusion_region: RegionGeom | None,
    ) -> None:
        if not isinstance(image.frame.fov_frame, frame_type):
            if required_frame is not None:
                raise TypeError(f"Background map requires {required_frame.__name__} images")
            raise TypeError("All CameraImage objects must use the same FoV frame type")
        if image is not reference and not all(
            np.array_equal(getattr(image, axis), getattr(reference, axis)) for axis in ("x_edges", "y_edges", "e_edges")
        ):
            raise ValueError("All CameraImage objects must have identical axes")
        if (
            exclusion_region is not None
            and issubclass(frame_type, FoVAltAzFrame)
            and not image.frame.fov_frame.origin.isscalar
        ):
            raise ValueError("AltAz exclusion requires scalar-pointing time slices")

    def _iter_validated_images(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None = None,
        *,
        required_frame: type[FoVICRSFrame] | type[FoVAltAzFrame] | None = None,
    ) -> Iterator[CameraImage]:
        if isinstance(images, CameraImage):
            stream: Iterator[CameraImage] = iter((images,))
        else:
            try:
                stream = iter(images)
            except TypeError as error:
                raise TypeError("images must be a CameraImage or iterable of CameraImage objects") from error

        try:
            reference = next(stream)
        except StopIteration:
            raise ValueError("At least one CameraImage is required") from None

        if not isinstance(reference, CameraImage):
            raise TypeError("images must contain only CameraImage objects")

        if required_frame is not None:
            frame_type = required_frame
        elif isinstance(reference.frame.fov_frame, FoVICRSFrame):
            frame_type = FoVICRSFrame
        else:
            frame_type = FoVAltAzFrame

        self._validate_reference_axes(reference)
        self._validate_image(reference, reference, frame_type, required_frame, exclusion_region)
        yield reference

        for image in stream:
            if not isinstance(image, CameraImage):
                raise TypeError("images must contain only CameraImage objects")
            self._validate_image(image, reference, frame_type, required_frame, exclusion_region)
            yield image

    @staticmethod
    def _exclusion_mask(
        image: CameraImage,
        exclusion_region: RegionGeom | None,
    ) -> np.ndarray:
        x_axis, y_axis = image.histogram.axes[0], image.histogram.axes[1]
        if exclusion_region is None:
            return np.zeros((x_axis.size, y_axis.size), dtype=bool)
        x, y = np.meshgrid(x_axis.centers, y_axis.centers, indexing="ij")
        coordinates = SkyCoord(x * u.deg, y * u.deg, frame=image.frame.fov_frame)
        return np.asarray(exclusion_region.contains(coordinates.icrs), dtype=bool)

    @staticmethod
    def _masked_counts(image: CameraImage, excluded: np.ndarray) -> hist.Hist:
        counts = image.histogram.copy()
        counts.values()[excluded, :] = 0.0
        return counts

    @staticmethod
    def _masked_exposure(image: CameraImage, excluded: np.ndarray) -> hist.Hist:
        x_axis, y_axis = image.histogram.axes[0], image.histogram.axes[1]
        exposure = hist.Hist(x_axis, y_axis)
        exposure.values()[:] = np.where(excluded, 0.0, image.live_time.to_value(u.s))
        return exposure

    def _accumulate(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None,
        *,
        required_frame: type[FoVICRSFrame] | type[FoVAltAzFrame] | None = None,
    ) -> tuple[hist.Hist, hist.Hist]:
        stream = self._iter_validated_images(images, exclusion_region, required_frame=required_frame)
        first = next(stream)
        excluded = self._exclusion_mask(first, exclusion_region)
        counts = self._masked_counts(first, excluded)
        exposure = self._masked_exposure(first, excluded)
        for image in stream:
            excluded = self._exclusion_mask(image, exclusion_region)
            counts += self._masked_counts(image, excluded)
            exposure += self._masked_exposure(image, excluded)
        return counts, exposure

    def _accumulate_counts(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None,
    ) -> hist.Hist:
        stream = self._iter_validated_images(images, exclusion_region)
        first = next(stream)
        counts = self._masked_counts(first, self._exclusion_mask(first, exclusion_region))
        for image in stream:
            counts += self._masked_counts(image, self._exclusion_mask(image, exclusion_region))
        return counts

    def _accumulate_exposure(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None,
    ) -> tuple[hist.Hist, hist.axis.AxesMixin]:
        stream = self._iter_validated_images(images, exclusion_region)
        first = next(stream)
        e_axis = first.histogram.axes[2]
        exposure = self._masked_exposure(first, self._exclusion_mask(first, exclusion_region))
        for image in stream:
            exposure += self._masked_exposure(image, self._exclusion_mask(image, exclusion_region))
        return exposure, e_axis

    def _downsample_counts(self, counts: hist.Hist) -> hist.Hist:
        factor = self.downsampling
        if factor == 1:
            return counts
        coarse_counts = counts[:: hist.rebin(factor), :: hist.rebin(factor), :]
        return coarse_counts

    def _downsample_exposure(self, exposure: hist.Hist) -> hist.Hist:
        factor = self.downsampling
        if factor == 1:
            return exposure
        coarse_exposure = exposure[:: hist.rebin(factor), :: hist.rebin(factor)]
        coarse_exposure.values()[:] /= factor**2
        return coarse_exposure

    @staticmethod
    def _make_kernel(radius: float, pixel_size: tuple[float, float]) -> np.ndarray:
        dx, dy = pixel_size
        nx = int(np.ceil(radius / dx))
        ny = int(np.ceil(radius / dy))
        x = np.arange(-nx, nx + 1) * dx
        y = np.arange(-ny, ny + 1) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return (xx**2 + yy**2 <= radius**2).astype(float)

    @staticmethod
    def _spatial_pixel_size(
        x_axis: hist.axis.AxesMixin,
        y_axis: hist.axis.AxesMixin,
    ) -> tuple[float, float]:
        dx = np.diff(x_axis.edges)
        dy = np.diff(y_axis.edges)
        if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
            raise ValueError("Uniform spatial binning is required")
        pixel_size = (float(dx[0]), float(dy[0]))
        if not all(np.isfinite(pixel_size)) or min(pixel_size) <= 0:
            raise ValueError("Spatial axes must have positive finite bin widths")
        return pixel_size

    @staticmethod
    def _validate_kernel_radii(
        kernel_radius: Iterable[float] | u.Quantity,
        n_energy: int,
    ) -> np.ndarray:
        radii = np.atleast_1d(u.Quantity(kernel_radius, u.deg).to_value(u.deg))
        if radii.size == 1:
            radii = np.repeat(radii, n_energy)
        if radii.size != n_energy:
            raise ValueError("kernel_radius must be scalar or contain one radius per energy bin")
        if not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("kernel_radius must be finite and nonnegative")
        return radii

    def _convolve_3d(
        self,
        source: hist.Hist,
        e_axis: hist.axis.AxesMixin,
        kernel_radius: Iterable[float] | u.Quantity,
    ) -> hist.Hist:
        x_axis, y_axis = source.axes[0], source.axes[1]
        pixel_size = self._spatial_pixel_size(x_axis, y_axis)
        radii = self._validate_kernel_radii(kernel_radius, e_axis.size)

        result = hist.Hist(x_axis, y_axis, e_axis)
        source_values = source.values()
        out_values = result.values()
        for index, radius in enumerate(radii):
            kernel = self._make_kernel(radius, pixel_size)
            plane = source_values if source.ndim == 2 else source_values[:, :, index]
            out_values[:, :, index] = ndi_convolve(plane, kernel, mode="constant", cval=0.0)
        return result

    def convolve(
        self,
        counts: hist.Hist,
        exposure: hist.Hist,
        kernel_radius: Iterable[float] | u.Quantity,
    ) -> tuple[hist.Hist, hist.Hist]:
        """Convolve counts and exposure histograms with identical circular top-hat kernels."""
        if not isinstance(counts, hist.Hist) or not isinstance(exposure, hist.Hist):
            raise TypeError("counts and exposure must be hist.Hist objects")
        if counts.ndim != 3:
            raise ValueError("counts must be a 3D (x, y, energy) histogram")
        if exposure.ndim not in (2, 3):
            raise ValueError("exposure must be a 2D (x, y) or 3D (x, y, energy) histogram")
        if not (
            np.array_equal(counts.axes[0].edges, exposure.axes[0].edges)
            and np.array_equal(counts.axes[1].edges, exposure.axes[1].edges)
            and (exposure.ndim == 2 or np.array_equal(counts.axes[2].edges, exposure.axes[2].edges))
        ):
            raise ValueError("counts and exposure must have matching axis edges")

        e_axis = counts.axes[2]
        return (
            self._convolve_3d(counts, e_axis, kernel_radius),
            self._convolve_3d(exposure, e_axis, kernel_radius),
        )

    @staticmethod
    def _differential_rate(counts: hist.Hist, exposure: hist.Hist) -> hist.Hist:
        x_edges = counts.axes[0].edges
        y_edges = counts.axes[1].edges
        e_edges = counts.axes[2].edges

        energy_width = np.diff(u.Quantity(e_edges, u.TeV).to_value(u.MeV))
        lon_width = np.diff(np.deg2rad(x_edges))
        lat_width = np.diff(np.deg2rad(y_edges))
        solid_angle = lon_width[:, None] * lat_width[None, :]

        exposure_3d = exposure.values()[:, :, None] if exposure.ndim == 2 else exposure.values()
        denominator = exposure_3d * energy_width[None, None, :] * solid_angle[:, :, None]

        rate_values = np.full(counts.values().shape, np.nan, dtype=float)
        np.divide(counts.values(), denominator, out=rate_values, where=denominator > 0)

        rate = hist.Hist(*counts.axes)
        rate.values()[:] = rate_values
        return rate

    @staticmethod
    def _to_background_3d(
        rate: hist.Hist,
        alignment: FoVAlignment,
        filename: str | Path | None,
    ) -> Background3D:
        values = rate.values()
        if not np.all(np.isfinite(values)):
            raise ValueError("The background rate contains undefined values for pixels with zero exposure")
        if rate.axes[2].edges[0] <= 0:
            raise ValueError("Energy bin edges must be positive")
        background = Background3D(
            axes=[
                MapAxis.from_edges(rate.axes[2].edges, name="energy", unit="TeV", interp="log"),
                MapAxis.from_edges(rate.axes[0].edges, name="fov_lon", unit="deg"),
                MapAxis.from_edges(rate.axes[1].edges, name="fov_lat", unit="deg"),
            ],
            data=np.moveaxis(values, 2, 0),
            unit="s-1 MeV-1 sr-1",
            fov_alignment=alignment,
        )
        if filename is not None:
            background.write(filename, overwrite=True)
        return background

    def make_counts_map(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None = None,
        kernel_radius: Iterable[float] | u.Quantity | None = None,
        *,
        downsample: bool = True,
    ) -> hist.Hist:
        """Build the masked and accumulated ``(x, y, energy)`` counts histogram.

        Accumulates ``counts`` on the input ``CameraImage`` grid. When
        ``downsample=True`` (default) and ``downsampling > 1``, sum-rebins into
        ``coarse_counts``. Pass ``downsample=False`` to inspect the native
        accumulated ``counts`` before downsampling. Optional circular top-hat
        ``kernel_radius`` smoothing is applied after downsampling.
        """
        if not downsample and kernel_radius is not None:
            raise ValueError("kernel_radius cannot be combined with downsample=False")
        counts = self._accumulate_counts(images, exclusion_region)
        if not downsample:
            return counts
        if self.downsampling > 1:
            coarse_counts = self._downsample_counts(counts)
            if kernel_radius is not None:
                coarse_counts = self._convolve_3d(coarse_counts, coarse_counts.axes[2], kernel_radius)
            return coarse_counts
        if kernel_radius is not None:
            counts = self._convolve_3d(counts, counts.axes[2], kernel_radius)
        return counts

    def make_exposure_map(
        self,
        images: CameraImage | Iterable[CameraImage],
        exclusion_region: RegionGeom | None = None,
        kernel_radius: Iterable[float] | u.Quantity | None = None,
        *,
        downsample: bool = True,
    ) -> hist.Hist:
        """Build the masked and accumulated exposure histogram in seconds.

        Accumulates ``exposure`` on the input ``CameraImage`` grid. When
        ``downsample=True`` (default) and ``downsampling > 1``, mean-rebins
        into a 2D ``(x, y)`` ``coarse_exposure``. Pass ``downsample=False`` to
        inspect the native accumulated ``exposure``. When ``kernel_radius`` is
        supplied, circular top-hat smoothing is applied after downsampling and
        a 3D ``(x, y, energy)`` histogram is returned.
        """
        if not downsample and kernel_radius is not None:
            raise ValueError("kernel_radius cannot be combined with downsample=False")
        exposure, e_axis = self._accumulate_exposure(images, exclusion_region)
        if not downsample:
            return exposure
        if self.downsampling > 1:
            coarse_exposure = self._downsample_exposure(exposure)
            if kernel_radius is not None:
                coarse_exposure = self._convolve_3d(coarse_exposure, e_axis, kernel_radius)
            return coarse_exposure
        if kernel_radius is not None:
            exposure = self._convolve_3d(exposure, e_axis, kernel_radius)
        return exposure

    def _make_background_map(
        self,
        images: CameraImage | Iterable[CameraImage],
        required_frame: type[FoVICRSFrame] | type[FoVAltAzFrame],
        alignment: FoVAlignment,
        exclusion_region: RegionGeom | None,
        kernel_radius: Iterable[float] | u.Quantity | None,
        filename: str | Path | None,
    ) -> Background3D:
        counts, exposure = self._accumulate(images, exclusion_region, required_frame=required_frame)
        if self.downsampling > 1:
            coarse_counts = self._downsample_counts(counts)
            coarse_exposure = self._downsample_exposure(exposure)
            if kernel_radius is not None:
                coarse_counts, coarse_exposure = self.convolve(coarse_counts, coarse_exposure, kernel_radius)
            rate = self._differential_rate(coarse_counts, coarse_exposure)
        else:
            if kernel_radius is not None:
                counts, exposure = self.convolve(counts, exposure, kernel_radius)
            rate = self._differential_rate(counts, exposure)
        return self._to_background_3d(rate, alignment, filename)

    def make_icrs_background_map(
        self,
        images: Iterable[CameraImage],
        exclusion_region: RegionGeom | None = None,
        kernel_radius: Iterable[float] | u.Quantity | None = None,
        filename: str | Path | None = None,
    ) -> Background3D:
        """Combine ICRS camera images into a differential background-rate IRF."""
        return self._make_background_map(
            images, FoVICRSFrame, FoVAlignment.RADEC, exclusion_region, kernel_radius, filename
        )

    def make_altaz_background_map(
        self,
        images: Iterable[CameraImage],
        exclusion_region: RegionGeom | None = None,
        kernel_radius: Iterable[float] | u.Quantity | None = None,
        filename: str | Path | None = None,
    ) -> Background3D:
        """Combine AltAz camera images into a differential background-rate IRF."""
        return self._make_background_map(
            images, FoVAltAzFrame, FoVAlignment.ALTAZ, exclusion_region, kernel_radius, filename
        )
