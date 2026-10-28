"""Combine camera observations into Gammapy background models."""

from collections.abc import Iterable, Sequence
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.irf import Background3D, FoVAlignment
from gammapy.maps import MapAxis, RegionGeom
from gammapy.utils.coordinates import FoVICRSFrame
from scipy.ndimage import convolve as ndi_convolve

from .camera import CameraImage


class BkgMaker:
    """Own masking, exposure, combination, and background-rate construction."""

    def __init__(self, exposure_oversampling: int = 4) -> None:
        if not isinstance(exposure_oversampling, (int, np.integer)) or exposure_oversampling < 1:
            raise ValueError("exposure_oversampling must be a positive integer")
        self.exposure_oversampling = int(exposure_oversampling)

    @staticmethod
    def _validate_images(images: Sequence[CameraImage]) -> list[CameraImage]:
        images = list(images)
        if not images:
            raise ValueError("At least one CameraImage is required")
        if any(not isinstance(image, CameraImage) for image in images):
            raise TypeError("images must contain only CameraImage objects")

        reference = images[0]
        for image in images:
            if not isinstance(image.frame.fov_frame, FoVICRSFrame):
                raise ValueError("make_icrs_background_map requires FoVICRSFrame images")
            if not all(
                np.array_equal(getattr(image, axis), getattr(reference, axis))
                for axis in ("x_edges", "y_edges", "e_edges")
            ):
                raise ValueError("All CameraImage objects must have identical axes")
        return images

    @staticmethod
    def _refine_edges(edges: np.ndarray, factor: int) -> np.ndarray:
        fractions = np.arange(factor) / factor
        refined = edges[:-1, None] + np.diff(edges)[:, None] * fractions
        return np.concatenate([refined.ravel(), [edges[-1]]])

    @staticmethod
    def _centers(edges: np.ndarray) -> np.ndarray:
        return 0.5 * (edges[:-1] + edges[1:])

    @staticmethod
    def _exclusion_mask(
        image: CameraImage,
        x: np.ndarray,
        y: np.ndarray,
        exclusion_region: RegionGeom | None,
    ) -> np.ndarray:
        if exclusion_region is None:
            return np.zeros(x.shape, dtype=bool)
        coordinates = SkyCoord(x * u.deg, y * u.deg, frame=image.frame.fov_frame)
        return np.asarray(exclusion_region.contains(coordinates), dtype=bool)

    def _masked_counts(self, image: CameraImage, exclusion_region: RegionGeom | None) -> np.ndarray:
        x, y = np.meshgrid(self._centers(image.x_edges), self._centers(image.y_edges), indexing="ij")
        excluded = self._exclusion_mask(image, x, y, exclusion_region)
        counts = image.histogram.values().copy()
        counts[excluded, :] = 0
        return counts

    def _exposure_map(self, image: CameraImage, exclusion_region: RegionGeom | None) -> np.ndarray:
        factor = self.exposure_oversampling
        x_edges = self._refine_edges(image.x_edges, factor)
        y_edges = self._refine_edges(image.y_edges, factor)
        x, y = np.meshgrid(self._centers(x_edges), self._centers(y_edges), indexing="ij")
        excluded = self._exclusion_mask(image, x, y, exclusion_region)
        return np.where(excluded, 0.0, image.live_time.to_value(u.s))

    def _accumulate(
        self,
        images: Sequence[CameraImage],
        exclusion_region: RegionGeom | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = np.zeros_like(images[0].histogram.values(), dtype=float)
        exposure_map = None
        for image in images:
            counts += self._masked_counts(image, exclusion_region)
            image_exposure = self._exposure_map(image, exclusion_region)
            exposure_map = image_exposure if exposure_map is None else exposure_map + image_exposure
        return counts, exposure_map

    def _coarse_exposure(self, exposure_map: np.ndarray, counts: np.ndarray) -> np.ndarray:
        factor = self.exposure_oversampling
        n_x, n_y = exposure_map.shape
        coarse = exposure_map.reshape(n_x // factor, factor, n_y // factor, factor).mean(axis=(1, 3))
        return np.broadcast_to(coarse[:, :, None], counts.shape).copy()

    @staticmethod
    def _make_kernel(radius: float, pixel_size: tuple[float, float]) -> np.ndarray:
        dx, dy = pixel_size
        nx = int(np.ceil(radius / dx))
        ny = int(np.ceil(radius / dy))
        x = np.arange(-nx, nx + 1) * dx
        y = np.arange(-ny, ny + 1) * dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return (xx**2 + yy**2 <= radius**2).astype(float)

    def convolve(
        self,
        counts: np.ndarray,
        exposure: np.ndarray,
        pixel_size: tuple[float, float],
        kernel_radius: Iterable[float] | u.Quantity,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convolve counts and exposure with identical circular top-hat kernels."""
        counts = np.asarray(counts)
        exposure = np.asarray(exposure)
        if counts.ndim != 3 or counts.shape != exposure.shape:
            raise ValueError("counts and exposure must have identical (x, y, energy) shapes")
        if len(pixel_size) != 2 or not all(np.isfinite(pixel_size)) or min(pixel_size) <= 0:
            raise ValueError("pixel_size must contain two positive finite values")

        radii = np.atleast_1d(u.Quantity(kernel_radius, u.deg).to_value(u.deg))
        n_energy = counts.shape[2]
        if radii.size == 1:
            radii = np.repeat(radii, n_energy)
        if radii.size != n_energy:
            raise ValueError("kernel_radius must be scalar or contain one radius per energy bin")
        if not np.all(np.isfinite(radii)) or np.any(radii < 0):
            raise ValueError("kernel_radius must be finite and nonnegative")

        convolved_counts = np.empty_like(counts, dtype=float)
        convolved_exposure = np.empty_like(exposure, dtype=float)
        for index, radius in enumerate(radii):
            kernel = self._make_kernel(radius, pixel_size)
            convolved_counts[:, :, index] = ndi_convolve(counts[:, :, index], kernel, mode="constant", cval=0.0)
            convolved_exposure[:, :, index] = ndi_convolve(
                exposure[:, :, index], kernel, mode="constant", cval=0.0
            )
        return convolved_counts, convolved_exposure

    @staticmethod
    def _differential_rate(counts: np.ndarray, exposure: np.ndarray, image: CameraImage) -> np.ndarray:
        energy_edges = u.Quantity(image.e_edges, u.TeV).to_value(u.MeV)
        energy_width = np.diff(energy_edges)
        denominator = exposure * energy_width[None, None, :] * image.pixel_solid_angle[:, :, None]
        rate = np.full_like(counts, np.nan, dtype=float)
        np.divide(counts, denominator, out=rate, where=denominator > 0)
        return rate

    @staticmethod
    def _to_background_3d(
        rate: np.ndarray,
        image: CameraImage,
        filename: str | Path | None,
    ) -> Background3D:
        if not np.all(np.isfinite(rate)):
            raise ValueError("The background rate contains undefined values for pixels with zero exposure")
        background = Background3D(
            axes=[
                MapAxis.from_edges(image.e_edges, name="energy", unit="TeV"),
                MapAxis.from_edges(image.x_edges, name="fov_lon", unit="deg"),
                MapAxis.from_edges(image.y_edges, name="fov_lat", unit="deg"),
            ],
            data=np.moveaxis(rate, 2, 0),
            unit="s-1 MeV-1 sr-1",
            fov_alignment=FoVAlignment.RADEC,
        )
        if filename is not None:
            background.write(filename, overwrite=True)
        return background

    def make_icrs_background_map(
        self,
        images: Sequence[CameraImage],
        exclusion_region: RegionGeom | None = None,
        kernel_radius: Iterable[float] | u.Quantity | None = None,
        filename: str | Path | None = None,
    ) -> Background3D:
        """Combine ICRS camera images into a differential background-rate IRF."""
        images = self._validate_images(images)
        reference = images[0]
        counts, exposure_map = self._accumulate(images, exclusion_region)
        exposure = self._coarse_exposure(exposure_map, counts)
        if kernel_radius is not None:
            counts, exposure = self.convolve(counts, exposure, reference.pixel_size, kernel_radius)
        rate = self._differential_rate(counts, exposure, reference)
        return self._to_background_3d(rate, reference, filename)
