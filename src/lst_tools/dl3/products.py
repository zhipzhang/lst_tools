"""Discovery and explicit configuration-based selection of LST DL3 products."""

import warnings
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

from ..helper import find_lst_data_path


@dataclass(frozen=True)
class DL3Product:
    """A discovered DL3 file together with metadata encoded in its path."""

    date: int
    run_number: int
    analysis_type: str
    background_type: str
    cut_config: str
    path: Path
    processing_version: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.analysis_type, self.background_type, self.cut_config


@dataclass(frozen=True)
class DL3Request:
    """One DL3 product variant requested by the analysis configuration."""

    name: str
    analysis_type: str
    background_type: str
    cut_config: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.analysis_type, self.background_type, self.cut_config


def parse_dl3_path(path: str | Path, date: int, run_number: int) -> DL3Product:
    """Parse metadata from ``.../<analysis>/<background>/<cuts>/irf_interp/file``."""
    path = Path(path)
    irf_directory = path.parent
    cut_directory = irf_directory.parent
    background_directory = cut_directory.parent
    analysis_directory = background_directory.parent

    if irf_directory.name != "irf_interp":
        raise ValueError(f"DL3 file is not inside an irf_interp directory: {path}")
    if not cut_directory.name or not background_directory.name or not analysis_directory.name:
        raise ValueError(f"DL3 path does not contain the expected product hierarchy: {path}")

    processing_version = next(
        (directory.parent.name for directory in path.parents if directory.name == "std"),
        None,
    )

    return DL3Product(
        date=int(date),
        run_number=int(run_number),
        analysis_type=analysis_directory.name,
        background_type=background_directory.name,
        cut_config=cut_directory.name,
        path=path,
        processing_version=processing_version,
    )


def discover_lst_dl3_products(
    date: int,
    run_number: int,
    path_finder=find_lst_data_path,
) -> list[DL3Product]:
    """Discover and parse all supported DL3 products available for one run."""
    raw_paths = path_finder(date, run_number, level="dl3") or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]

    products = []
    for raw_path in sorted(raw_paths, key=str):
        try:
            products.append(parse_dl3_path(raw_path, date, run_number))
        except ValueError as error:
            warnings.warn(str(error), stacklevel=2)

    return products


def _validate_path_component(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"DL3 {field_name} must be a non-empty string")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"DL3 {field_name} must be a single safe path component: {value!r}")
    return value


def load_dl3_requests(dl3_config: dict) -> tuple[DL3Request, ...]:
    """Build the configured product × cut-configuration cross-product."""
    cut_configs = dl3_config.get("cut_configs", [])
    product_configs = dl3_config.get("products", [])

    if not isinstance(cut_configs, list):
        raise ValueError("DL3 cut_configs must be a TOML array")
    if not isinstance(product_configs, list):
        raise ValueError("DL3 products must be an array of TOML tables")
    if not cut_configs:
        raise ValueError("DL3 is enabled but no cut_configs were configured")
    if not product_configs:
        raise ValueError("DL3 is enabled but no products were configured")

    normalized_cuts = tuple(_validate_path_component(value, "cut_config") for value in cut_configs)
    if len(set(normalized_cuts)) != len(normalized_cuts):
        raise ValueError("DL3 cut_configs must be unique")

    requests = []
    product_names = set()
    product_keys = set()
    for product in product_configs:
        if not isinstance(product, dict):
            raise ValueError("Each DL3 product must be a TOML table")
        missing_fields = {"name", "analysis_type", "background_type"}.difference(product)
        if missing_fields:
            raise ValueError(f"DL3 product is missing fields: {sorted(missing_fields)}")

        name = _validate_path_component(product["name"], "product name")
        analysis_type = _validate_path_component(product["analysis_type"], "analysis_type")
        background_type = _validate_path_component(product["background_type"], "background_type")

        if name in product_names:
            raise ValueError(f"Duplicate DL3 product name: {name}")
        product_key = analysis_type, background_type
        if product_key in product_keys:
            raise ValueError(f"Duplicate DL3 product definition: {product_key}")
        product_names.add(name)
        product_keys.add(product_key)

        requests.extend(
            DL3Request(
                name=name,
                analysis_type=analysis_type,
                background_type=background_type,
                cut_config=cut_config,
            )
            for cut_config in normalized_cuts
        )

    return tuple(requests)


def select_configured_dl3_products(
    available: list[DL3Product],
    requests: tuple[DL3Request, ...],
) -> dict[DL3Request, DL3Product]:
    """Select exact configured products from the newest processing version."""
    requested_keys = {request.key for request in requests}
    products_by_key = {}
    for product in available:
        if product.key not in requested_keys:
            continue
        products_by_key.setdefault(product.key, []).append(product)

    available_by_key = {
        key: _select_newest_processing_version(key, products)
        for key, products in products_by_key.items()
    }

    return {request: available_by_key[request.key] for request in requests if request.key in available_by_key}


def _select_newest_processing_version(
    key: tuple[str, str, str],
    products: list[DL3Product],
) -> DL3Product:
    if len(products) == 1:
        return products[0]

    versioned_products = []
    for product in products:
        if product.processing_version is None:
            raise RuntimeError(
                f"Cannot choose the newest DL3 file for {key}; processing version is missing from {product.path}"
            )
        try:
            version = Version(product.processing_version.removeprefix("v"))
        except InvalidVersion as error:
            raise RuntimeError(
                f"Cannot parse DL3 processing version {product.processing_version!r} from {product.path}"
            ) from error
        versioned_products.append((version, product))

    newest_version = max(version for version, _ in versioned_products)
    newest_products = [product for version, product in versioned_products if version == newest_version]
    if len(newest_products) > 1:
        paths = ", ".join(str(product.path) for product in newest_products)
        raise RuntimeError(f"Multiple DL3 files match {key} at newest processing version {newest_version}: {paths}")

    return newest_products[0]
