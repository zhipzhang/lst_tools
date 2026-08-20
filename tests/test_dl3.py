import pytest

from lst_ulities.dl3 import (
    DL3Product,
    discover_lst_dl3_products,
    load_dl3_requests,
    parse_dl3_path,
    select_configured_dl3_products,
)


DL3_CONFIG = {
    "cut_configs": ["gheff0.7_thetacont0.7", "gheff0.9_thetacont0.7"],
    "products": [
        {
            "name": "point",
            "analysis_type": "point",
            "background_type": "ring-wobble",
        },
        {
            "name": "full_diffuse",
            "analysis_type": "full-enclosure",
            "background_type": "diffuse",
        },
    ],
}


def product_path(
    tmp_path,
    analysis_type,
    background_type,
    cut_config,
    run_number=1,
    processing_version="0.11.4",
):
    return (
        tmp_path
        / "v0.11"
        / "tailcut84"
        / "nsb_tuning_0.22"
        / processing_version
        / "std"
        / "20250212_v0.10.17_allsky_interp_dl2_irfs_nsb_0.22"
        / "src_indep"
        / analysis_type
        / background_type
        / cut_config
        / "irf_interp"
        / f"dl3_LST-1.Run{run_number:05d}.fits"
    )


def test_parse_dl3_path_extracts_product_metadata(tmp_path):
    path = product_path(
        tmp_path,
        "full-enclosure",
        "diffuse",
        "gheff0.9_thetacont0.7",
        run_number=42,
    )

    product = parse_dl3_path(path, date=20240101, run_number=42)

    assert product.analysis_type == "full-enclosure"
    assert product.background_type == "diffuse"
    assert product.cut_config == "gheff0.9_thetacont0.7"
    assert product.processing_version == "0.11.4"
    assert product.path == path


def test_discover_dl3_products_uses_existing_raw_finder(tmp_path):
    paths = [
        product_path(tmp_path, "point", "ring-wobble", "gheff0.7_thetacont0.7"),
        product_path(tmp_path, "full-enclosure", "diffuse", "gheff0.9_thetacont0.7"),
    ]

    def find_paths(date, run_number, level):
        assert (date, run_number, level) == (20240101, 1, "dl3")
        return [str(path) for path in reversed(paths)]

    products = discover_lst_dl3_products(20240101, 1, path_finder=find_paths)

    assert [product.path for product in products] == sorted(paths, key=str)


def test_load_dl3_requests_builds_product_cut_cross_product():
    requests = load_dl3_requests(DL3_CONFIG)

    assert len(requests) == 4
    assert {request.key for request in requests} == {
        ("point", "ring-wobble", "gheff0.7_thetacont0.7"),
        ("point", "ring-wobble", "gheff0.9_thetacont0.7"),
        ("full-enclosure", "diffuse", "gheff0.7_thetacont0.7"),
        ("full-enclosure", "diffuse", "gheff0.9_thetacont0.7"),
    }


def test_select_dl3_products_ignores_unconfigured_product(tmp_path):
    requests = load_dl3_requests(DL3_CONFIG)
    available = [
        parse_dl3_path(
            product_path(tmp_path, "point", "ring-wobble", "gheff0.7_thetacont0.7"),
            20240101,
            1,
        ),
        parse_dl3_path(
            product_path(tmp_path, "point", "ring-wobble", "gheff0.5_thetacont0.7"),
            20240101,
            1,
        ),
    ]

    selected = select_configured_dl3_products(available, requests)

    assert len(selected) == 1
    assert next(iter(selected.values())).cut_config == "gheff0.7_thetacont0.7"


def test_select_dl3_products_uses_newest_processing_version(tmp_path):
    request = load_dl3_requests(DL3_CONFIG)[0]
    older = parse_dl3_path(
        product_path(
            tmp_path,
            request.analysis_type,
            request.background_type,
            request.cut_config,
            run_number=22987,
            processing_version="0.10.18",
        ),
        20251120,
        22987,
    )
    newer = parse_dl3_path(
        product_path(
            tmp_path,
            request.analysis_type,
            request.background_type,
            request.cut_config,
            run_number=22987,
            processing_version="0.11.4",
        ),
        20251120,
        22987,
    )

    selected = select_configured_dl3_products([older, newer], (request,))

    assert selected[request] == newer


def test_select_dl3_products_rejects_duplicate_newest_version(tmp_path):
    request = load_dl3_requests(DL3_CONFIG)[0]
    first = DL3Product(20240101, 1, *request.key, tmp_path / "first.fits", "0.11.4")
    second = DL3Product(20240101, 1, *request.key, tmp_path / "second.fits", "v0.11.4")

    with pytest.raises(RuntimeError, match="at newest processing version"):
        select_configured_dl3_products([first, second], (request,))


def test_select_dl3_products_ignores_duplicate_unconfigured_products(tmp_path):
    request = load_dl3_requests(DL3_CONFIG)[0]
    unconfigured_key = ("full-enclosure", "ring-wobble", "unconfigured-cut")
    first = DL3Product(20240101, 1, *unconfigured_key, tmp_path / "first.fits", "0.11.4")
    second = DL3Product(20240101, 1, *unconfigured_key, tmp_path / "second.fits", "0.11.4")

    assert select_configured_dl3_products([first, second], (request,)) == {}


@pytest.mark.parametrize(
    "change",
    [
        {"cut_configs": []},
        {"products": []},
        {"cut_configs": ["gheff0.7", "gheff0.7"]},
    ],
)
def test_load_dl3_requests_rejects_invalid_config(change):
    config = {**DL3_CONFIG, **change}

    with pytest.raises(ValueError):
        load_dl3_requests(config)
