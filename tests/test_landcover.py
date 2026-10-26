"""指数、精度、变化方向、预处理和整条管线。"""

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import Affine

from landcover.change import bearing_degrees, change_codes, sector_index, summarize_change
from landcover.classify import GaussianMLC
from landcover.config import load_config
from landcover.demo_scene import land_cover
from landcover.indices import mndwi, ndbi, ndvi, ndwi_swir
from landcover.legend import CLASSES, FEATURE_NAMES
from landcover.metrics import cohen_kappa, evaluate
from landcover.pipeline import run
from landcover.rasters import read_onto, write_raster


def test_indices_match_definitions():
    nir = np.array([0.5, 0.0], dtype=np.float32)
    red = np.array([0.25, 0.0], dtype=np.float32)
    result = ndvi(nir, red)
    assert result[0] == pytest.approx(1 / 3, rel=1e-5)
    assert np.isnan(result[1])
    assert ndbi(np.array([0.3]), np.array([0.1]))[0] == pytest.approx(0.5)
    assert mndwi(np.array([0.2]), np.array([0.05]))[0] == pytest.approx(0.6)
    assert ndwi_swir(np.array([0.4]), np.array([0.1]))[0] == pytest.approx(0.6)


def test_kappa_and_f1():
    matrix_perfect = np.array([[50, 0], [0, 50]])
    assert cohen_kappa(matrix_perfect) == pytest.approx(1.0)
    report = evaluate(np.array([1, 1, 1, 2, 2]), np.array([1, 1, 2, 2, 2]), [1, 2])
    assert report["oa"] == pytest.approx(0.8)
    assert report["per_class"][0]["f1"] == pytest.approx(0.8)


def test_change_area_and_eastward_sector():
    t1 = np.array([[3, 3, 1], [2, 4, 4]], dtype=np.uint8)
    t2 = np.array([[3, 1, 3], [2, 4, 3]], dtype=np.uint8)
    codes = change_codes(t1, t2)
    assert codes.tolist() == [[2, 4, 3], [1, 1, 3]]
    transform = Affine(10, 0, 0, 0, -10, 100)
    summary = summarize_change(t1, t2, transform, years=8)
    assert summary["expansion_km2"] == pytest.approx(2 * 100 / 1_000_000)
    assert summary["loss_km2"] == pytest.approx(100 / 1_000_000)
    assert bearing_degrees(np.array([100.0]), np.array([0.0]))[0] == pytest.approx(90)
    assert sector_index(np.array([90.0]))[0] == 2


def test_mlc_separates_two_blobs():
    rng = np.random.default_rng(0)
    first = rng.normal(0, 0.1, size=(40, 2))
    second = rng.normal(3, 0.1, size=(40, 2))
    features = np.vstack([first, second])
    labels = np.array([1] * 40 + [2] * 40)
    model = GaussianMLC().fit(features, labels)
    predicted = model.predict(features)
    assert (predicted == labels).mean() > 0.95


def test_demo_labels_cover_five_classes_and_expand():
    t1 = land_cover(80, 80, "t1")
    t2 = land_cover(80, 80, "t2")
    assert set(np.unique(t1)).issuperset({1, 2, 3, 4, 5})
    assert (t2 == 3).sum() > (t1 == 3).sum()


def test_resample_20m_band_onto_10m(tmp_path: Path):
    coarse = np.array([[1000, 3000], [5000, 7000]], dtype=np.uint16)
    transform = Affine(20, 0, 500000, 0, -20, 2500000)
    write_raster(tmp_path / "B11.tif", coarse, transform, "EPSG:32649", "uint16", 0)
    target = Affine(10, 0, 500000, 0, -10, 2500000)
    fine = read_onto(tmp_path / "B11.tif", target, "EPSG:32649", 4, 4, __import__("rasterio").enums.Resampling.bilinear, True)
    assert fine.shape == (4, 4)
    assert np.nanmean(fine) == pytest.approx(0.4, abs=0.05)


def test_majority_filter_removes_speckle():
    from landcover.classify import majority_filter

    labels = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.uint8)
    valid = np.ones_like(labels, dtype=bool)
    assert majority_filter(labels, valid)[1, 1] == 1


def test_feature_list_has_no_reference_band():
    assert "reference" not in FEATURE_NAMES
    assert {"NDVI", "NDBI", "MNDWI", "NDWI_SWIR"}.issubset(FEATURE_NAMES)


def test_qml_matches_legend():
    text = Path("qgis/styles/landcover.qml").read_text(encoding="utf-8")
    for spec in CLASSES:
        assert f'value="{spec.id}"' in text
        assert spec.color in text
        assert spec.zh in text


def test_stac_query_uses_cloud_threshold():
    from landcover.stac import search_body

    body = search_body({"west": 1, "south": 2, "east": 3, "north": 4}, "2016-09-01", "2016-11-30", 10)
    assert body["collections"] == ["sentinel-2-l2a"]
    assert body["query"]["eo:cloud_cover"]["lt"] == 10
    assert body["bbox"] == [1, 2, 3, 4]
    assert body["datetime"] == "2016-09-01T00:00:00Z/2016-11-30T23:59:59Z"


def test_pipeline_end_to_end():
    metrics = run(load_config())
    assert metrics["random_forest"]["oa"] >= 0.85
    assert metrics["random_forest"]["kappa"] > 0.8
    assert metrics["change"]["expansion_km2"] > metrics["change"]["loss_km2"] > 0
    assert metrics["alignment"]["offset_m"] == pytest.approx(0.0, abs=1e-6)
    assert Path("outputs/figures/overview.png").exists()
    assert Path("outputs/maps/class_t1.tif").exists()
    assert Path("outputs/tables/metrics.json").exists()
