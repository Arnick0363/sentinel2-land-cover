"""云掩膜、重采样、波段堆叠与指数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.enums import Resampling

from landcover.indices import mndwi, ndbi, ndvi, ndwi_swir
from landcover.legend import BANDS_20M, FEATURE_NAMES, REFLECTANCE_BANDS, SCL_INVALID
from landcover.rasters import alignment_report, read_onto, read_profile


def load_scene(folder: Path, target: dict | None = None) -> dict:
    anchor = read_profile(folder / "B02.tif")
    grid = target or anchor
    transform = grid["transform"]
    crs = grid["crs"]
    width = grid["width"]
    height = grid["height"]
    bands = {}
    for name in REFLECTANCE_BANDS:
        bands[name] = read_onto(
            folder / f"{name}.tif",
            transform,
            crs,
            width,
            height,
            Resampling.bilinear,
            scale_reflectance=True,
        )
    scl = read_onto(
        folder / "SCL.tif",
        transform,
        crs,
        width,
        height,
        Resampling.nearest,
        scale_reflectance=False,
    )
    scl = np.where(np.isfinite(scl), np.rint(scl), 0).astype(np.uint8)
    reference_path = folder / "reference.tif"
    reference = None
    if reference_path.exists():
        reference_raw = read_onto(
            reference_path,
            transform,
            crs,
            width,
            height,
            Resampling.nearest,
            scale_reflectance=False,
        )
        reference = np.where(np.isfinite(reference_raw), np.rint(reference_raw), 0).astype(np.uint8)
    valid = ~np.isin(scl, SCL_INVALID)
    for name in REFLECTANCE_BANDS:
        valid &= np.isfinite(bands[name])
    indices = {
        "NDVI": ndvi(bands["B08"], bands["B04"]),
        "NDBI": ndbi(bands["B11"], bands["B08"]),
        "MNDWI": mndwi(bands["B03"], bands["B11"]),
        "NDWI_SWIR": ndwi_swir(bands["B08"], bands["B12"]),
    }
    planes = [bands[name] for name in REFLECTANCE_BANDS] + [indices[name] for name in FEATURE_NAMES if name not in REFLECTANCE_BANDS]
    stack = np.stack(planes, axis=-1).astype(np.float32)
    valid &= np.isfinite(stack).all(axis=-1)
    return {
        "bands": bands,
        "indices": indices,
        "stack": stack,
        "feature_names": list(FEATURE_NAMES),
        "valid": valid,
        "scl": scl,
        "reference": reference,
        "profile": {
            "width": width,
            "height": height,
            "crs": crs,
            "transform": transform,
            "nodata": anchor["nodata"],
            "dtype": anchor["dtype"],
        },
        "native_profile": anchor,
        "cloud_fraction": float((~valid).mean()),
        "bands_20m_expected": list(BANDS_20M),
    }


def check_dates(scene_t1: dict, scene_t2: dict) -> dict:
    return alignment_report(scene_t1["profile"], scene_t2["profile"])
