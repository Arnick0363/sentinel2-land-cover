"""读写 GeoTIFF，以及反射率与 SCL 的网格对齐。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


def write_raster(
    path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: str,
    dtype: str,
    nodata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 2,
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.asarray(array), 1)
        if dtype in {"uint8", "uint16"} and nodata == 0:
            # 调色板只用于分类图；反射率不写调色板。调用方按需覆盖。
            pass


def write_paletted(path: Path, array: np.ndarray, transform: Affine, crs: str, colormap: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(np.uint8), 1)
        dataset.write_colormap(1, colormap)


def read_profile(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs,
            "transform": dataset.transform,
            "nodata": dataset.nodata,
            "dtype": dataset.dtypes[0],
        }


def read_onto(
    path: Path,
    transform: Affine,
    crs,
    width: int,
    height: int,
    resampling: Resampling,
    scale_reflectance: bool,
) -> np.ndarray:
    """把任意分辨率的单波段读到目标 10 m 网格上。"""
    with rasterio.open(path) as dataset:
        source = dataset.read(1)
        src_nodata = dataset.nodata
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=source.astype(np.float32),
            destination=destination,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    if scale_reflectance:
        finite = destination[np.isfinite(destination)]
        if finite.size and np.nanpercentile(finite, 99) > 2:
            destination = destination / 10000.0
        # 反射率产品用 0 作 nodata 时，重投影后仍可能留下 0。
        destination[destination == 0] = np.nan
    return destination.astype(np.float32)


def alignment_report(left: dict, right: dict) -> dict:
    dx = float(right["transform"].c - left["transform"].c)
    dy = float(right["transform"].f - left["transform"].f)
    same_shape = left["width"] == right["width"] and left["height"] == right["height"]
    same_transform = left["transform"] == right["transform"] and str(left["crs"]) == str(right["crs"])
    return {
        "same_grid": bool(same_shape and same_transform),
        "same_shape": bool(same_shape),
        "same_crs": str(left["crs"]) == str(right["crs"]),
        "origin_offset_x_m": dx,
        "origin_offset_y_m": dy,
        "offset_m": float(np.hypot(dx, dy)),
        "t1_size": [int(left["height"]), int(left["width"])],
        "t2_size": [int(right["height"]), int(right["width"])],
    }
