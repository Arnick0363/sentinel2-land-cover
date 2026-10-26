"""构造珠海横琴外包矩形上的双时相教学示例景。

像元值是按 Sentinel-2 L2A 地表反射率量级合成的，带 SCL 云掩膜和参考类别。
空间格局只用来把管线跑通，不是 2016–2024 年横琴的实测扩张。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin

from landcover.legend import BANDS_10M, BANDS_20M, CLASSES
from landcover.rasters import write_raster

# 波段顺序：B02, B03, B04, B08, B11, B12。数值为地表反射率。
_SIGNATURES = {
    1: np.array([0.058, 0.078, 0.046, 0.021, 0.012, 0.010], dtype=np.float32),
    2: np.array([0.031, 0.056, 0.028, 0.470, 0.120, 0.058], dtype=np.float32),
    3: np.array([0.118, 0.126, 0.138, 0.162, 0.228, 0.198], dtype=np.float32),
    4: np.array([0.162, 0.174, 0.198, 0.224, 0.328, 0.286], dtype=np.float32),
    5: np.array([0.042, 0.082, 0.052, 0.335, 0.188, 0.108], dtype=np.float32),
}
_SCL_PRIORITY = np.array([50, 60, 5, 70, 10, 10, 10, 5, 90, 100, 80, 40], dtype=np.int16)


def grid_from_aoi(aoi: dict) -> dict:
    """把 WGS84 外包矩形落到 UTM 上，并让宽高为偶数，方便 20 m 聚合。"""
    resolution = float(aoi["resolution_m"])
    transformer = Transformer.from_crs("EPSG:4326", aoi["crs"], always_xy=True)
    west, south = transformer.transform(float(aoi["west"]), float(aoi["south"]))
    east, north = transformer.transform(float(aoi["east"]), float(aoi["north"]))
    west = float(np.floor(west / resolution) * resolution)
    south = float(np.floor(south / resolution) * resolution)
    east = float(np.ceil(east / resolution) * resolution)
    north = float(np.ceil(north / resolution) * resolution)
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    width -= width % 2
    height -= height % 2
    if width < 20 or height < 20:
        raise ValueError("研究区在 10 m 网格上过小")
    transform = from_origin(west, north, resolution, resolution)
    return {
        "crs": aoi["crs"],
        "transform": transform,
        "width": width,
        "height": height,
        "resolution": resolution,
        "bounds_utm": {
            "west": west,
            "south": float(north - height * resolution),
            "east": float(west + width * resolution),
            "north": north,
        },
    }


def _norm_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    return np.broadcast_to(y, (height, width)).copy(), np.broadcast_to(x, (height, width)).copy()


def _ellipse(x: np.ndarray, y: np.ndarray, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def land_cover(height: int, width: int, period: str) -> np.ndarray:
    """生成参考类别。边界加了固定起伏，避免整幅图只剩正圆和矩形。"""
    y, x = _norm_grids(height, width)
    labels = np.full((height, width), 2, dtype=np.uint8)
    coast = 0.785 + 0.015 * np.sin(2 * np.pi * 1.7 * x) + 0.008 * np.sin(2 * np.pi * 3.1 * x)
    water = y > coast
    channel = np.abs(x - (0.22 + 0.16 * y + 0.01 * np.sin(2 * np.pi * 2.2 * y))) < 0.007
    water = water | (channel & (y < coast))

    crop_west = 0.44 + 0.035 * np.sin(2 * np.pi * 1.4 * y)
    cropland = (x > crop_west) & (y > 0.24 + 0.025 * np.sin(2 * np.pi * 1.1 * x)) & (y < coast - 0.015)
    labels[cropland] = 5
    bare = (
        (x > 0.33 + 0.02 * np.sin(2 * np.pi * 2 * y))
        & (x < 0.62)
        & (y > 0.42)
        & (y < coast - 0.03)
        & ~water
    )
    labels[bare] = 4

    built = _ellipse(x, y, 0.30, 0.57, 0.105, 0.125) | _ellipse(x, y, 0.36, 0.50, 0.055, 0.045)
    labels[built & ~water] = 3

    if period == "t2":
        expand = (
            _ellipse(x, y, 0.46, 0.46, 0.15, 0.11)
            | _ellipse(x, y, 0.56, 0.40, 0.07, 0.055)
            | _ellipse(x, y, 0.40, 0.40, 0.06, 0.045)
        )
        labels[expand & ~water] = 3
        reclaim = (x > 0.26) & (x < 0.38) & (y > coast - 0.012) & (y < coast + 0.045)
        labels[reclaim] = 3
        labels[_ellipse(x, y, 0.27, 0.63, 0.032, 0.04)] = 4
        labels[water & ~reclaim] = 1
    else:
        labels[water] = 1
    return labels


def _field(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    y, x = _norm_grids(height, width)
    field = np.zeros((height, width), dtype=np.float32)
    for _ in range(4):
        frequency_x = float(rng.uniform(1.0, 3.2))
        frequency_y = float(rng.uniform(1.0, 3.2))
        phase = float(rng.uniform(0, 2 * np.pi))
        field += np.sin((2 * np.pi) * (frequency_x * x + frequency_y * y) + phase)
    return field / 4.0


def _box_mean(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0


def reflectance(
    labels: np.ndarray,
    rng: np.random.Generator,
    field: np.ndarray,
    illumination: np.ndarray,
    pure_masks: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """按类别填光谱。少量像元向易混类别靠拢，但不是整类对调。"""
    height, width = labels.shape
    cube = np.zeros((height, width, 6), dtype=np.float32)
    for class_id, signature in _SIGNATURES.items():
        cube[labels == class_id] = signature

    def pull(mask: np.ndarray, other: int, weight: float) -> None:
        if mask.any():
            cube[mask] = (1.0 - weight) * cube[mask] + weight * _SIGNATURES[other]

    # 建成/裸地、植被/耕地只在平滑场的尾部靠近，留下可解释的混淆。
    pull((labels == 3) & (field > 0.38), 4, 0.72)
    pull((labels == 4) & (field > 0.38), 3, 0.72)
    pull((labels == 5) & (field < -0.38), 2, 0.72)
    pull((labels == 2) & (field < -0.38), 5, 0.72)
    if pure_masks:
        for class_id, mask in pure_masks.items():
            cube[mask] = _SIGNATURES[class_id]

    mixed = np.stack([_box_mean(cube[:, :, band]) for band in range(6)], axis=-1)
    purity = np.zeros((height, width), dtype=np.float32)
    for class_id in _SIGNATURES:
        purity = np.maximum(purity, _box_mean((labels == class_id).astype(np.float32)))
    edge = (1.0 - purity)[:, :, None]
    cube = (1.0 - 0.45 * edge) * cube + (0.45 * edge) * mixed
    if pure_masks:
        for class_id, mask in pure_masks.items():
            cube[mask] = _SIGNATURES[class_id]

    cube *= illumination[:, :, None]
    cube += rng.normal(0.0, 0.008, size=cube.shape).astype(np.float32)
    return np.clip(cube, 0.004, 0.95).astype(np.float32)


def scene_classification(labels: np.ndarray, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 SCL 以及需要压暗或增亮的云/影掩膜（作用在反射率上）。"""
    scl = np.full(labels.shape, 4, dtype=np.uint8)
    scl[labels == 1] = 6
    scl[np.isin(labels, [3, 4])] = 5
    y, x = _norm_grids(*labels.shape)
    cloud = np.zeros(labels.shape, dtype=bool)
    shadow = np.zeros(labels.shape, dtype=bool)
    if period == "t1":
        cirrus = (x > 0.84) & (y < 0.16)
        scl[cirrus] = 10
        cloud = cirrus
    else:
        cloud = ((x - 0.18) ** 2 + (y - 0.15) ** 2) < 0.075**2
        shadow = ((x - 0.25) ** 2 + (y - 0.22) ** 2) < 0.055**2
        shadow &= ~cloud
        scl[cloud] = 9
        scl[shadow] = 3
    return scl, cloud, shadow


def _aggregate_mean(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    return values.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def _aggregate_scl(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    blocks = values.reshape(height // 2, 2, width // 2, 2).transpose(0, 2, 1, 3).reshape(height // 2, width // 2, 4)
    priority = _SCL_PRIORITY[blocks]
    chosen = priority.argmax(axis=2)
    return np.take_along_axis(blocks, chosen[:, :, None], axis=2)[:, :, 0].astype(np.uint8)


def _to_dn(reflectance_band: np.ndarray) -> np.ndarray:
    return np.clip(np.round(reflectance_band * 10000.0), 1, 10000).astype(np.uint16)


def build_period(
    folder: Path,
    grid: dict,
    period: str,
    date: str,
    seed: int,
    field: np.ndarray,
    illumination: np.ndarray,
) -> dict:
    rng = np.random.default_rng(seed)
    labels = land_cover(grid["height"], grid["width"], period)
    y, x = _norm_grids(grid["height"], grid["width"])
    # 旧城核心和拆改斑块保持纯净光谱，避免变化信号被材料噪声淹没。
    pure = {3: _ellipse(x, y, 0.30, 0.57, 0.05, 0.055) & (labels == 3)}
    if period == "t2":
        pure[4] = _ellipse(x, y, 0.27, 0.63, 0.032, 0.04) & (labels == 4)
    cube = reflectance(labels, rng, field, illumination, pure)
    scl, cloud, shadow = scene_classification(labels, period)
    cube[cloud] = np.clip(cube[cloud] + 0.42, 0, 1)
    cube[shadow] = cube[shadow] * 0.4

    transform: Affine = grid["transform"]
    crs = grid["crs"]
    folder.mkdir(parents=True, exist_ok=True)
    band_names = ("B02", "B03", "B04", "B08", "B11", "B12")
    for index, name in enumerate(band_names):
        if name in BANDS_10M:
            write_raster(folder / f"{name}.tif", _to_dn(cube[:, :, index]), transform, crs, "uint16", 0)
        else:
            coarse = _aggregate_mean(cube[:, :, index])
            transform_20 = Affine(20, 0, transform.c, 0, -20, transform.f)
            write_raster(folder / f"{name}.tif", _to_dn(coarse), transform_20, crs, "uint16", 0)
    scl_20 = _aggregate_scl(scl)
    transform_20 = Affine(20, 0, transform.c, 0, -20, transform.f)
    write_raster(folder / "SCL.tif", scl_20, transform_20, crs, "uint8", None)
    write_raster(folder / "reference.tif", labels, transform, crs, "uint8", 0)
    meta = {
        "product": "SIMULATED_S2_L2A",
        "date": date,
        "period": period,
        "crs": crs,
        "resolution_m": {"10m": list(BANDS_10M), "20m": list(BANDS_20M) + ["SCL"]},
        "classes": [{"id": item.id, "zh": item.zh} for item in CLASSES],
        "note": "教学示例景。反射率按 L2A 量级合成，不能当作 Copernicus 原始影像。",
    }
    (folder / "scene.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"labels": labels, "scl": scl, "cloud_pixels": int(np.isin(scl, (0, 1, 3, 8, 9, 10, 11)).sum())}


def build_demo(config: dict) -> Path:
    from landcover.config import resolve

    root = resolve(config, config["paths"]["demo_root"])
    grid = grid_from_aoi(config["aoi"])
    seed = int(config["seed"])
    shared = np.random.default_rng(seed)
    field = _field(grid["height"], grid["width"], shared)
    illumination = 1.0 + 0.05 * _field(grid["height"], grid["width"], shared)
    build_period(root / "t1", grid, "t1", config["dates"]["t1"], seed, field, illumination)
    build_period(root / "t2", grid, "t2", config["dates"]["t2"], seed + 1, field, illumination * np.float32(1.02))
    (root / "grid.json").write_text(
        json.dumps(
            {
                "crs": grid["crs"],
                "width": grid["width"],
                "height": grid["height"],
                "resolution_m": grid["resolution"],
                "bounds_utm": grid["bounds_utm"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


# 重采样枚举留在本模块，便于测试确认 20 m 波段不会被最近邻错读成反射率。
REFLECTANCE_RESAMPLING = Resampling.bilinear
SCL_RESAMPLING = Resampling.nearest
