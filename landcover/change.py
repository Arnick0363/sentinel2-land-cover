"""分类后比较：新增建设用地、建成转出、面积与扩张方向。"""

from __future__ import annotations

import numpy as np
from rasterio.transform import Affine

from landcover.legend import BUILT_ID, CHANGE_CLASSES, NODATA, SECTORS


def pixel_area_m2(transform: Affine) -> float:
    return abs(float(transform.a) * float(transform.e))


def change_codes(class_t1: np.ndarray, class_t2: np.ndarray, built_id: int = BUILT_ID) -> np.ndarray:
    """生成变化编码。任一期为 0（云或无效）的像元保持 0。

    1 稳定非建成，2 稳定建成，3 非建成→建成，4 建成→非建成。
    """
    t1 = np.asarray(class_t1)
    t2 = np.asarray(class_t2)
    if t1.shape != t2.shape:
        raise ValueError(f"两期分类图尺寸不一致：{t1.shape} 与 {t2.shape}")
    valid = (t1 != NODATA) & (t2 != NODATA)
    built1 = t1 == built_id
    built2 = t2 == built_id
    out = np.zeros(t1.shape, dtype=np.uint8)
    out[valid & ~built1 & ~built2] = 1
    out[valid & built1 & built2] = 2
    out[valid & ~built1 & built2] = 3
    out[valid & built1 & ~built2] = 4
    return out


def _pixel_centers(transform: Affine, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return xs, ys


def bearing_degrees(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """由东向位移 dx、北向位移 dy 计算从正北顺时针的方位角（度）。"""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def sector_index(bearing: np.ndarray) -> np.ndarray:
    """八方向扇区。每个扇区宽 45°，正北居中（337.5–22.5）。"""
    return np.floor((np.asarray(bearing) + 22.5) % 360.0 / 45.0).astype(int) % 8


def summarize_change(
    class_t1: np.ndarray,
    class_t2: np.ndarray,
    transform: Affine,
    years: float,
    built_id: int = BUILT_ID,
) -> dict:
    codes = change_codes(class_t1, class_t2, built_id)
    area = pixel_area_m2(transform)

    def km2(mask: np.ndarray) -> float:
        return float(mask.sum() * area / 1_000_000.0)

    expansion = codes == 3
    loss = codes == 4
    stable_built = codes == 2
    valid = codes != NODATA
    t1_built = (class_t1 == built_id) & (class_t2 != NODATA) & (class_t1 != NODATA)
    t2_built = (class_t2 == built_id) & (class_t1 != NODATA) & (class_t2 != NODATA)

    rows, cols = np.nonzero(expansion)
    sector_km2 = {name: 0.0 for name in SECTORS}
    centroid_bearing = None
    centroid_distance_km = None
    if len(rows):
        built_rows, built_cols = np.nonzero(t1_built)
        if len(built_rows) == 0:
            origin_rows, origin_cols = rows, cols
        else:
            origin_rows, origin_cols = built_rows, built_cols
        ox, oy = _pixel_centers(transform, origin_rows, origin_cols)
        cx, cy = float(ox.mean()), float(oy.mean())
        ex, ey = _pixel_centers(transform, rows, cols)
        dx = ex - cx
        dy = ey - cy
        bearings = bearing_degrees(dx, dy)
        indices = sector_index(bearings)
        for i, name in enumerate(SECTORS):
            sector_km2[name] = float((indices == i).sum() * area / 1_000_000.0)
        mean_dx = float(dx.mean())
        mean_dy = float(dy.mean())
        centroid_bearing = float(bearing_degrees(np.array([mean_dx]), np.array([mean_dy]))[0])
        centroid_distance_km = float(np.hypot(mean_dx, mean_dy) / 1000.0)

    dominant = max(sector_km2, key=sector_km2.get) if rows.size else None
    expansion_km2 = km2(expansion)
    t1_built_km2 = km2(t1_built)
    valid_km2 = km2(valid)
    return {
        "codes_present": [int(spec.id) for spec in CHANGE_CLASSES],
        "expansion_km2": expansion_km2,
        "loss_km2": km2(loss),
        "stable_built_km2": km2(stable_built),
        "stable_non_built_km2": km2(codes == 1),
        "t1_built_km2": t1_built_km2,
        "t2_built_km2": km2(t2_built),
        "valid_km2": valid_km2,
        "expansion_share_of_t1_built": (expansion_km2 / t1_built_km2) if t1_built_km2 else None,
        "expansion_share_of_valid": (expansion_km2 / valid_km2) if valid_km2 else None,
        "annual_expansion_km2": (expansion_km2 / years) if years else None,
        "years": years,
        "dominant_direction": dominant,
        "sectors_km2": sector_km2,
        "expansion_centroid_bearing_deg": centroid_bearing,
        "expansion_centroid_distance_km": centroid_distance_km,
        "pixel_area_m2": area,
        "change": codes,
    }


def transition_matrix(class_t1: np.ndarray, class_t2: np.ndarray, labels: list[int], transform: Affine) -> dict:
    """有效像元上的转移矩阵，单位为像元数和平方公里。"""
    area = pixel_area_m2(transform)
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {label: i for i, label in enumerate(labels)}
    valid = (class_t1 != NODATA) & (class_t2 != NODATA)
    t1 = class_t1[valid].astype(int)
    t2 = class_t2[valid].astype(int)
    for a, b in zip(t1, t2):
        if a in index and b in index:
            counts[index[a], index[b]] += 1
    return {
        "labels": list(labels),
        "counts": counts.tolist(),
        "km2": (counts * area / 1_000_000.0).tolist(),
    }
