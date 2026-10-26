"""光谱指数。分母接近 0 时返回 NaN，避免 NoData 假值。"""

from __future__ import annotations

import numpy as np

# 反射率量级下，分母小于该阈值视为无效。
_DENOM_EPS = 1e-6


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算 (a-b)/(a+b)。a、b 为反射率。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = a - b
    denominator = a + b
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > _DENOM_EPS)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (B08 - B04) / (B08 + B04)。"""
    return normalized_difference(nir, red)


def ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDBI = (B11 - B08) / (B11 + B08)。"""
    return normalized_difference(swir1, nir)


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI = (B03 - B11) / (B03 + B11)。"""
    return normalized_difference(green, swir1)


def ndwi_swir(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """IndiaSAT 使用的 NDWI = (B08 - B12) / (B08 + B12)。"""
    return normalized_difference(nir, swir2)
