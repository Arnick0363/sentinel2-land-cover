"""从参考图或外部样本点提取训练 / 验证样本。

示例景的参考图只用来模拟目视解译后的纯净像元，不进入特征。
验证块与训练块不重叠，避免同一位置同时出现在两侧。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from landcover.legend import CLASS_BY_ID, CLASSES


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    current = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(current, 1, constant_values=False)
        current = (
            padded[:-2, :-2]
            & padded[:-2, 1:-1]
            & padded[:-2, 2:]
            & padded[1:-1, :-2]
            & padded[1:-1, 1:-1]
            & padded[1:-1, 2:]
            & padded[2:, :-2]
            & padded[2:, 1:-1]
            & padded[2:, 2:]
        )
    return current


def _candidates(reference: np.ndarray, valid: np.ndarray, erode_pixels: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    found = {}
    for spec in CLASSES:
        mask = _erode(reference == spec.id, erode_pixels) & valid
        found[spec.id] = np.nonzero(mask)
    return found


def _split_blocks(candidate_sets: list[dict], block_size: int, fraction: float, rng: np.random.Generator) -> dict[int, set]:
    validation_blocks = {}
    for spec in CLASSES:
        blocks = set()
        for candidates in candidate_sets:
            rows, cols = candidates[spec.id]
            if len(rows):
                blocks.update(zip((rows // block_size).tolist(), (cols // block_size).tolist()))
        ordered = list(blocks)
        rng.shuffle(ordered)
        if not ordered:
            raise ValueError(f"类别 {spec.zh} 没有可用的纯净像元")
        n_val = max(1, int(round(len(ordered) * fraction)))
        n_val = min(n_val, len(ordered) - 1) if len(ordered) > 1 else 1
        validation_blocks[spec.id] = set(map(tuple, ordered[:n_val]))
    return validation_blocks


def _draw(
    rows: np.ndarray,
    cols: np.ndarray,
    block_size: int,
    val_blocks: set,
    want_train: int,
    want_val: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(rows) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty, empty
    block_rows = rows // block_size
    block_cols = cols // block_size
    keys = list(zip(block_rows.tolist(), block_cols.tolist()))
    is_val = np.array([key in val_blocks for key in keys], dtype=bool)

    def take(pool_rows, pool_cols, count):
        if len(pool_rows) == 0 or count <= 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        choice = rng.choice(len(pool_rows), size=min(count, len(pool_rows)), replace=False)
        return pool_rows[choice], pool_cols[choice]

    train_rows, train_cols = take(rows[~is_val], cols[~is_val], want_train)
    val_rows, val_cols = take(rows[is_val], cols[is_val], want_val)
    return train_rows, train_cols, val_rows, val_cols


def sample_from_references(scenes: dict[str, dict], sample_cfg: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    erode_pixels = int(sample_cfg.get("erode_pixels", 2))
    block_size = int(sample_cfg.get("block_size", 40))
    fraction = float(sample_cfg.get("val_block_fraction", 0.35))
    prepared = {}
    for period, scene in scenes.items():
        if scene["reference"] is None:
            raise ValueError(f"{period} 缺少 reference.tif，示例模式无法布设样本")
        prepared[period] = _candidates(scene["reference"], scene["valid"], erode_pixels)
    val_blocks = _split_blocks(list(prepared.values()), block_size, fraction, rng)
    records = []
    for period, candidates in prepared.items():
        transform = scenes[period]["profile"]["transform"]
        for spec in CLASSES:
            rows, cols = candidates[spec.id]
            train_rows, train_cols, val_rows, val_cols = _draw(
                rows,
                cols,
                block_size,
                val_blocks[spec.id],
                int(sample_cfg["per_class_train"]),
                int(sample_cfg["per_class_val"]),
                rng,
            )
            for split, split_rows, split_cols in (
                ("train", train_rows, train_cols),
                ("val", val_rows, val_cols),
            ):
                for row, col in zip(split_rows.tolist(), split_cols.tolist()):
                    x = float(transform.c + (col + 0.5) * transform.a)
                    y = float(transform.f + (row + 0.5) * transform.e)
                    records.append(
                        {
                            "period": period,
                            "split": split,
                            "class_id": spec.id,
                            "class_key": spec.key,
                            "class_zh": spec.zh,
                            "row": int(row),
                            "col": int(col),
                            "x": x,
                            "y": y,
                        }
                    )
    return records


def _to_lonlat(records: list[dict], crs: str) -> None:
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    if not records:
        return
    lon, lat = transformer.transform([item["x"] for item in records], [item["y"] for item in records])
    for item, lon_i, lat_i in zip(records, lon, lat):
        item["lon"] = float(lon_i)
        item["lat"] = float(lat_i)


def write_samples(records: list[dict], folder: Path, crs: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _to_lonlat(records, crs)
    for split in ("train", "val"):
        subset = [item for item in records if item["split"] == split]
        features = []
        for item in subset:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "class_id": item["class_id"],
                        "class_key": item["class_key"],
                        "class_zh": item["class_zh"],
                        "period": item["period"],
                        "split": item["split"],
                        "row": item["row"],
                        "col": item["col"],
                    },
                    "geometry": {"type": "Point", "coordinates": [round(item["lon"], 7), round(item["lat"], 7)]},
                }
            )
        collection = {"type": "FeatureCollection", "features": features}
        (folder / f"{split}.geojson").write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
        with (folder / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["period", "split", "class_id", "class_zh", "row", "col", "x", "y", "lon", "lat"],
            )
            writer.writeheader()
            for item in subset:
                writer.writerow({key: item[key] for key in writer.fieldnames})


def load_samples(folder: Path) -> list[dict]:
    records = []
    for split in ("train", "val"):
        with (folder / f"{split}.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    {
                        "period": row["period"],
                        "split": row["split"],
                        "class_id": int(row["class_id"]),
                        "class_zh": row["class_zh"],
                        "class_key": CLASS_BY_ID[int(row["class_id"])].key,
                        "row": int(row["row"]),
                        "col": int(row["col"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                    }
                )
    return records


def features_from_records(scenes: dict[str, dict], records: list[dict], split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    kept = []
    rows_x = []
    rows_y = []
    for item in records:
        if item["split"] != split:
            continue
        stack = scenes[item["period"]]["stack"]
        valid = scenes[item["period"]]["valid"]
        row, col = item["row"], item["col"]
        if not valid[row, col]:
            continue
        values = stack[row, col, :]
        if not np.isfinite(values).all():
            continue
        rows_x.append(values)
        rows_y.append(item["class_id"])
        kept.append(item)
    if not rows_x:
        raise ValueError(f"{split} 样本在云掩膜后为空")
    return np.vstack(rows_x).astype(np.float32), np.asarray(rows_y, dtype=int), kept
