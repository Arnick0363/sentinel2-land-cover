"""端到端：示例景或外部景 → 预处理 → 分类 → 精度 → 变化检测 → 图件。"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np

from landcover.change import summarize_change, transition_matrix
from landcover.classify import GaussianMLC, majority_filter, predict_raster, train_random_forest
from landcover.config import load_config, resolve
from landcover.demo_scene import build_demo
from landcover.legend import BUILT_ID, CHANGE_CLASSES, CLASSES, CLASS_BY_ID, FEATURE_NAMES
from landcover.metrics import binary_built_labels, evaluate
from landcover.preprocess import check_dates, load_scene
from landcover.rasters import read_profile, write_paletted, write_raster
from landcover.samples import features_from_records, load_samples, sample_from_references, write_samples
from landcover import visualize


def _years(config: dict) -> float:
    start = date.fromisoformat(config["dates"]["t1"])
    end = date.fromisoformat(config["dates"]["t2"])
    return (end - start).days / 365.25


def _colormap(specs, nodata_color=(255, 255, 255, 0)) -> dict:
    table = {0: nodata_color}
    for item in specs:
        color = item.color.lstrip("#")
        table[item.id] = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4)) + (255,)
    return table


def _class_area(labels: np.ndarray, valid: np.ndarray, pixel_km2: float) -> dict:
    areas = {}
    for spec in CLASSES:
        areas[spec.id] = float(((labels == spec.id) & valid).sum() * pixel_km2)
    return areas


def _round(value, digits=4):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item, digits) for item in value]
    return value


def run(config: dict) -> dict:
    root = Path(config["_root"])
    output = resolve(config, config["paths"]["outputs"])
    map_dir = output / "maps"
    fig_dir = output / "figures"
    table_dir = output / "tables"
    sample_dir = output / "samples"
    for folder in (map_dir, fig_dir, table_dir, sample_dir, output / "models"):
        folder.mkdir(parents=True, exist_ok=True)

    mode = config.get("mode", "demo")
    if mode == "demo":
        demo_root = build_demo(config)
        folder_t1 = demo_root / "t1"
        folder_t2 = demo_root / "t2"
    elif mode == "external":
        folder_t1 = resolve(config, config["paths"]["external_t1"])
        folder_t2 = resolve(config, config["paths"]["external_t2"])
    else:
        raise ValueError(f"未知 mode：{mode}")

    scene_t1 = load_scene(folder_t1)
    native_t2 = read_profile(folder_t2 / "B02.tif")
    alignment = check_dates({"profile": scene_t1["native_profile"]}, {"profile": native_t2})
    # 变化检测前把后期对齐到前期网格。示例景本身已经共网格，偏移应为 0。
    scene_t2 = load_scene(folder_t2, target=scene_t1["profile"])
    scenes = {"t1": scene_t1, "t2": scene_t2}

    if mode == "demo":
        records = sample_from_references(scenes, config["samples"], int(config["seed"]))
        write_samples(records, sample_dir, scene_t1["profile"]["crs"].to_string() if hasattr(scene_t1["profile"]["crs"], "to_string") else str(scene_t1["profile"]["crs"]))
    else:
        records = load_samples(resolve(config, config["paths"]["samples"]))

    x_train, y_train, train_kept = features_from_records(scenes, records, "train")
    x_val, y_val, val_kept = features_from_records(scenes, records, "val")
    present = set(np.unique(y_train)).intersection(np.unique(y_val))
    missing = [spec.zh for spec in CLASSES if spec.id not in present]
    if missing:
        raise RuntimeError("下列类别在训练或验证中缺失：" + "、".join(missing))

    forest = train_random_forest(x_train, y_train, config["model"], int(config["seed"]))
    mlc = GaussianMLC().fit(x_train, y_train)
    pred_rf = forest.predict(x_val)
    pred_mlc = mlc.predict(x_val)
    labels = [spec.id for spec in CLASSES]
    accuracy_rf = evaluate(y_val, pred_rf, labels)
    accuracy_mlc = evaluate(y_val, pred_mlc, labels)
    bu_accuracy = evaluate(binary_built_labels(y_val, BUILT_ID), binary_built_labels(pred_rf, BUILT_ID), [0, 1])

    map_t1 = majority_filter(predict_raster(forest, scene_t1["stack"], scene_t1["valid"]), scene_t1["valid"])
    map_t2 = majority_filter(predict_raster(forest, scene_t2["stack"], scene_t2["valid"]), scene_t2["valid"])
    years = _years(config)
    change = summarize_change(map_t1, map_t2, scene_t1["profile"]["transform"], years, BUILT_ID)
    transitions = transition_matrix(map_t1, map_t2, labels, scene_t1["profile"]["transform"])

    per_date = {}
    for period in ("t1", "t2"):
        truth = np.array([item["class_id"] for item in val_kept if item["period"] == period])
        predicted = pred_rf[np.array([item["period"] == period for item in val_kept])]
        if len(truth):
            per_date[period] = evaluate(truth, predicted, labels)

    full_map = None
    if scene_t1["reference"] is not None and scene_t2["reference"] is not None:
        parts_true = []
        parts_pred = []
        for scene, prediction in ((scene_t1, map_t1), (scene_t2, map_t2)):
            mask = scene["valid"] & (scene["reference"] > 0)
            parts_true.append(scene["reference"][mask])
            parts_pred.append(prediction[mask])
        full_map = evaluate(np.concatenate(parts_true), np.concatenate(parts_pred), labels)

    names_zh = [spec.zh for spec in CLASSES]
    accuracy_rf["per_class_zh"] = {
        spec.zh: accuracy_rf["per_class"][index] for index, spec in enumerate(CLASSES)
    }
    pixel_km2 = change["pixel_area_m2"] / 1_000_000.0
    valid_both = (map_t1 != 0) & (map_t2 != 0)
    area_t1 = _class_area(map_t1, valid_both, pixel_km2)
    area_t2 = _class_area(map_t2, valid_both, pixel_km2)

    curves = {}
    for spec in CLASSES:
        subset = x_train[y_train == spec.id][:, :6]
        curves[spec.zh] = subset.mean(axis=0) if len(subset) else np.zeros(6)

    resolution = abs(float(scene_t1["profile"]["transform"].a))
    date_t1 = config["dates"]["t1"][:4]
    date_t2 = config["dates"]["t2"][:4]
    visualize.save_classification(fig_dir / "classification_t1.png", map_t1, f"{date_t1} 土地利用分类", resolution)
    visualize.save_classification(fig_dir / "classification_t2.png", map_t2, f"{date_t2} 土地利用分类", resolution)
    visualize.save_change(fig_dir / "urban_expansion.png", change["change"], "新增建设用地与建成转出", resolution)
    visualize.save_rgb(
        fig_dir / "rgb_t1.png",
        scene_t1["bands"]["B04"],
        scene_t1["bands"]["B03"],
        scene_t1["bands"]["B02"],
        f"{date_t1} 真彩色（2%–98% 拉伸）",
    )
    visualize.save_rgb(
        fig_dir / "rgb_t2.png",
        scene_t2["bands"]["B04"],
        scene_t2["bands"]["B03"],
        scene_t2["bands"]["B02"],
        f"{date_t2} 真彩色（2%–98% 拉伸）",
    )
    visualize.save_confusion(
        fig_dir / "confusion_matrix.png",
        np.array(accuracy_rf["confusion_matrix"]),
        names_zh,
        "随机森林独立验证",
    )
    visualize.save_directions(fig_dir / "expansion_direction.png", change["sectors_km2"], change["dominant_direction"])
    visualize.save_areas(fig_dir / "class_areas.png", area_t1, area_t2)
    visualize.save_signatures(fig_dir / "spectral_signatures.png", curves)
    visualize.save_importance(fig_dir / "feature_importance.png", list(FEATURE_NAMES), forest.feature_importances_)
    summary_lines = [
        f"样本总体精度 {accuracy_rf['oa'] * 100:.1f}%   Kappa {accuracy_rf['kappa']:.3f}",
        f"最大似然总体精度 {accuracy_mlc['oa'] * 100:.1f}%   Kappa {accuracy_mlc['kappa']:.3f}",
        f"新增建设用地 {change['expansion_km2']:.3f} km²，主方向 {change['dominant_direction']}",
        f"建成转出 {change['loss_km2']:.3f} km²，年均扩张 {change['annual_expansion_km2']:.3f} km²/年",
    ]
    visualize.save_overview(
        fig_dir / "overview.png",
        map_t1,
        map_t2,
        change["change"],
        np.array(accuracy_rf["confusion_matrix"]),
        names_zh,
        summary_lines,
        resolution,
    )

    crs = scene_t1["profile"]["crs"]
    transform = scene_t1["profile"]["transform"]
    write_paletted(map_dir / "class_t1.tif", map_t1, transform, crs, _colormap(CLASSES))
    write_paletted(map_dir / "class_t2.tif", map_t2, transform, crs, _colormap(CLASSES))
    write_paletted(map_dir / "change.tif", change["change"], transform, crs, _colormap(CHANGE_CLASSES))
    for period, scene in scenes.items():
        for name, array in scene["indices"].items():
            write_raster(map_dir / f"{name.lower()}_{period}.tif", array.astype(np.float32), transform, crs, "float32", np.nan)

    joblib.dump({"model": forest, "feature_names": list(FEATURE_NAMES)}, output / "models" / "random_forest.joblib")

    def count_split(split: str) -> dict:
        counts = {spec.zh: 0 for spec in CLASSES}
        for item in records:
            if item["split"] == split:
                counts[CLASS_BY_ID[item["class_id"]].zh] += 1
        return counts

    metrics = {
        "mode": mode,
        "study_area": config["aoi"]["name"],
        "dates": {"t1": config["dates"]["t1"], "t2": config["dates"]["t2"]},
        "crs": str(crs),
        "resolution_m": resolution,
        "shape": [int(map_t1.shape[0]), int(map_t1.shape[1])],
        "alignment": alignment,
        "cloud_fraction": {"t1": scene_t1["cloud_fraction"], "t2": scene_t2["cloud_fraction"]},
        "samples": {"train": count_split("train"), "val": count_split("val")},
        "feature_names": list(FEATURE_NAMES),
        "post_classification_filter": "3x3 majority",
        "random_forest": {
            "oa": accuracy_rf["oa"],
            "kappa": accuracy_rf["kappa"],
            "n_val": accuracy_rf["n"],
            "n_train": int(len(y_train)),
            "confusion_matrix": accuracy_rf["confusion_matrix"],
            "labels_zh": names_zh,
            "per_class": accuracy_rf["per_class_zh"],
            "feature_importance": {
                name: float(value) for name, value in zip(FEATURE_NAMES, forest.feature_importances_)
            },
            "per_date": {
                period: {"oa": item["oa"], "kappa": item["kappa"], "n": item["n"]} for period, item in per_date.items()
            },
        },
        "maximum_likelihood": {
            "oa": accuracy_mlc["oa"],
            "kappa": accuracy_mlc["kappa"],
            "n_val": accuracy_mlc["n"],
            "confusion_matrix": accuracy_mlc["confusion_matrix"],
            "per_class": {spec.zh: accuracy_mlc["per_class"][i] for i, spec in enumerate(CLASSES)},
        },
        "built_vs_non_built": {
            "oa": bu_accuracy["oa"],
            "kappa": bu_accuracy["kappa"],
            "confusion_matrix": bu_accuracy["confusion_matrix"],
            "labels": ["非建成", "建成"],
        },
        "full_map_vs_reference": None
        if full_map is None
        else {
            "oa": full_map["oa"],
            "kappa": full_map["kappa"],
            "n": full_map["n"],
            "note": "仅示例景。含混合边界像元，对照整幅参考图，不是抽样精度。",
        },
        "area_km2": {
            "t1": {CLASS_BY_ID[key].zh: value for key, value in area_t1.items()},
            "t2": {CLASS_BY_ID[key].zh: value for key, value in area_t2.items()},
        },
        "change": {key: value for key, value in change.items() if key != "change"},
        "transition_km2": {
            "labels_zh": names_zh,
            "km2": transitions["km2"],
            "counts": transitions["counts"],
        },
        "data_note": config.get("data_note", ""),
    }
    metrics_path = table_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_round(metrics, 6), ensure_ascii=False, indent=2), encoding="utf-8")

    with (table_dir / "confusion_matrix_rf.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["参考\\预测"] + names_zh)
        for name, row in zip(names_zh, accuracy_rf["confusion_matrix"]):
            writer.writerow([name] + row)
    with (table_dir / "class_area_km2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["类别", date_t1, date_t2])
        for spec in CLASSES:
            writer.writerow([spec.zh, round(area_t1[spec.id], 6), round(area_t2[spec.id], 6)])
    with (table_dir / "expansion_direction.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["方向", "新增建设用地_km2"])
        for name in change["sectors_km2"]:
            writer.writerow([name, round(change["sectors_km2"][name], 6)])
    with (table_dir / "transition_km2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["前期\\后期"] + names_zh)
        for name, row in zip(names_zh, transitions["km2"]):
            writer.writerow([name] + [round(value, 6) for value in row])

    # 供 README 直接引用的一行结果，避免手抄时抄错。
    (table_dir / "headline.txt").write_text(
        "\n".join(summary_lines + [f"网格偏移 {alignment['offset_m']:.3f} m", f"结果目录 {output.relative_to(root)}"]),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Sentinel-2 土地利用分类与城市扩张检测")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config/config.yaml")
    parser.add_argument("--mode", choices=["demo", "external"], default=None, help="覆盖配置中的 mode")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    metrics = run(config)
    rf = metrics["random_forest"]
    change = metrics["change"]
    print(
        f"随机森林 OA={rf['oa']:.4f}  Kappa={rf['kappa']:.4f}  "
        f"新增建设用地={change['expansion_km2']:.4f} km²  主方向={change['dominant_direction']}"
    )


if __name__ == "__main__":
    main()
