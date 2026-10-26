"""分类图、变化图、混淆矩阵与光谱曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from landcover.legend import CHANGE_CLASSES, CLASSES, SECTORS


def setup_font() -> None:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _listed(colors: list[str]) -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    return cmap, BoundaryNorm(bounds, cmap.N)


def _decorate(ax, title: str, height: int, resolution: float) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    px = 1000.0 / resolution
    x0, y0 = 14, height - 22
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=2.6, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - 8, "1 km", ha="center", va="bottom", fontsize=8)
    ax.annotate(
        "",
        xy=(0.94, 0.20),
        xytext=(0.94, 0.07),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
    )
    ax.text(0.94, 0.205, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)


def _legend(ax, specs) -> None:
    handles = [Patch(facecolor=item.color, edgecolor="#555555", label=item.zh) for item in specs]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.92)


def save_classification(path: Path, labels: np.ndarray, title: str, resolution: float) -> None:
    setup_font()
    cmap, norm = _listed(["#f7f7f7"] + [item.color for item in CLASSES])
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=140)
    ax.imshow(labels, cmap=cmap, norm=norm, interpolation="nearest")
    _decorate(ax, title, labels.shape[0], resolution)
    _legend(ax, CLASSES)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_change(path: Path, codes: np.ndarray, title: str, resolution: float) -> None:
    setup_font()
    cmap, norm = _listed(["#ffffff"] + [item.color for item in CHANGE_CLASSES])
    fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=140)
    ax.imshow(codes, cmap=cmap, norm=norm, interpolation="nearest")
    _decorate(ax, title, codes.shape[0], resolution)
    _legend(ax, CHANGE_CLASSES)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_rgb(path: Path, red: np.ndarray, green: np.ndarray, blue: np.ndarray, title: str) -> None:
    setup_font()
    bands = []
    for band in (red, green, blue):
        valid = np.isfinite(band)
        lo, hi = np.percentile(band[valid], [2, 98]) if valid.any() else (0, 1)
        bands.append(np.clip((band - lo) / (hi - lo + 1e-6), 0, 1))
    rgb = np.dstack(bands)
    rgb[~np.isfinite(red)] = 1
    fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=140)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_confusion(path: Path, matrix: np.ndarray, labels_zh: list[str], title: str) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=140)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels_zh)), labels_zh, rotation=30, ha="right")
    ax.set_yticks(range(len(labels_zh)), labels_zh)
    ax.set_xlabel("预测")
    ax.set_ylabel("参考")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_directions(path: Path, sectors: dict, dominant: str | None) -> None:
    setup_font()
    values = [sectors[name] for name in SECTORS]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    colors = ["#d73027" if name == dominant else "#92c5de" for name in SECTORS]
    ax.bar(SECTORS, values, color=colors)
    ax.set_ylabel("新增建设用地（km²）")
    ax.set_title("扩张像元相对前期建成区质心的方向")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_areas(path: Path, area_t1: dict, area_t2: dict) -> None:
    setup_font()
    names = [item.zh for item in CLASSES]
    keys = [item.id for item in CLASSES]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=140)
    ax.bar(x - 0.18, [area_t1[key] for key in keys], width=0.36, label="2016", color="#74add1")
    ax.bar(x + 0.18, [area_t2[key] for key in keys], width=0.36, label="2024", color="#d73027")
    ax.set_xticks(x, names)
    ax.set_ylabel("面积（km²）")
    ax.set_title("两期分类面积")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_signatures(path: Path, curves: dict[str, np.ndarray]) -> None:
    setup_font()
    wavelengths = [492, 560, 665, 833, 1614, 2202]
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=140)
    for spec in CLASSES:
        ax.plot(wavelengths, curves[spec.zh], color=spec.color, marker="o", label=spec.zh)
    ax.set_xlabel("近似中心波长（nm）")
    ax.set_ylabel("地表反射率")
    ax.set_title("训练样本平均光谱")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_importance(path: Path, names: list[str], importance: np.ndarray) -> None:
    setup_font()
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=140)
    ax.barh(np.array(names)[order], importance[order], color="#4575b4")
    ax.set_xlabel("基尼重要度")
    ax.set_title("随机森林特征重要度")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_overview(
    path: Path,
    class_t1: np.ndarray,
    class_t2: np.ndarray,
    change: np.ndarray,
    matrix: np.ndarray,
    labels_zh: list[str],
    summary_lines: list[str],
    resolution: float,
) -> None:
    setup_font()
    class_cmap, class_norm = _listed(["#f7f7f7"] + [item.color for item in CLASSES])
    change_cmap, change_norm = _listed(["#ffffff"] + [item.color for item in CHANGE_CLASSES])
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4), dpi=140)
    panels = (
        (axes[0, 0], class_t1, class_cmap, class_norm, "2016 土地利用"),
        (axes[0, 1], class_t2, class_cmap, class_norm, "2024 土地利用"),
        (axes[1, 0], change, change_cmap, change_norm, "城市扩张"),
    )
    for ax, image, cmap, norm, title in panels:
        ax.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
        _decorate(ax, title, image.shape[0], resolution)
    _legend(axes[0, 1], CLASSES)
    _legend(axes[1, 0], CHANGE_CLASSES)
    ax = axes[1, 1]
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels_zh)), labels_zh, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels_zh)), labels_zh, fontsize=8)
    ax.set_xlabel("预测")
    ax.set_ylabel("参考")
    ax.set_title("独立验证混淆矩阵")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    ax.text(
        0.0,
        -0.38,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        family=plt.rcParams["font.family"],
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
