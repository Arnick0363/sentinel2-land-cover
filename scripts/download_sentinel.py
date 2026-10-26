"""检索并可选下载 Sentinel-2 L2A。不需要把整景放进仓库。

公开 STAC（Element84 Earth Search，AWS 上的 L2A COG）不需要 Copernicus 账号。
只检索时不写栅格；确认景号后再按 AOI 窗口读取波段。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landcover.config import load_config
from landcover.stac import search_body

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
ASSETS = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B11": "swir16",
    "B12": "swir22",
    "SCL": "scl",
}


def search(aoi: dict, start: str, end: str, cloud_max: float) -> list[dict]:
    payload = json.dumps(search_body(aoi, start, end, cloud_max)).encode("utf-8")
    request = urllib.request.Request(
        STAC_SEARCH,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        document = json.loads(response.read().decode("utf-8"))
    found = []
    for feature in document.get("features", []):
        props = feature.get("properties", {})
        assets = feature.get("assets", {})
        found.append(
            {
                "id": feature.get("id"),
                "datetime": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover"),
                "assets": {band: assets.get(key, {}).get("href") for band, key in ASSETS.items()},
            }
        )
    return found


def download_scene(item: dict, folder: Path) -> None:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    folder.mkdir(parents=True, exist_ok=True)
    config = load_config()
    aoi = config["aoi"]
    for band, href in item["assets"].items():
        if not href:
            raise RuntimeError(f"{item['id']} 缺少 {band}")
        with rasterio.open(href) as source:
            west, south, east, north = transform_bounds(
                "EPSG:4326", source.crs, aoi["west"], aoi["south"], aoi["east"], aoi["north"]
            )
            window = from_bounds(west, south, east, north, source.transform)
            window = window.round_offsets().round_lengths()
            data = source.read(1, window=window, boundless=True)
            transform = source.window_transform(window)
            profile = source.profile.copy()
            profile.update(width=data.shape[1], height=data.shape[0], transform=transform, compress="deflate")
            with rasterio.open(folder / f"{band}.tif", "w", **profile) as dataset:
                dataset.write(data, 1)
        print(f"已写入 {folder / (band + '.tif')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索横琴附近的 Sentinel-2 L2A")
    parser.add_argument("--download", action="store_true", help="把云量最低的一景按 AOI 裁剪到 data/raw")
    parser.add_argument("--period", choices=["t1", "t2"], default="t1")
    args = parser.parse_args()
    config = load_config()
    if args.period == "t1":
        start, end = "2016-09-01", "2016-11-30"
        folder = ROOT / "data" / "raw" / "t1"
    else:
        start, end = "2024-09-01", "2024-11-30"
        folder = ROOT / "data" / "raw" / "t2"
    items = search(config["aoi"], start, end, float(config["dates"]["cloud_cover_max"]))
    if not items:
        print("没有符合云量阈值的景。")
        return
    for item in items:
        print(f"{item['id']}  云量 {item['cloud_cover']}  {item['datetime']}")
    if args.download:
        download_scene(items[0], folder)
        print("下载完成。请在 QGIS 中采集样本，并把 config 的 mode 改为 external。")


if __name__ == "__main__":
    main()
