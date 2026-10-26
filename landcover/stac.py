"""构造 Element84 Earth Search 的 STAC 检索体。下载逻辑在 scripts/download_sentinel.py。"""

from __future__ import annotations


def search_body(aoi: dict, start: str, end: str, cloud_max: float) -> dict:
    return {
        "collections": ["sentinel-2-l2a"],
        "bbox": [float(aoi["west"]), float(aoi["south"]), float(aoi["east"]), float(aoi["north"])],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": cloud_max}},
        "limit": 5,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
