"""类别、波段与指数的统一定义，供分类、制图和 QML 共用。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSpec:
    id: int
    key: str
    zh: str
    color: str


# 五类体系：必须包含建设用地，才能做城市扩张。编号与 QML 一致。
CLASSES = (
    ClassSpec(1, "water", "水体", "#2c7bb6"),
    ClassSpec(2, "vegetation", "植被", "#1a9850"),
    ClassSpec(3, "built", "建设用地", "#d73027"),
    ClassSpec(4, "bare", "裸地", "#fdae61"),
    ClassSpec(5, "cropland", "耕地", "#66bd63"),
)

CLASS_BY_ID = {c.id: c for c in CLASSES}
BUILT_ID = 3
NODATA = 0

# 10 m 波段与 20 m 波段分开存放，预处理时重采样到 10 m。
BANDS_10M = ("B02", "B03", "B04", "B08")
BANDS_20M = ("B11", "B12")
REFLECTANCE_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# NDWI_SWIR 对应 IndiaSAT 中的 NDWI = (B8-B12)/(B8+B12)，不是 McFeeters NDWI。
FEATURE_NAMES = REFLECTANCE_BANDS + ("NDVI", "NDBI", "MNDWI", "NDWI_SWIR")

# Sen2Cor SCL：无数据、饱和、云影、云、卷云、雪。
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)

# 变化图编码。稳定非建成对应 IndiaSAT 的 CNBU，稳定建成对应 CBU，
# 新增建设用地对应 Changing。建成转出是本项目额外保留的一类。
CHANGE_CLASSES = (
    ClassSpec(1, "stable_non_built", "稳定非建成", "#d9d9d9"),
    ClassSpec(2, "stable_built", "稳定建成", "#737373"),
    ClassSpec(3, "expansion", "新增建设用地", "#d73027"),
    ClassSpec(4, "loss", "建成转出", "#4575b4"),
)

SECTORS = ("北", "东北", "东", "东南", "南", "西南", "西", "西北")
