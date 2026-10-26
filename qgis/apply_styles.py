"""在 QGIS Python 控制台中运行：加载结果图层并套用 QML。

本机若未安装 QGIS，本脚本不会被命令行 Python 执行。
在 QGIS 中：插件 → Python 控制台 → 显示编辑器 → 打开本文件 → 运行。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from qgis.core import QgsCoordinateReferenceSystem, QgsProject, QgsRasterLayer, QgsVectorLayer

    project = QgsProject.instance()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:32649"))
    layers = [
        ("2016 土地利用", ROOT / "outputs" / "maps" / "class_t1.tif", ROOT / "qgis" / "styles" / "landcover.qml"),
        ("2024 土地利用", ROOT / "outputs" / "maps" / "class_t2.tif", ROOT / "qgis" / "styles" / "landcover.qml"),
        ("城市扩张", ROOT / "outputs" / "maps" / "change.tif", ROOT / "qgis" / "styles" / "change.qml"),
        ("2016 NDVI", ROOT / "outputs" / "maps" / "ndvi_t1.tif", ROOT / "qgis" / "styles" / "ndvi.qml"),
        ("2024 NDVI", ROOT / "outputs" / "maps" / "ndvi_t2.tif", ROOT / "qgis" / "styles" / "ndvi.qml"),
    ]
    for title, raster_path, style_path in layers:
        layer = QgsRasterLayer(str(raster_path), title)
        if not layer.isValid():
            print(f"无法加载 {raster_path}")
            continue
        layer.loadNamedStyle(str(style_path))
        layer.triggerRepaint()
        project.addMapLayer(layer)
        print(f"已加载 {title}")

    aoi = QgsVectorLayer(str(ROOT / "data" / "aoi" / "hengqin.geojson"), "横琴范围", "ogr")
    samples = QgsVectorLayer(str(ROOT / "outputs" / "samples" / "val.geojson"), "验证样本", "ogr")
    if aoi.isValid():
        project.addMapLayer(aoi)
    if samples.isValid():
        project.addMapLayer(samples)
    print("图层已加入当前工程。打印布局仍需在 工程 → 新建打印布局 中加图例、比例尺和指北针。")


if __name__ == "__main__":
    main()
