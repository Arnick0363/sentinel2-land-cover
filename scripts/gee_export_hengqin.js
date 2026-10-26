// 珠海横琴 Sentinel-2 L2A 双时相导出。
// 思路来自 IndiaSAT 的 sentinel2_classification.js（云掩膜、云量阈值、NDVI/NDBI），
// 但这里使用 L2A（COPERNICUS/S2_SR_HARMONIZED）和 SCL，而不是原脚本的 L1C + QA60。
// 分类在本地 Python 中完成，GEE 只负责出图。

var aoi = ee.Geometry.Rectangle([113.490, 22.110, 113.535, 22.145]);

function maskS2(image) {
  var scl = image.select("SCL");
  var clear = scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(clear);
}

function composite(start, end) {
  var collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    .map(maskS2);
  var image = collection.median().clip(aoi);
  var ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI");
  var ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI");
  var mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI");
  return image.addBands(ndvi).addBands(ndbi).addBands(mndwi);
}

var t1 = composite("2016-09-01", "2016-11-30");
var t2 = composite("2024-09-01", "2024-11-30");

Export.image.toDrive({
  image: t1.select(["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]),
  description: "hengqin_s2_l2a_2016",
  region: aoi,
  scale: 10,
  maxPixels: 1e8
});

Export.image.toDrive({
  image: t2.select(["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]),
  description: "hengqin_s2_l2a_2024",
  region: aoi,
  scale: 10,
  maxPixels: 1e8
});
