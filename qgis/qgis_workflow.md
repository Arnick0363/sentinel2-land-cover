# QGIS 对照步骤

本机命令行没有安装 QGIS，因此分类结果由 Python 生成。下面是同一套流程在 QGIS 3.x 加半自动分类插件（Semi-Automatic Classification Plugin，SCP）里的菜单操作。步骤对应 Luca Congedo 的教程 [Basic Tutorial 2](https://fromgistors.blogspot.com/2016/09/basic-tutorial-2.html)（清单里的 fromgists.blogspot.com 当前返回 404，实际站点是 fromgistors.blogspot.com）。

## 1. 安装

1. 安装 QGIS 3.28 或更新的 3.x。
2. 插件 → 管理并安装插件，搜索 `Semi-Automatic Classification Plugin`，安装。
3. 可选：再安装 QuickMapServices，用在线底图辅助目视解译。

## 2. 准备波段

真实景从 Copernicus Browser 或 `python scripts/download_sentinel.py` 得到。每个时相一个文件夹，内有 `B02.tif`、`B03.tif`、`B04.tif`、`B08.tif`、`B11.tif`、`B12.tif`、`SCL.tif`。

1. 图层 → 添加图层 → 添加栅格图层，载入全部波段。
2. 若使用示例景，先在项目根目录运行 `python -m landcover`，再载入 `data/demo/t1` 与 `data/demo/t2`。示例景已经是缩放后的地表反射率（DN / 10000），不要再做一次 DOS1。
3. 真实 L1C 才需要 SCP → Preprocessing → Sentinel-2，勾选 Apply DOS1 atmospheric correction。L2A 已由 Sen2Cor 做成地表反射率，跳过 DOS1。

## 3. 裁剪与重采样

1. 图层 → 添加图层 → 添加矢量图层，载入 `data/aoi/hengqin.geojson`。
2. 处理工具箱 → GDAL → 栅格提取 → 按掩膜图层裁剪栅格（Clip raster by mask layer）。对每个波段裁剪到横琴范围。
3. B11、B12、SCL 是 20 m。处理工具箱 → GDAL → 栅格投影 → 变形（重投影）（Warp）。目标分辨率 10 m，反射率用双线性，SCL 用最近邻，对齐到 B02 的网格。Python 侧的 `landcover/preprocess.py` 做的是同一件事。

## 4. 云掩膜

1. 栅格 → 栅格计算器。对 SCL 写出无效掩膜：`SCL = 0 OR SCL = 1 OR SCL = 3 OR SCL = 8 OR SCL = 9 OR SCL = 10 OR SCL = 11`。
2. 用该掩膜把反射率设为空值。SCP 里也可以把云画成 ROI，宏类 ID 设为 0，分类结果中这些像元保持未分类。教程第 5 节就是这种做法。

## 5. 波段集与指数

1. SCP → Band set。按 B02、B03、B04、B08、B11、B12 的顺序加入，Quick wavelength settings 选 Sentinel-2。20 m 波段由 SCP 重采样到 10 m。
2. 工作条 RGB 先用 `4-3-2`（自然色，对应 B04-B03-B02），再用 `B11-B08-B04` 一类的假彩色区分建成区。教程使用的 `3-2-1` 和 `3-7-10` 是它那套波段编号，换成上面的波段名即可。
3. SCP → Band calc，或栅格计算器：
   - NDVI：`("B08"-"B04")/("B08"+"B04")`
   - NDBI：`("B11"-"B08")/("B11"+"B08")`
   - MNDWI：`("B03"-"B11")/("B03"+"B11")`
4. NDVI 图层属性 → 符号系统 → 单波段伪彩色，或直接加载 `qgis/styles/ndvi.qml`。采 ROI 时打开 SCP 的 NDVI 显示，植被应高于裸地。

## 6. 样本

1. SCP 停靠窗口 → Training input，新建一个训练文件。
2. 宏类：1 水体、2 植被、3 建设用地、4 裸地、5 耕地。同一宏类下用不同 C ID 分开屋顶、道路、亮土、密林，这是教程强调的做法。
3. 用区域增长画纯净像元，避开地类边界。Python 示例用参考图腐蚀 2 个像元来模仿这一步。
4. 两个时相都要采。训练和验证分开保存；不要用同一批点既训练又报精度。
5. SCP → ROI Signature list，打开光谱曲线，检查建设用地和裸地是否贴得太近。贴得太近就补样本或删掉不纯的 ROI。

## 7. 分类

1. SCP → Classification → Algorithm。优先 Random forest（与本仓库 `n_estimators: 100` 对应）。若该版本只有 Minimum Distance、Maximum Likelihood、Spectral Angle Mapping，就选 Maximum Likelihood，用来对照 Python 里的高斯最大似然。
2. 勾选 MC ID，让输出是宏类而不是细类。
3. Classification preview 设 500 像元，点图查看。满意后再 Classification output 跑全图。两个时相各跑一次，类别编号必须一致。

## 8. 精度

1. 用没有参与训练的验证 ROI。SCP → Postprocessing → Accuracy，或 Band processing 里的 Accuracy。
2. 读取混淆矩阵、总体精度和 Kappa。本仓库的对应表在 `outputs/tables/confusion_matrix_rf.csv` 和 `outputs/tables/metrics.json`。
3. 总体精度低于 85% 时，先看裸地/建设用地、耕地/植被这两对，回到第 6 步改样本，不要只改随机种子。

## 9. 变化检测

1. 确认两期分类图的像元网格一致（同一 CRS、同一分辨率、原点对齐）。不一致就用 Warp 对齐到 2016 年。
2. 栅格计算器，建设用地编号为 3：
   - 新增：`"class_t1@1" != 3 AND "class_t2@1" = 3`
   - 转出：`"class_t1@1" = 3 AND "class_t2@1" != 3`
3. 或者 SCP → Postprocessing → Land cover change，得到 from-to 图。
4. 栅格 → 分析 → 栅格图层唯一值报告，或用分区统计。像元数乘 100 m² 再除以 1e6 得到平方公里。Python 用的是 UTM 49N 上的像元面积，不是经纬度度数。

## 10. 制图

1. 运行 `qgis/apply_styles.py`（QGIS Python 控制台），或对 `outputs/maps/class_t1.tif`、`class_t2.tif` 加载 `qgis/styles/landcover.qml`，对 `change.tif` 加载 `qgis/styles/change.qml`。
2. 工程 → 新建打印布局。加入两期分类图和变化图。
3. 添加图例、比例尺、指北针，坐标系注明 EPSG:32649。
4. 布局 → 导出为图像，得到 PNG。

`outputs/figures/` 里的 PNG 是 Matplotlib 按同一套配色导出的，方便没有 QGIS 时先看结果。正式排版仍以打印布局为准。
