<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0" hasScaleBasedVisibilityFlag="0">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" band="1" type="paletted">
      <rasterTransparency/>
      <colorPalette>
        <paletteEntry color="#d9d9d9" value="1" label="稳定非建成" alpha="255"/>
        <paletteEntry color="#737373" value="2" label="稳定建成" alpha="255"/>
        <paletteEntry color="#d73027" value="3" label="新增建设用地" alpha="255"/>
        <paletteEntry color="#4575b4" value="4" label="建成转出" alpha="255"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]"/>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeOn="0" colorizeRed="255" colorizeBlue="128" grayscaleMode="0" saturation="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
