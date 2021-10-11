```python
from pathlib import Path
import sys
import os
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.plot
import matplotlib.pyplot as plt

path_mod = f"{Path(os.path.dirname(os.path.abspath(''))).parents[1]}/"
sys.path.append(path_mod)

# Setup

iso3 = "mdg"
raw_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'raw', iso3)
processed_dir = os.path.join(os.environ["AA_DATA_DIR"], 'public', 'processed', iso3)

ADM_LEVEL = "adm3"
adm3_path = os.path.join(raw_dir, 'cod_ab', f"mdg_admbnda_adm3_BNGRC_OCHA_20181031.shp")
adm3 = gpd.read_file(adm3_path, crs='4326').to_crs('ESRI:54009')

adm2_path = os.path.join(raw_dir, 'cod_ab', f"mdg_admbnda_adm2_BNGRC_OCHA_20181031.shp")
adm2 = gpd.read_file(adm2_path, crs='4326').to_crs('ESRI:54009')

def plot_urban(adm, adm2_string):
    """
    Simple function to plot urban areas alongside
    the ADM3 / commune boundaries, based off simple
    string search for ADM2 names.
    """
    adm = adm[adm.ADM2_EN.str.contains(adm2_string)]
    shapes = adm["geometry"]
    with rasterio.open(os.path.join(raw_dir, 'ghs', 'mdg_SMOD_2015_1km_mosaic.tif')) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        
    fig, ax = plt.subplots(figsize=(15, 15))
    rasterio.plot.show(np.ma.array(out_image[0,:,:], mask = out_image[0,:,:] == -200),
                   transform=out_transform,ax=ax)
    adm.plot(ax=ax, facecolor='none', edgecolor='w')
    plt.title(adm2_string)
```

### Quick check Antsirabe GHS classification

This script is just to quickly plot Antsirabe to see why it has not been classified as an urban area. Let's load in the raster data and crop to just the 2 ADM2 areas for Antsirabe.

```python
plot_urban(adm3, "Antsirabe")
```

We can see here that there is only a small portion of the area of Antsirabe classified as urban, and the largest urban agglomeration lies along the boundaries of primary communes. How does this compare to other areas classified as urban. Let's check out Toamisana.

```python
plot_urban(adm3, "Toamasina")
```

Aha, in fact, it looks like this is a relic more of commune boundary definitions than of actual urban area. Something we might want to try could be looking at ADM2 boundaries themselves and classifying with a lower threshold. Let's see how this would look not just for Antsirabe and Taomasina, but also for a variety of other areas that had been classified as urban using the ADM3 threshold.

```python
plot_urban(adm2, "Antsirabe")
```

```python
plot_urban(adm2, "Toamasina")
```

```python
plot_urban(adm2, "Toliary")
```

```python
plot_urban(adm2, "Fianarantsoa")
```

```python
plot_urban(adm2, "Antananarivo|Arrondissement")
```

```python
plot_urban(adm2, "Mahajanga")
```

Overall, it seems that most of these areas should still be classified as urban areas. However, places like Mahajanga fall into a different trap where ADM2 boundaries cover wide swaths of area, with only a small portion classified as urban. Given that the classification for urban consists of raster scores 11-13 for non-urban, 21-23 for suburban - dense, and 30 for urban centre, maybe we could look at weighting the percent to account for areas where it drops from urban centre to rural abruptly with limited suburban/semi-urban spaces.

```python

```
