# Data sources

These data sources have either been applied or explored in our flooding anticipatory action work. 

### 1. Global Flood Awareness System (GloFAS) modelled river discharge

__Description__: Daily modelled river discharge data. Provides modelled estimates of the amount of water flowing through a given river section. Average over each 24 hour period, with up to 30 days lead time forecasted. Simulations are run based on a hydrological river routing model with modelled gridded runoff data from global reanalysis. 

__Format__: Raster. 

__Resolution__: 0.1 x 0.1 degree. 

__Access__: [Via API](https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical?tab=overview). 

__Availability__: Global data since 1 January 1979. 

__Update frequency__: Daily. 

__Limitations__: Modelled, rather than historical river discharge. See [this documentation](https://www.globalfloods.eu/) for details about other GloFAS products. 

__Related script(s)__: [Download GloFAS as .csv](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/main/analyses/bangladesh/scripts/d01_data/GetGLOFAS_data.py)

---

### 2. Flood Forecasting and Warning Centre (FFWC)

__Description__:  

__Format__: 

__Resolution__:

__Access__: Not openly accessible in machine readable format. 

__Availability__: Bangladesh.  

__Update frequency__:

__Limitations__: Specific to Bangladesh. Not publicly accessible. 

__Related script(s)__: 

---

### 3. Joint Research Centre Global Surface Water (JRC GSW)

__Description__: A collection of products that address different elements of surface water dynamics, from a historical perspective. Datasets show metrics including occurrence, recurrence, change intensity, and seasonality. Find further details through the [Data Users Guide](https://00f74ba44baafcc43182abbb5e11092807c424eccf-apidata.googleusercontent.com/download/storage/v1/b/global-surface-water/o/downloads_ancillary%2FDataUsersGuidev2019.pdf?jk=AFshE3WP6mlQo_K7Jf5An7ugsTvKKTaYhKQvIiwjjWfrNh1R9yyKBKutULA94kUJfUYNGkOXMjtYbmcGPNpTtkuRBUxOsxLApwAnkJxnARtmDox7rXtxrPAGDf-s9OK8PIZ09x_tRnRtMceofXLuKIa0ugNVur-4Qqpu1jVPVc2dxBISr3-FloLVpXhL62RlbZSt2PYgpgv3BPAe8cq-h90d5gNaaIYR6Hhc7vMkUEvmDITe7CnDUDcaeWOIJn8y--jps5-BEIpsyJdNuZvJWGxxEgYJXPgZCtAEMwNOkPJ2JBWbcW_MY_-_uQ0Hs2QvVv2QwQv30qlvz7GUWjcStJ_UIGXutIMKjECcP54Ed4QjEm1ksDE-7jRBCB7-hjIvrHNMcF-PcINI-KeNLeTxs-QYSoChNyYluiK365orH8vJBnUgi2rKDg5UMuffYAo4_fF4Ag-xGB_BoKmLSusWs-I607iq8-a1DqbUJJc-lexUMHatePGSioMKO2M8H_PHplUfuLVqi9ZpmafLY6PNQg-AURW8U_hiLbaP5ae5y0ZJVxHW5zQ8Detm5XLR90MfdWOvg-aqywwXly98T_YcWCqU4POMjP-EzR6TpxEhIrGQkYr3GGREXaJ6V0k6csIl8AyqzEfR0Lm9tTtPJTLboO2hsjui5ZQI7aGTrX5VPKni04XNazDfFnE0q6-FRMuVVS19k2HES7UKD7zJQc7WHTdaJf7xDELoZd4Hg1jn4o4C4Z8KhT40rucKlOahKA4-UEShoxEplaRyLpnKk5ygYkgHIw6CN26GNxi-sMKsz-AWT0X3TOAVuqAgYGmzl_XxHiDJUKkNTOSPGH2kXITokdhrgKKxZq7B1RYdCybGV9CWucvtlsAWNChJ1Te8FUpYo_R6Li-XyE2oYFkzem1sK7Rp5DM43B1aWekngkcK0rTNCTpsMhmPk_mHubQGfJK7vlgpu_Tz4Lg6AjJ5&isca=1).

__Format__: Raster.  

__Resolution__: 

__Access__: Available through [Google Earth Engine](https://developers.google.com/earth-engine/tutorials/tutorial_global_surface_water_01) or through [online download](https://global-surface-water.appspot.com/download). Also provided as WMS for access through websites or desktop GIS, for example.  

__Availability__: Global. Variable temporal attributes. Datasets developed from analysis of imagery from 1984-2019. Seasonality dataset is from 2019 specifically. 

__Update frequency__: N/A. 

__Limitations__: Not for real-time monitoring or forecasting. 

__Related script(s)__: [Get permanent water](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/main/utils_general/process_gsw_data.py)

---

### 4. Sentinel-1 synthetic aperture radar (SAR) imagery 

__Description__: The Sentinel-1 mission, developed by the European Space Agency (ESA) as part of the Copernicus Program, includes C-band synthetic aperture radar (SAR) imaging. Sentinel-1 data is thus not impeded by cloud cover and can be collected during both day and night. Sentinel-1 SAR data is frequently used to map flood extent. 

__Format__: Raster. 

__Resolution__: Approx 30 x 30m resolution.   

__Access__: The Sentinel-1 Ground Range Detected (GRD) product is available on [Google Earth Engine (GEE)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD). This dataset has already undergone the following preprocessing steps: 1) thermal noise removal, 2) radiometric calibration, and 3) terrain correction.

__Availability__: Global coverage since 2014. 

__Update frequency__: Variable, depending on location. Up to 12-day revisit frequency. 

__Limitations__: Not daily coverage. 

__Related script(s)__: [Flood extent processing](https://code.earthengine.google.com/46e61d848d78a074e69ed5fc4a7d1a2c)

---