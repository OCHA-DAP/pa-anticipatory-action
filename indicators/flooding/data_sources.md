# Data sources

### 1. Global Flood Awareness System (GloFAS) modelled river discharge

__Description__: Daily modelled river discharge data. Provides modelled estimates of the amount of water flowing through a given river section. Average over each 24 hour period. Simulations are run based on a hydrological river routing model with modelled gridded runoff data from global reanalysis. 

__Format__: Raster. 0.1 x 0.1 degree resolution. 

__Access__: [Via API](https://cds.climate.copernicus.eu/cdsapp#!/dataset/cems-glofas-historical?tab=overview). 

__Coverage__: Daily global data since 1 January 1979. 

__Limitations__: Modelled, rather than historical river discharge. See [this documentation](https://www.globalfloods.eu/) for details about other GloFAS products. 

__Related script(s)__: [Download GloFAS as .csv](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/main/analyses/bangladesh/scripts/d01_data/GetGLOFAS_data.py)

---

### 2. Flood Forecasting and Warning Centre (FFWC)

__Description__:  

__Format__:  

__Access__: 

__Coverage__: Bangladesh.  

__Limitations__: Specific to Bangladesh. Not publicly accessible. 

__Related script(s)__: 

---

### 3. Joint Research Centre Global Surface Water (JRC GSW)

__Description__: A collection of products that address different elements of surface water dynamics, from a historical perspective. Datasets show metrics including occurrence, recurrence, change intensity, and seasonality. 

__Format__: Raster.  

__Access__: Available through [Google Earth Engine](https://developers.google.com/earth-engine/tutorials/tutorial_global_surface_water_01) or through [online download](https://global-surface-water.appspot.com/download). 

__Coverage__: Global.   

__Limitations__:

__Related script(s)__: [Get permanent water](https://github.com/OCHA-DAP/pa-anticipatory-action/blob/main/utils_general/process_gsw_data.py)

---

### 4. Sentinel-1 synthetic aperture radar (SAR) imagery 

__Description__: The Sentinel-1 mission, developed by the European Space Agency (ESA) as part of the Copernicus Program, includes C-band synthetic aperture radar (SAR) imaging. Sentinel-1 data is thus not impeded by cloud cover and can be collected during both day and night. Sentinel-1 SAR data is frequently used to map flood extent. 

__Format__: Raster. Approx 30 x 30m resolution.   

__Access__: The Sentinel-1 Ground Range Detected (GRD) product is available on [Google Earth Engine (GEE)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD). This dataset has already undergone the following preprocessing steps: 1) thermal noise removal, 2) radiometric calibration, and 3) terrain correction.

__Coverage__: Global coverage. Available at up to 12 day intervals since 2014. 

__Limitations__: Not daily coverage. 

__Related script(s)__: [Flood extent processing](https://code.earthengine.google.com/46e61d848d78a074e69ed5fc4a7d1a2c)

---