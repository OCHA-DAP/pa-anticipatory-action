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

__Description__: A collection of products that address different elements of surface water dynamics, from a historical perspective. Datasets show metrics including occurrence, recurrence, change intensity, and seasonality. Find further details through the [Data Users Guide](https://00f74ba44b6beb5706df0b6f3a5eefe78972d14410-apidata.googleusercontent.com/download/storage/v1/b/global-surface-water/o/downloads_ancillary%2FDataUsersGuidev2019.pdf?jk=AFshE3WcaUobfExgnU6qVPsSSXo9rogNoXppKfJrDOPg0xmHXjThwPmFKiZKOJTq1OgK1VCpEnCtMAf0XX7S-zQaeq8C9YjfvTdmofXU73ZwdNcpHuBBbc-48TnMtUtMeUUIi5wmoMbPVvXR65ac4yuMuV-4evGbOlYh6WtXhlBkmA3mLYuK79RsnKmbzqJPcuCG8HP9wDeo6sx4JC-qBnVyyuTpSIi2GBqTh2A1tQpFYO8cJcieMdnQ7MuP_ScKA9rts7UF9MvJ1qa98usCWswcNfDGUpRCGFzCzUP9V5oINe-guP8cQ3FMGYp8_7ESs25lqN-8utIZh_rMGYGbvMog2JGVTZkJaDow-_rJMp2Wth_HGfGk_CHRB_EwriPq2YOlgolYKXIYzO19uWIQuKy5jZhK8nIcW6enHGJaHfYSWZN26fBLHI7wECp1GFC1KFvRJoviipbauohq0VUxfZPk3UppuKIpibuiIFyB6dRAegfJyzzWKp2aHmPtsufS-rZrome1IulxxRJO7S0xjaX8BKLZbUEJupCe7AtksjYnvZN1TrCNOmm8spHkslMk7VKTxDV6GSFAHxl_ou_Ngjk3EeOdUP6I1FFex1PecpfyjfasRpDk-rxkbVFM-_MgTNeiqMu9UvCNNWKUZqoPKWbsvQqPQH8rUdcQzdhfF8xNEXI2NlqGbAcPQMSvHWk5Oi7VXu6yWXgKuO8Fo911qyUlV6tppRt8PlfbDda6hwEv157SqJQijczNpQGkxLOUrj-473ZHO3ApQ2yGuHh1x1__38xKAydKjNCwL5ipPm1BlwhBrpezXsoSYSgI7qZNfa5yIaL5KjQ-myon2hoVmOXiu_YUDsGDde9CRO5VU5qzpBRQHN-69gmrDTNSv3jSgU780pU0lVSe43GpksWcZ6xNqZ7edIfJTIlXcmZfe8YWF-pXrUELyOkfkcLxsQCvs-BX2rG0FwlR8Dgp&isca=1).

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