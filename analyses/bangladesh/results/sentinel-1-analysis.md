## Analysing Sentinel-1 satellite imagery to identify flood extent over time

### Related script(s)

[Google Earth Engine script](https://code.earthengine.google.com/46e61d848d78a074e69ed5fc4a7d1a2c)

### Background to Sentinel-1

The Sentinel-1 mission, developed by the European Space Agency (ESA) as part of the Copernicus Program, includes C-band synthetic aperture radar (SAR) imaging. Sentinel-1 data is thus not impeded by cloud cover and can be collected during both day and night. More details about SAR imagery can be found [here](https://earthdata.nasa.gov/learn/backgrounders/what-is-sar). This mission is made up of two satellites that each follow either ascending or descending path directions. Each satellite has a revisit time of 12 days. With the two satellites together, many areas are revisted at 6-day intervals.  

The Sentinel-1 Ground Range Detected (GRD) product is [available on Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) (GEE). This dataset has already undergone the following preprocessing steps: 1) thermal noise removal, 2) radiometric calibration, and 3) terrain correction. 

### Processing methodology

Sentinel-1 imagery has been frequently applied to identify flood extent for a given area of interest ([for example](https://www.mdpi.com/2073-4441/11/12/2454/htm)). The change-detection methodology applied in this analysis is largely derived from [UN-SPIDER guidance](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping/step-by-step).

This processing was conducted in GEE using Javascript. As a cloud platform, GEE allows for relatively fast processing of large volumes of satellite imagery, with simple access to the necessary Sentinel-1 imagery. GEE is also freely accessible for nonprofit use, with sign-up available [here](https://signup.earthengine.google.com/#!/). In addition to the Sentinel-1 imagery, it was also required to upload a shapefile to delineate the area of interest (in this case accessed from HDX). 

### Results 

For each date of imagery coverage between June and August, a shapefile delineating flood extent (excluding permanent water bodies) was created. We then combined this data with administrative unit shapefiles to calculate the percent of flooded area per union (admin 4 level) over time. 