library(chirps)
library(AOI)
library(climateR)
library(sf)
library(raster)
library(rasterVis)
library(dplyr)
library(exactextractr)

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- shapefiles_path <-  paste0(data_dir, "/raw/malawi/Shapefiles/mwi_adm_nso_20181016_shp")
mwi_adm2 <- st_read(paste0(shapefiles_path, "/mwi_admbnda_adm2_nso_20181016.shp"))


# 1. Testing the climateR package -----------------------------------------

# This seems like close to what we need
# https://github.com/mikejohnson51/climateR/issues/2

# --- These all throw the 'path too long' error...
# mwi <- aoi_get(sf::read_sf(paste0(shapefiles_path, "/mwi_admbnda_adm2_nso_20181016.shp")))
# mwi <- st_read(paste0(shapefiles_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
# mwi <- aoi_get(country = "Malawi") 
# mwi <- aoi_get(country = "Brazil") # trying another country...
mwi = aoi_get(country = "Kenya") # WHY IS THIS THE ONLY ONE THAT WORKS?? Canada also works...
plot(mwi$geometry)

# It takes approx 4 mins to get a year's worth of gridded data for a country
start.time <- Sys.time()
s = getCHIRPS(mwi, 
    startDate = "2018-06-01",
    endDate = "2019-06-03")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


# 2. Testing the chirps package -------------------------------------------

# To begin, reproducing example from:
# https://www.theoj.org/joss-papers/joss.02419/10.21105.joss.02419.pdf

# Getting 10 years worth of daily values for 10 points within an area of interest
data("tapajos",package ="chirps")
set.seed(1234)
tp <-st_sample(tapajos,20)
tp <-st_as_sf(tp)
dt <-get_chirps(tp,dates =c("2008-01-01","2018-01-31")) # This just takes a couple of minutes
p_ind <-precip_indices(dt,timeseries =TRUE,intervals =30)
