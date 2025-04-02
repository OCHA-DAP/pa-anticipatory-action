library(tidyverse)
library(sf)
library(raster)
library(tmap)

###################
#### SET PATHS ####
###################

dd <- Sys.getenv("AA_DATA_DIR")

arc2_path <- file.path(
  dd,
  "public",
  "exploration",
  "mwi",
  "arc2",
  "example_arc2_rollsum_ds.nc"
)

shp_path <- file.path(
  dd,
  "public",
  "raw",
  "mwi",
  "cod_ab",
  "mwi_adm_nso_20181016_shp",
  "mwi_admbnda_adm2_nso_20181016.shp"
)

save_path <- file.path(
  dd,
  "public",
  "processed",
  "mwi",
  "plots",
  "dry_spells",
  "arc2",
  "example_raster_ds_map.png"
)

###################
#### LOAD DATA ####
###################

arc2_df <- raster(arc2_path,
                 crs = "EPSG:4326")

adm_sf <- read_sf(shp_path)
arc2_df <- crop(arc2_df, adm_sf)

####################
#### CREATE MAP ####
####################

# Map the SF

p <- adm_sf %>%
  tm_shape() +
  tm_polygons() +
  tm_shape(arc2_df) +
  tm_raster(alpha = 0.5,
            style = "cat",
            palette = c(NA, "#151515"),
            title = "Raster dry spells") +
tm_layout(main.title = "Dry spells",
          main.title.size = 1.5,
          frame = F)

tmap_save(p, save_path)
