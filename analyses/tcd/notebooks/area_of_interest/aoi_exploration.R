library(tidyverse)
library(sf)
library(terra)
library(rasterVis)

aa_dir <- Sys.getenv("AA_DATA_DIR")

sf <- read_sf(file.path(
  aa_dir, "public", "raw", "tcd", "cod_ab", "tcd_admbnda_adm2_ocha_20170615", "tcd_admbnda_adm2_ocha_20170615.shp"
))

df <- readxl::read_excel(file.path(
  aa_dir, "public", "raw", "tcd", "drought", "tcd_priority_adm2.xlsx"
),
sheet = "Sheet1")

# fixing names

df <- transmute(df,
             Departement = case_when(
               Departement == "Sud Kanem" ~ "Wadi-Bissam",
               Departement == "IRIBA" ~ "Kobé",
               Departement == "Dar Tama" ~ "Dar-Tama",
               TRUE ~ str_replace(Departement, "El-Gazal", "El-Gazel")
             ),
             area_of_interest = TRUE)

sf <- left_join(sf, df, by = c("admin2Name" = "Departement"))
sf_aoi <- filter(sf, area_of_interest)

# raster

dr <- rast(file.path(
  aa_dir, "public", "raw", "glb", "biomasse", "BiomassValueMean.tif"
))

dr <- crop(dr, sf_aoi)

# plot it

dr %>%
  as.data.frame(xy=TRUE) %>%
  filter(BiomassValueMean >= 0) %>%
  ggplot() +
  geom_raster(aes(x = x, y = y, fill = BiomassValueMean)) +
  scale_fill_gradient(low = "white", high = "#00FF00") +
  geom_sf(data = sf, fill = NA, color = "lightgrey") +
  geom_sf(data = sf_aoi, fill = NA, color = "darkgrey")


dr %>%
  as.data.frame(xy=TRUE) %>%
  filter(BiomassValueMean == 0) %>%
  ggplot() +
  geom_raster(aes(x = x, y = y, fill = BiomassValueMean)) +
  scale_fill_gradient(low = "white", high = "#00FF00") +
  geom_sf(data = sf, fill = NA, color = "lightgrey") +
  geom_sf(data = sf_aoi, fill = NA, color = "darkgrey")
