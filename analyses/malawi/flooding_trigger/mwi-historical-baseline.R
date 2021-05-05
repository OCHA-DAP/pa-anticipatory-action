library(dplyr)
library(sf)
library(lubridate)
library(readxl)
library(yaml)

# Get overall config params and read in data
data_dir <- Sys.getenv("AA_DATA_DIR")
data_private_dir <- Sys.getenv('AA_DATA_PRIVATE_DIR')
config <- read_yaml('../../src/malawi/config.yml')
shp_adm2 <- st_read(paste0(data_dir, '/raw/malawi/Shapefiles/', config$path_admin2_shp))
shp_adm3 <- st_read(paste0(data_dir, '/raw/malawi/Shapefiles/', config$path_admin3_shp))
df_mvac_floods <- read.csv(paste0(data_private_dir, "/processed/malawi/mvac_dodma_flood_info.csv"))
stations <- config$glofas$stations
stations <- data.frame(do.call(rbind.data.frame, stations))

# Get the TAs associates with each of the GloFAS stations
shp_stations <- stations %>%
  st_as_sf(coords = c("long", "lat"), crs = config$crs_degrees) %>%
  st_join(shp_adm3) %>%
  select('ADM2_EN', 'ADM3_EN', 'name') %>%
  st_set_geometry(NULL) %>%
  mutate(ADM3_EN = substr(ADM3_EN, 4, nchar(ADM3_EN)))

# MVAC Flooding from DoDMA ------------------------------------------------

# Filter the MVAC data by the TAs that have GloFAS stations
# TA names are not unique so we need to make sure that those selected 
# are in either Chikwawa or Nsanje
df_mvac_floods_ta <- df_mvac_floods %>%
  filter(TA %in% shp_stations$ADM3_EN & District %in% c('Chikwawa', 'Nsanje'))%>%
  full_join(shp_stations, by=c('TA'='ADM3_EN'))

# Now filter the MVAC data by the districts that have GloFAS stations 
# This is a bit of a looser definition of 'impact', as the flooding might 
# not be close to the GloFAS stations that we're monitoring
df_mvac_floods_district <- df_mvac_floods %>%
  filter(District %in% c('Chikwawa', 'Nsanje'))%>%
  full_join(shp_stations, by=c('District'='ADM2_EN'))

write.csv(df_mvac_floods_district, paste0(data_private_dir, '/processed/malawi/mvac_dodma_flood_district.csv'), row.names = FALSE)
write.csv(df_mvac_floods_ta, paste0(data_private_dir, '/processed/malawi/mvac_dodma_flood_ta.csv'), row.names = FALSE)


