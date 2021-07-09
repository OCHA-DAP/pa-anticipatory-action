library(dplyr)
library(sf)

# Setup -------------------------------------------------------------------

data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/raw/malawi/Shapefiles/mwi_adm_nso_20181016_shp")

shp_adm1 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))

wrsi_dir <- paste0(data_dir, '/exploration/malawi/wrsi/')
wrsi_files <- list.files(path=wrsi_dir, pattern = "do.bil")
wrsi_files <- paste0(wrsi_dir, wrsi_files)


# Process WRSI ------------------------------------------------------------

wrsi_list <- stack(wrsi_files)
crs(wrsi_list) <- CRS('+init=EPSG:4326')

wrsi_list[wrsi_list > 100] <- NA

computeLayerStat <- function(data, layer, stat, data_stat_values, adm){
  data_layer <- subset(data, layer)
  data_layer.stat <- raster::extract(data_layer, adm, fun = stat, df = T, na.rm=T)
  data_stat_values <- merge(data_stat_values, data_layer.stat, by = "ID", all.x = T)
  return(data_stat_values)
}

wrsi_min_adm1 <- data.frame(ID = 1:nrow(shp_adm1))

for (i in 1:nlayers(wrsi_list)) {
  wrsi_min_adm1 <- computeLayerStat(wrsi_list, i, min, wrsi_min_adm1, shp_adm1)
}

# Clean and save the output -----------------------------------------------

# Get mean WRSI by dekad for adm1 regions
wrsi_min <- wrsi_min_adm1 %>%
  mutate(ID = shp_adm1$ADM1_EN)%>%
  gather(date, wrsi, -ID) %>%
  mutate(year = substr(date, 2, 5)) %>%
  mutate(dekad = substr(date, 6,7)) %>%
  dplyr::select(-date) 

# Write out to csv to save results 
write.csv(wrsi_min, paste0(wrsi_dir, 'wrsi_min_adm1.csv'))
