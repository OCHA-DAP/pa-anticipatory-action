# script to be sourced within analyses/uga/docs/uga_flood_risk.Rmd

# set paths
public_raw_glb_dir <- paste0(data_dir,'/public/raw/glb')
flood_risk_filename <- paste0(public_raw_glb_dir, "/ec_jrc/floodMapGL_rp10y/floodMapGL_rp10y.tif")
private_exploration_uga_ec_jrc_dir <- paste0(data_dir,'/private/exploration/uga/ec_jrc')

# read in raster
raster_data <- raster::raster(flood_risk_filename)

# crop and mask raster to uga adm3 boundaries
adm3_shp <- sf::st_read(uga_adm3_shapefile_path)
adm3_spatial_extent <- sf::st_bbox(adm3_shp)

data_cropped <- raster::crop(x = raster_data, y = adm3_spatial_extent)
data_masked <- raster::mask(data_cropped, mask = adm3_shp)

ec_flood_risk_map <- raster::plot(data_masked, axes = FALSE, box = FALSE)

# save as png
# png(paste0(private_exploration_uga_ec_jrc_dir, '/ec_jrc_uga_flood_risk_adm3_20220302.png'))
# raster::plot(data_masked,
#              main = "Flood Prone Areas\n 10-year return period",
#              xlim = c(29.57501, 35.00001),
#              ylim = c(-1.483236, 4.233431),
#              axes = FALSE,
#              box = FALSE,
#              legend = TRUE,
#              legend.args = list(text = 'Water Depth (m)',
#                                 font = 2,
#                                 line = 1,
#                                 cex = 0.8))
# raster::plot(adm3_shp$geometry, border = "grey", add = TRUE, lwd = 0.5)
# dev.off()
