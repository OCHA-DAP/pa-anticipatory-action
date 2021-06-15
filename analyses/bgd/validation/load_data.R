data_dir <- Sys.getenv("AA_DATA_DIR")
df_summ <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/FE_Results/June_Aug/MAUZ_flood_summary_QA.csv'))
df_sent <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/FE_Results/June_Aug/MAUZ_flood_extent_sentinel.csv'))
df_gaus <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/FE_Results/June_Aug/MAUZ_flood_extent_interpolated.csv'))
shp_mauz <- st_read(paste0(data_dir, '/exploration/bangladesh/ADM_Shp/selected_distict_mauza.shp'))%>%
  st_transform('EPSG:4326')
df_glofas <- read.csv(paste0(data_dir, '/exploration/bangladesh/GLOFAS_Data/2020.csv'))
stations <- st_read(paste0(data_dir, '/exploration/bangladesh/GLOFAS_Data/stations.shp'))%>%
  st_transform('EPSG:4326')%>%
  filter(Station %in% c('Bahadurabad', 'Chilmari', 'Kazipur', 'Sariakandi'))

adm0 <- st_read(paste0(data_dir, '/exploration/bangladesh/ADM_Shp/bgd_admbnda_adm0_bbs_20201113.shp'))
adm2 <- st_read(paste0(data_dir, '/exploration/bangladesh/ADM_Shp/bgd_admbnda_adm2_bbs_20201113.shp'))
shp_river <- st_read(paste0(data_dir, '/exploration/bangladesh/ADM_Shp/river_extent.shp'))

df_sent_adm4 <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/FE_Results/June_Aug/ADM4_flood_extent_sentinel.csv'))
df_gaus_adm4 <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/FE_Results/June_Aug/ADM4_flood_extent_interpolated.csv'))
df_int <- read.csv(file = paste0(data_dir, '/exploration/bangladesh/CDP_Informant/Sentinel-1-BGD-Flooding_INTERVIEW.csv'))