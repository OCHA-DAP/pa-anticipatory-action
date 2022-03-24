# process CHIRPS raster files for postmortem of 2021-2022 season
# used to generate processed data files that load into chirps_2021_2022_postseason_overview.Rmd

#####
## setup
#####

# load libraries
packages <- c('tidyverse', 'sf', 'raster')
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

lapply(packages, library, character.only = TRUE)

# load functions
source("../dryspells_trigger/01b_chirps_dry_spells_functions.R")

# set options
rasterOptions(maxmemory = 1e+09)
options(scipen = 999)

# set directory paths
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")
chirps_path <- paste0(data_dir, "/public/raw/glb/chirps/")
dry_spell_processed_path <- paste0(data_dir, "/public/processed/mwi/dry_spells/")

#####
# create list of adm3's
#####

#adm3_list <- mwi_adm3[, c("ADM3_PCODE", "ADM3_EN", "geometry")]
#adm3_names <- as.data.frame(adm3_list) %>%
#  dplyr::select(-geometry)

#####
## process observational rainfall data (CHIRPS)
#####

# read in shapefile
mwi_adm1 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))

# read in CHIRPS data (multiple multi-layer raster files) into a single stack
s2021 <- raster::stack(paste0(chirps_path, "chirps_global_daily_2021_p05.nc"))
s2022 <- raster::stack(paste0(chirps_path, "chirps_global_daily_2022_p05.nc")) 

s2021_s2022 <- stack(s2021, s2022) # all files combined into a stack

# crop and masked area outside of MWI
mwi_adm1_spatial_extent <- st_bbox(mwi_adm1)
mwi_adm1_ids <- as.data.frame(mwi_adm1) %>% dplyr::select("ADM1_PCODE", "ADM1_EN")

s2021_s2022_cropped <- crop(x = s2021_s2022, y = extent(mwi_adm1_spatial_extent))
data_masked <- mask(s2021_s2022_cropped, mask = mwi_adm1)

# select time window of interest (Oct 2021 - March 2022)
window <- seq(from = as.Date('2021.10.01', format = '%Y.%m.%d'), 
              to = as.Date('2022.05.01', format = '%Y.%m.%d'), 
              by = 'day')
window <- str_replace_all(window, "-", ".")
window <- paste("X", window, sep = "")

# subset dataset for time window and Southern region
data <- subset(data_masked, window)


# saveRDS(data_masked, paste0(dry_spell_processed_path, "mwi_2021_2022_overview_r5.RDS"))) # 5-deg resolution
