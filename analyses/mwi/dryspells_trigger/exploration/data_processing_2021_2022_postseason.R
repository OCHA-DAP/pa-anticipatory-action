# process precipitation raster files for postseason analysis of 2021-2022 season
# used to generate data files that load into 2021_2022_postseason_overview.Rmd

#####
## setup
#####

# load libraries
packages <- c('tidyverse', 'sf', 'terra')
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

lapply(packages, library, character.only = TRUE)

# set options
rasterOptions(maxmemory = 1e+09)
options(scipen = 999)

# set directory paths
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")
arc2_filepath <- paste0(data_dir, "/public/raw/mwi/arc2/arc2_daily_precip_mwi_32E_36E_20S_5S_main.nc")
#dry_spell_processed_path <- paste0(data_dir, "/public/processed/mwi/dry_spells/")

# prep to mask area outside of MWI & Southern region
mwi_adm1_vect <- terra::vect(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))
southern_vect <- mwi_adm1_vect[mwi_adm1_vect$ADM1_EN == 'Southern',]

#####
## process ARC2 observational rainfall data (already cropped)
#####

raw <- terra::rast(arc2_filepath) 
arc2_terra_masked <- terra::mask(raw, mask = southern_vect)

arc2 <- subset(arc2_terra_masked, start_date:end_date)
start_date <- 7959 #7958 = number of days between 19 dec 1999 and 1 oct 2021
#end_date <- 8141 #8140 = number of days between 19 dec 1999 and 1 apr 2022  
end_date <- 8130 ###FIX ME ONCE DATA UP TO 1 APR AVAILABLE

plot(arc2[[start_date:end_date]]) 

x <- raster::extract(arc2, cellnumbers = T, df = T, nl = nlayers(masked))

data.frame(values(arc2)) -> r

# saveRDS(data_masked, paste0(dry_spell_processed_path, "mwi_2021_2022_overview_r5.RDS"))) # 5-deg resolution
