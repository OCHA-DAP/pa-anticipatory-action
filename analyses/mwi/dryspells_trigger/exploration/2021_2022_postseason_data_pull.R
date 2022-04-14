# process precipitation raster files for postseason analysis of 2021-2022 season
# used to generate data files that load into 2021_2022_postseason_overview.Rmd

#####
## setup
#####

# load libraries
packages <- c('tidyverse', 'sf', 'terra', 'zoo')
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

lapply(packages, library, character.only = TRUE)

# set options
options(scipen = 999)

# set directory paths
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")
arc2_filepath <- paste0(data_dir, "/public/raw/mwi/arc2/arc2_daily_precip_mwi_32E_36E_20S_5S_main.nc")
dry_spell_processed_path <- paste0(data_dir, "/public/processed/mwi/dry_spells/")

# prep to mask area outside of MWI & Southern region
mwi_adm1_vect <- vect(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))
southern_vect <- mwi_adm1_vect[mwi_adm1_vect$ADM1_EN == 'Southern',]

#####
## process ARC2 observational rainfall data (already cropped)
#####

# read in arc2 files (full history, cropped to MWI) and crop to Southern region
raw <- rast(arc2_filepath) 
res(raw)

cropped <- crop(raw, southern_vect, mask = T)

# subset period of interest
start_date <- 7959 #7958 = number of days between 19 dec 1999 and 1 oct 2021
end_date <- 8129 #8140 = number of days between 19 dec 1999 and 1 apr 2022  ###FIX ME ONCE DATA UP TO 1 APR AVAILABLE
date_numbers <- seq(from = start_date, to = end_date, by = 1)
dates_chr <- paste0("est_prcp_T=", date_numbers)
  
data_r <- subset(cropped, dates_chr)

# extract values
cell_numbers <- cells(data_r, 
                      southern_vect, 
                      touches = T, # all cells touched by polygons are extracted not just those whose center point is within the polygon
                      exact = T) # weights =  exact fraction of each cell that is covered

data <- extract(data_r, 
                southern_vect,
                touches = T,
                cells = T, # return cell numbers
                xy = T) # return cell coordinates

data <- data %>%
          select(ID, cell, x, y, everything()) # move cell number + coordinates columns to first positions

# saveRDS(data, paste0(dry_spell_processed_path, "2021_2022_postseason/", "mwi_2021_2022_postseason_overview.RDS"))
