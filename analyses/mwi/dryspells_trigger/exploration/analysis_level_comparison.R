## Generate comparison between computing metrics at cell, adm2 or adm1 levels

# setup
knitr::opts_chunk$set(echo = FALSE) # do not print code by default
knitr::opts_chunk$set(include = TRUE) # include chunk output by default
knitr::opts_chunk$set(message = FALSE) # do not print messages by default
knitr::opts_chunk$set(warning = FALSE) # do not print warnings by default

options(scipen = 999) # turn off scientific notation
options(encoding = "UTF-8") # set encoding to UTF-8 instead of ANSI


packages <- c('tidyverse', 'ggthemes', 'kableExtra', 'knitr', 'flextable', 'terra')

# install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages], repos = "https://cran.rstudio.com")
}

# load libraries 
lapply(packages, library, character.only = TRUE)

# set paths
data_dir <- Sys.getenv("AA_DATA_DIR")
shp_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")
postseason_folder_path <- paste0(data_dir, "/public/processed/mwi/dry_spells/2021_2022_postseason/")

# read in shapefiles for adm1 and adm2 in southern region
mwi_adm2 <- vect(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))
sr_adm2 <- subset(mwi_adm2, mwi_adm2$ADM1_EN == "Southern")

mwi_adm1 <- vect(paste0(shapefile_path, "/mwi_admbnda_adm1_nso_20181016.shp"))
sr_adm1 <- subset(mwi_adm1, mwi_adm1$ADM1_EN == "Southern")

# read in data
adm3_shp_all <- vect(paste0(shp_path, "/mwi_admbnda_adm3_nso_20181016.shp")) # for display purposes
adm3_shp <- subset(adm3_shp_all, adm3_shp_all$ADM1_EN == "Southern")

static_r <- rast(paste0(postseason_folder_path, "static_r.tif"))
data <- static_r[["rain_seas_tot"]]

# compute mean per adm
adm2_means <- terra::extract(data, sr_adm2, fun = mean)
sr_adm2$rain_means_adm2 <- as.numeric(adm2_means$rain_seas_tot)

adm1_means <- terra::extract(data, sr_adm1, fun = mean)
sr_adm1$rain_means_adm1 <- as.numeric(adm1_means$rain_seas_tot)

# create graphs
mean_values <- c(min(data$rain_seas_tot, sr_adm2$rain_means_adm2, sr_adm1$rain_means_adm1))

rain_min <- min(minmax(data[["rain_seas_tot"]])[1], min(sr_adm2$rain_means_adm2), min(sr_adm1$rain_means_adm1))
rain_min <- plyr::round_any(rain_min, 100, f=floor)

rain_max <- max(minmax(data[["rain_seas_tot"]])[2], max(sr_adm2$rain_means_adm2), max(sr_adm1$rain_means_adm1))
rain_max <- plyr::round_any(rain_max, 100, f=ceiling)

rain_breaks <- seq(from = rain_min, to = rain_max, by = 100)

par(mfrow = c(1,3))

plot(sr_adm1, 
     "rain_means_adm1", 
     breaks = rain_breaks, 
     main = "Rainfall per Adm1",
     axes = F,
     legend = NULL,
     cex.main = 1.3,
     col = "#41B6C4") # fourth value in the brewer.pal scale of other two graphs
plot(sr_adm2, add = T)

plot(sr_adm2, 
     "rain_means_adm2", 
     breaks = rain_breaks, 
     main = "Rainfall per Adm2",
     axes = F,
     legend = NULL,
     cex.main = 1.3,
     col = RColorBrewer::brewer.pal(n = 7, name = "YlGnBu"))
plot(sr_adm2, add = T)

plot(data, 
     "rain_seas_tot", 
     breaks = rain_breaks, 
     main = "Rainfall per Cell",
     axes = F,
     legend = "top",
     cex.main = 1.3,
     col = RColorBrewer::brewer.pal(n = 7, name = "YlGnBu"))
plot(sr_adm2, add = T)

# save plot
#ggsave(paste0(postseason_folder_path, "/comparison_analysis_levels.png"))


