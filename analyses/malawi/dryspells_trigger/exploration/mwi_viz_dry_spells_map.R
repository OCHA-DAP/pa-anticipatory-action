
#####
## setup
#####

# load libraries
library(tidyverse)
library(sf)
library(raster)

# set options
rasterOptions(maxmemory = 1e+09)
options(scipen = 999)

# set directory paths
# AA_DATA_DIR is set as a variable in .Renviron or .bashprofile
data_dir <- Sys.getenv("AA_DATA_DIR")
shapefile_path <- paste0(data_dir, "/public/raw/mwi/cod_ab/mwi_adm_nso_20181016_shp")
plot_path <- paste0(data_dir, "/public/processed/mwi/plots/dry_spells/dry_spell_plots")

# import dry spells groundtruth dataset
dry_spells_groundtruth_dataset_url <- "https://data.humdata.org/dataset/df68065d-704c-4556-b111-394df000cee4/resource/951c0454-23ab-4057-a7fb-7cfeed4f842d/download/full_list_dry_spells.csv"
ds <- read.csv(dry_spells_groundtruth_dataset_url)

# read in country shapefiles
mwi_adm2 <- st_read(paste0(shapefile_path, "/mwi_admbnda_adm2_nso_20181016.shp"))

#####
## map  historical dry spells
#####

# summarise rainy-season dry spells frequency per district / year
ds_counts_per_year <- ds %>%
  filter(during_rainy_season == 1) %>% # excludes dry spells outside the rainy season or in rainy seasons without confirmed onset/cessation dates
  group_by(pcode, during_rainy_season) %>%
  summarise(n_ds = n())

# summarise rainy-season dry spells per district (across years). Note: only reports districts that have experienced rainy-season DSs
ds_counts <- ds %>%
  filter(during_rainy_season ==1) %>%
  group_by(pcode) %>%
  summarise(n_ds = n())

# combine dry spells groundtruth dataset and shapefiles
data <- mwi_adm2 %>%
  left_join(ds_counts, by = c("ADM2_PCODE" = "pcode"))

# create map
plt_ds_map <- data %>%
  ggplot() +
  geom_sf(aes(fill = n_ds)) +
  scale_fill_continuous("Total number of events", trans = 'reverse') +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  ggtitle("Dry spells in Malawi", subtitle = "2000-2020") +
  theme(
    legend.position = "right",
    plot.title = element_text(size = 16),
    plot.subtitle = element_text(size = 12),
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.title.x=element_blank(),
    axis.text.x=element_blank(),
    axis.ticks = element_blank(),
    axis.ticks.x = element_blank(),
    panel.background = element_rect(fill = "white"),
    panel.grid = element_blank()
  )

# save plot
# ggsave(paste0(plot_path, "/ds_count_country_map.png"), plot = plt_ds_map, dpi = 1200)