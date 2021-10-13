library(tidyverse)
library(sf)
library(tmap)

###################
#### SET PATHS ####
###################

dd <- Sys.getenv("AA_DATA_DIR")

precip_path <- file.path(
  dd,
  "public",
  "exploration",
  "mwi",
  "arc2",
  "mwi_arc2_precip_long_raw.csv"
)

shp3_path <- file.path(
  dd,
  "public",
  "raw",
  "mwi",
  "cod_ab",
  "mwi_adm_nso_20181016_shp",
  "mwi_admbnda_adm3_nso_20181016.shp"
)

shp2_path <- file.path(
  dd,
  "public",
  "raw",
  "mwi",
  "cod_ab",
  "mwi_adm_nso_20181016_shp",
  "mwi_admbnda_adm2_nso_20181016.shp"
)

save_path <- file.path(
  dd,
  "public",
  "processed",
  "mwi",
  "plots",
  "dry_spells",
  "arc2",
  "fake_precip_map.png"
)

###################
#### LOAD DATA ####
###################

precip_df <- read_csv(precip_path) %>%
  select(ADM2_PCODE,
         date,
         mean_cell)

adm3_sf <- read_sf(shp3_path) %>%
  filter(ADM1_PCODE == "MW3")
adm2_sf <- read_sf(shp2_path) %>%
  filter(ADM1_PCODE == "MW3")

####################
#### CREATE MAP ####
####################

# Create SF with precipitation data
# for 2018 monitoring period up until
# the trigger would have been met

precip_sf <- precip_df %>%
  filter(date >= "2017-12-24",
         date <= "2018-01-31") %>%
  group_by(ADM2_PCODE) %>%
  summarize(precip = sum(mean_cell),
            .groups = "drop") %>%
  right_join(adm3_sf, by = "ADM2_PCODE") %>%
  mutate(precip = precip + rnorm(nrow(.), sd = 20)) %>%
  st_as_sf()

# Map the SF

p <- precip_sf %>%
  tm_shape() +
  tm_fill(
    col = "precip",
    palette = "Blues",
    title = "Cumulative rainfall, mm, Dec. 24 - Jan. 31",
    style = "cont") +
  tm_shape(adm2_sf) +
  tm_borders(lwd=0.5) +
  tm_text("ADM2_EN") +
  tm_layout(main.title = "Rainfall and dry spells, 2018",
            main.title.size = 1,
            frame = F)

p

tmap_save(p, save_path)
