# look at 2020 flood timelines across counties

library(tidyverse)
library(lubridate)
library(geofacet)

#################
#### LOADING ####
#################

data_dir <- Sys.getenv("AA_DATA_DIR")

df <- read_csv(
  file.path(
    data_dir,
    "private",
    "processed",
    "ssd",
    "floodscan",
    "ssd_floodscan_stats_adm2.csv"
  )
) %>%
  select(-1) %>%
  mutate(
    year = year(date),
    month = month(date)
  )

##############
#### GRID ####
##############

ssd_grid <- tribble(
  ~code, ~row, ~col,
  "Awerial", 5, 1,
  "Bor South", 5, 2,
  "Yirol East", 4, 1,
  "Twic East", 4, 2,
  "Panyijiar", 3, 1,
  "Duk", 3, 2,
  "Leer", 2, 1, 
  "Ayod", 2, 2,
  "Koch", 1, 1,
  "Fangak", 1, 2,
) %>%
  mutate(name = code)

###############
#### GRAPH ####
###############

df_plot <- df %>%
  filter(
    year == 2020,
    ADM2_EN %in% ssd_grid$name
  )

df_plot %>%
  ggplot(
    aes(
      x = date,
      y = mean_cell
    )
  ) +
  stat_smooth(
    geom = "area",
    span = 1/4,
    fill = "#ef6666"
  ) +
  scale_x_date(
    date_breaks = "3 months",
    date_labels = "%b"
  ) +
  facet_geo(
    ~ADM2_EN,
    grid = ssd_grid
  ) +
  theme_minimal()
