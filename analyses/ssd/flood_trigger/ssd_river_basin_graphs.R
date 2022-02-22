# look at the use of exponential smoothing on flood extents
# in South Sudan to capture impact of residual flood levels
# from the previous year.

library(tidyverse)
library(janitor)
library(ggridges)
library(ggcorrplot)

#################
#### LOADING ####
#################

data_dir <- Sys.getenv("AA_DATA_DIR")
chirps_fp <- file.path(
  data_dir,
  "public",
  "raw",
  "ssd",
  "chirps"
)

# make monthly to match with rainfall
df_chirps <- map_dfr(
  list.files(chirps_fp),
  ~read_csv(
    file.path(
      chirps_fp,
      .x
    )
  ) %>%
    mutate(
      basin = str_extract(.x, "(?<=_)(.*?)(?= -)")
    )
) %>%
  clean_names() %>%
  group_by(basin, year, month) %>%
  summarize(
    across(
      ends_with("mm"),
      sum
    ),
    .groups = "drop"
  )
 
df_floodscan <- read_csv(
  file.path(
    data_dir,
    "public",
    "exploration",
    "ssd",
    "floods",
    "ssd_floodscan_roi.csv"
  )
) %>%
  select(-1) %>%
  group_by(year, month) %>%
  summarize(
    flood_extent = mean(mean_ADM0_PCODE),
    .groups = "drop"
  )

#################
#### COMPARE ####
#################

# compare CHIRPS and flood extents, just simple
# monthly comparison
# first join the data
df_join <- inner_join(
  df_chirps,
  df_monthly,
  by = c("year", "month")
) 

# get correlations

df_join %>%
  mutate(time = paste("y", year, month, sep = "_")) %>%
  select(basin, time, rainfall_mm) %>%
  pivot_wider(
    time,
    names_from = "basin",
    values_from = "rainfall_mm"
  ) %>%
  select(-time) %>%
  cor() %>%
  ggcorrplot(lab = TRUE)

# get worst years
df_join %>%
  bind_rows(
    df_join %>%
      group_by(year, month) %>%
      summarize(
        basin = "Total",
        across(
          ends_with("mm"),
          sum
        ),
        flood_extent = unique(flood_extent),
        .groups = "drop"
      )
  ) %>%
  pivot_wider(
    c(basin, year, month, flood_extent),
    names_from = basin,
    values_from = rainfall_mm
  ) %>%
  group_by(year) %>%
  summarize(
    flood_extent = max(flood_extent),
    across(
      `Bahrel Ghazal`:Total,
      sum
    )
  ) %>%
  pivot_longer(-year) %>%
  group_by(
    name
  ) %>%
  slice_max(value, n = 5) %>%
  right_join(
    expand_grid(
      year = 1998:2021,
      name = unique(.[["name"]])
    )
  ) %>%
  ungroup() %>%
  mutate(
    worst_year = case_when(
      is.na(value) ~ "not_worst",
      name != "flood_extent" ~ "rainfall",
      TRUE ~ "flood"
    ),
    name = ifelse(name == "flood_extent", "Flood extent", name),
    name = fct_relevel(
      name,
      levels = c(
        "Total",
        "Bahrel Ghazal",
        "Bameel Jable Sudd",
        "LakeKyoga-Lake Albert",
        "Pibor-Akabo-Sobat- Level 6",
        "Flood extent"
      )
    )
  ) %>%
  ggplot(
    aes(
      x = year,
      y = name,
      fill = worst_year
    )
  ) +
  geom_tile() +
  theme_light() +
  scale_fill_manual(
    values = c("#ed4d4d", "white", "#81d4fa")
  ) +
  theme(
    legend.position = "blank"
  ) +
  labs(
    x = "Year",
    y = "Rainfall and Sudd flood extent",
    title = "Comparison of largest yearly flooding with most rainfall by basin"
  )

