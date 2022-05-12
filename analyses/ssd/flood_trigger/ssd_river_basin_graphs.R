# look at the use of exponential smoothing on flood extents
# in South Sudan to capture impact of residual flood levels
# from the previous year.

library(tidyverse)
library(janitor)
library(ggridges)
library(ggcorrplot)
library(ggtext)

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
    "private",
    "exploration",
    "ssd",
    "floodscan",
    "ssd_floodscan_roi.csv"
  )
) %>%
  select(-1) %>%
  group_by(year, month) %>%
  summarize(
    time = min(time),
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
  df_floodscan,
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
df_wider <- df_join %>%
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
    id_cols = c("year", "month", "flood_extent"),
    names_from = basin,
    values_from = rainfall_mm
  )

df_worst <- df_wider %>%
  filter(year < 2022) %>%
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
  slice_max(value, n = 5) 

df_worst %>%
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

# look at z-scores to see differences across years

df_wider %>%
  group_by(year) %>%
  filter(year < 2022) %>%
  summarize(
    flood_extent = max(flood_extent),
    across(
      `Bahrel Ghazal`:Total,
      sum
    ),
    .groups = "drop"
  ) %>%
  pivot_longer(
    -year
  ) %>%
  group_by(
    name
  ) %>%
  mutate(
    value = (value - mean(value)) / sd(value),
  ) %>%
  ungroup() %>%
  left_join(
    mutate(
      df_worst,
      worst = TRUE
    ) %>%
      select(-value)
  ) %>%
  mutate(
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
    ),
    worst = ifelse(
      is.na(worst),
      "",
      "H"
    )
  ) %>%
  filter(
    name %in% c("Total", "Flood extent")
  ) %>%
  mutate(
    name = ifelse(name == "Total", "Total rainfall\n(Sudd and neighboring river basins)", "Sudd flood extent")
  ) %>%
  ggplot(
    aes(
      x = year,
      y = name,
      fill = value
    )
  ) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradient2() +
  labs(
    x = "Year",
    y = "",
    fill = "Z-score",
    title = "Standardized comparison of rainfall and flood values",
    caption = "**H** indicates it's one of the 5 highest years on record, for either flooding or rainfall"
  ) +
  theme(
    plot.caption = element_markdown()
  ) +
  geom_text(
    aes(
      label = worst
    ),
    size = 2,
    fontface = "bold"
  )


# graph time series

brks <- df_floodscan %>%
  filter(
    month == 1,
    year %in% c(2000, 2005, 2010, 2015, 2020, 2021, 2022)
  ) %>%
  pull(time)

df_floodscan %>%
  ggplot(
    aes(
      x = time,
      y = flood_extent
    )
  ) +
  geom_area(fill = "#ef6666") +
  theme_minimal() +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_x_date(
    breaks = brks,
    date_labels = "%Y"
  ) +
  labs(
    x = "Year",
    y = "% of areas flooded",
    title = "Flooded areas in the Sudd wetlands, South Sudan"
  )
