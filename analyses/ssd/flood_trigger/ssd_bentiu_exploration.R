# look at the use of exponential smoothing on flood extents
# in South Sudan to capture impact of residual flood levels
# from the previous year.

library(tidyverse)

#################
#### LOADING ####
#################

data_dir <- Sys.getenv("AA_DATA_DIR")
chirps_fp <- file.path(
  data_dir,
  "public",
  "processed",
  "ssd",
  "chirps",
  "daily",
  "ssd_chirps_bentiu_stats_p25.csv"
)

# make monthly to match with rainfall
df_chirps <- read_csv(chirps_fp) %>%
  mutate(month = lubridate::month(time)) %>%
  group_by(year, month) %>%
  summarize(
    precip = sum(mean_id),
    .groups = "drop"
  )
 
df_floodscan <- read_csv(
  file.path(
    data_dir,
    "private",
    "exploration",
    "ssd",
    "floodscan",
    "bentiu_flood.csv"
  )
) %>%
  group_by(year, month) %>%
  summarize(
    flood = max(mean_id),
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

# compare across time

df_join %>%
  group_by(year) %>%
  mutate(precip_year = sum(precip)) %>%
  ungroup() %>%
  mutate(
    precip = precip / 280
  ) %>%
  ggplot(
    aes(
      x = month
    )
  ) +
  geom_area(
    aes(
      y = precip,
      alpha = precip_year
    ),
    fill = "#ef6666"
  ) +
  geom_line(
    aes(
      y = flood
    )
  ) +
  facet_wrap(~year) +
  scale_y_continuous(
    labels = scales::label_percent(1),
    sec.axis = sec_axis(~ . * 280, name = "Precipitation (mm)")
  ) +
  scale_x_continuous(
    breaks = seq(3, 12, 3),
    labels = c("Mar", "Jun", "Sep", "Dec")
  ) +
  labs(
    x = "Month",
    y = "% area flooded",
    fill = "Total yearly precipitation",
    title = "Precipitation and flooding around Bentiu, Rubkona"
  )

# are things different if we look at cumulative precipitation
# until that point in the year?

df_join %>%
  group_by(year) %>%
  mutate(
    precip_cum = cumsum(precip),
    precip_year = sum(precip)
  ) %>%
  ungroup() %>%
  mutate(
    precip_cum = precip_cum / 1100
  ) %>%
  ggplot(
    aes(
      x = month
    )
  ) +
  geom_area(
    aes(
      y = precip_cum,
      alpha = precip_year
    ),
    fill = "#ef6666"
  ) +
  geom_line(
    aes(
      y = flood
    )
  ) +
  facet_wrap(~year) +
  scale_y_continuous(
    labels = scales::label_percent(1),
    sec.axis = sec_axis(~ . * 1100, name = "Cumulative precipitation (mm)")
  ) +
  scale_x_continuous(
    breaks = seq(3, 12, 3),
    labels = c("Mar", "Jun", "Sep", "Dec")
  ) +
  labs(
    x = "Month",
    y = "% area flooded",
    fill = "Total yearly precipitation",
    title = "Cumulative precipitation and flooding around Bentiu, Rubkona"
  )

# Abnormal cumulative precipitation?


df_join %>%
  group_by(year) %>%
  mutate(
    precip_cum = cumsum(precip),
    precip_year = sum(precip)
  ) %>%
  filter(month >= 4) %>%
  group_by(month) %>%
  mutate(precip_cum = 100 * precip_cum / mean(precip_cum)) %>%
  ungroup() %>%
  mutate(
    precip_cum = precip_cum / 300
  ) %>%
  ggplot(
    aes(
      x = month
    )
  ) +
  geom_area(
    aes(
      y = precip_cum,
      alpha = precip_year
    ),
    fill = "#ef6666"
  ) +
  geom_line(
    aes(
      y = flood
    )
  ) +
  facet_wrap(~year) +
  geom_hline(
    yintercept = 100 / 300,
    alpha = 0.7
  ) +
  geom_text(
    data = data.frame(
      month = 6,
      precip = 140 / 300,
      year = 2018
    ),
    aes(y = precip),
    label = "Normal cum. precip.",
    hjust = 0,
    size = 3
  ) +
  scale_y_continuous(
    labels = scales::label_percent(1),
    sec.axis = sec_axis(
      ~ . * 300,
      name = "Cumulative precipitation (mm), % of mean",
      labels = scales::percent_format(1, scale = 1))
  ) +
  scale_x_continuous(
    breaks = seq(3, 12, 3),
    labels = c("Mar", "Jun", "Sep", "Dec")
  ) +
  labs(
    x = "Month",
    y = "% area flooded",
    fill = "Total yearly precipitation",
    title = "Cumulative precipitation and flooding around Bentiu, Rubkona",
    subtitle = "Looking at cumulative precipitation relative to mean in that month"
  )
