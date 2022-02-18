# look at the use of exponential smoothing on flood extents
# in South Sudan to capture impact of residual flood levels
# from the previous year.

library(tidyverse)
library(forecast)

#################
#### LOADING ####
#################

data_dir <- Sys.getenv("AA_DATA_DIR")

df <- read_csv(
  file.path(
    data_dir,
    "public",
    "exploration",
    "ssd",
    "floods",
    "ssd_floodscan_roi.csv"
  )
) %>%
  select(-1)

df_monthly <- df %>%
  group_by(year, month) %>%
  summarize(
    flood_extent = mean(mean_ADM0_PCODE),
    .groups = "drop"
  )

df_chirps <- read_csv(
  file.path(
    data_dir,
    "public",
    "processed",
    "ssd",
    "chirps",
    "daily",
    "ssd_chirps_roi_stats_p25.csv"
  )
) %>%
  mutate(
    month = lubridate::month(time)
  ) %>%
  group_by(
    year, month
  ) %>%
  summarize(
    precip = sum(mean),
    .groups = "drop"
  )

#################
#### COMPARE ####
#################

# compare CHIRPS and flood extents, just simple
# monthly comparison
inner_join(
  df_chirps,
  df_monthly,
  by = c("year", "month")
) %>%
  group_by(
    year
  ) %>%
  mutate(precip_year = sum(precip),
         precip = scales::rescale(precip, to = c(0, 0.25))) %>%
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
    fill = "#ed4d4d"
  ) +
  geom_line(
    aes(
      y = flood_extent,
      linetype = "Flood extent"
    )
  ) +
  theme_light() +
  facet_wrap(~year) +
  scale_x_continuous(
    breaks = seq(3, 12, by = 3),
    labels = c("Mar", "Jun", "Sep", "Dec")
  ) +
  scale_y_continuous(
    labels = scales::percent_format(1)
  ) +
  labs(
    y = "Flooded extent",
    x = "Month",
    alpha = "Total\nyearly\nprecipitation",
    title = "Rainfall and flood extents, South Sudan region of interest",
    caption = paste0(
      "Rainfall patterns rescaled to 0 to ",
      "0.25 for purposes of comparison, do not ",
      "represent meaningful units."),
    linetype = "",
  ) +
  theme(
    plot.caption = element_text(hjust = 0)
  )
  

###################
#### FUNCTIONS ####
###################

generate_preds <- function(
  df,
  years = 2011:2021,
  pred_month = 8,
  last_month = 12
) {
  map_dfr(
    years,
    ~ df %>%
      filter(year < .x | (year == .x & month < {{ pred_month }})) %>%
      pull(flood_extent) %>%
      ts(
        frequency = 12
      ) %>%
      hw(
        h = 1 + last_month - pred_month,
        exponential = TRUE,
        seasonal = "multiplicative"
      ) %>%
      as.data.frame %>%
      mutate(
        year = .x,
        month = {{ pred_month }}:{{ last_month }}
      )
  )
}

###################
#### SMOOTHING ####
###################

# since we don't actually have separate time
# series to test, will allow training on all
# years up to 2010, then for each year
# following, train up to that year and then
# predict 6 months into the future from June.

# quick check

df_monthly %>%
  ggplot(
    aes(
      x = month,
      y = flood_extent
    )
  ) +
  stat_smooth(
    geom = "area",
    span = 1/4,
    fill = "#ef6666"
  ) +
  facet_wrap(
    ~year
  ) +
  theme_minimal()

# generate predictions

df_pred_july <- generate_preds(df_monthly)

###############
#### GRAPH ####
###############

# Forecasting from July

df_monthly %>%
  left_join(df_pred_july, by = c("year", "month")) %>%
  filter(year >= 2011) %>%
  group_by(year) %>%
  mutate(max_extent = max(flood_extent)) %>%
  ungroup() %>%
  mutate(
    line_check = ifelse(
      month >= 8,
      "Predicted",
      "Observed"
    ),
    worst_year = ifelse(
      max_extent %in% tail(sort(unique(max_extent)), 3),
      "WORST\nYEAR",
      ""
    )
  ) %>%
  ggplot(
    aes(
      x = month,
      y = flood_extent
    )
  ) +
  geom_line(
    aes(
      linetype = line_check
    )
  ) +
  geom_ribbon(
    aes(ymin = `Lo 80`,
        ymax = `Hi 80`),
    alpha = 0.5,
    fill = "#ed4d4d"
  ) +
  facet_wrap(
    ~year,
    scale = "free_x"
  ) +
  theme_minimal() +
  scale_x_continuous(
    breaks = c(3, 6, 9, 12),
    labels = c("Mar", "Jun", "Sep", "Dec")
  ) +
  geom_text(
    aes(label = worst_year),
    x = 2,
    y = 0.15,
    check_overlap = TRUE,
    size = 3,
    fontface = "bold",
    hjust = 0
  ) +
  labs(
    y = "Mean flood extents",
    x = "Month",
    linetype = "Data type",
    title = "Seasonal exponential smoothing, South Sudan flood extents",
    subtitle = "Forecasting 5 periods ahead from July, 2011 to 2021"
  )
