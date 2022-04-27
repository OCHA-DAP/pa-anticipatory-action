library(tidyverse)

# Graph for SSD blogpost

data_dir <- Sys.getenv("AA_DATA_DIR")

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


## multiline plot overlay

area_df <- df_floodscan %>%
  filter(year < 2021) %>%
  group_by(month) %>%
  summarize(
    min_extent = min(flood_extent),
    max_extent = max(flood_extent)
  )

line_df <- df_floodscan %>%
  filter(year >= 2021)

ggplot(
  mapping = aes(
    x = month
  )
) +
  geom_ribbon(
    data = area_df,
    mapping = aes(
      ymin = min_extent,
      ymax = max_extent
    ),
    fill = "#d7d7d7"
  ) +
  geom_line(
    data = line_df,
    mapping = aes(
      y = flood_extent,
      group = year
    ),
    lwd = 1,
    color = "#ef6666"
  ) +
  theme_minimal() +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_x_continuous(
    breaks = seq(3, 12, 3),
    labels = month.abb[seq(3, 12, 3)]
  ) +
  geom_text(
    data = data.frame(
      y = c(0.083, 0.138, 0.032),
      x = c(1.65, 1.65, 9.5),
      label = c(2021:2022, "1998 - 2020")
    ),
    aes(x = x, y = y, label = label),
    fontface = "bold"
  ) +
  labs(
    x = "Month",
    y = "% of areas flooded",
    title = "Flooded areas in the Sudd wetlands, South Sudan"
  ) +
  theme(
    text = element_text(family = "Helvetica")
  )
