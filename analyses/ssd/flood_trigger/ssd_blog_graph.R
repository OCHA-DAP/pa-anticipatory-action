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

plt_floodextent <- ggplot(
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
    fill = "#78D9D1"
  ) +
  geom_line(
    data = line_df,
    mapping = aes(
      y = flood_extent,
      group = year
    ),
    lwd = 1.5,
    color = "#F2645A"
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
    fontface = "bold",
    family = "Source Sans Pro",
    size = 5
  ) +
  labs(
    x = "Month",
    y = "% of areas flooded",
    title = "Flooded areas around the Sudd wetlands, South Sudan"
  ) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 22,
      margin = margin(10, 10, 10, 10, "pt"),
      family = "Source Sans Pro"
    ),
    plot.background = element_rect(
      fill = "white"
    ),
    axis.text = element_text(
      face = "bold",
      size = 10,
      family = "Source Sans Pro"
    ),
    legend.text = element_text(
      size = 12,
      family = "Source Sans Pro"
    ),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    legend.background = element_rect(fill = "transparent"),
    legend.box.background = element_rect(fill = "transparent"),
    strip.text = element_text(
      size = 16,
      family = "Source Sans Pro"
    )
  )

ggsave(file.path(
  data_dir,
  "private",
  "exploration",
  "ssd",
  "floodscan",
  "ssd_floodscan_roi_perc_flooded_timeseries.png"
),plot = plt_floodextent, width=10,height=5)
