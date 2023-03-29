library(tidyverse)
library(gghdx)
gghdx()
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

df_pre2021 <- df_floodscan %>%
  filter(year < 2021) %>%
  group_by(month) %>%
  summarize(
    min_extent = min(flood_extent),
    max_extent = max(flood_extent)
  )

df_2021 <- df_floodscan %>%
  filter(year == 2021) %>%
    mutate(
        min_extent = df_pre2021$max_extent
    )

df_2022 <- df_floodscan %>%
    filter(year == 2022) %>%
    mutate(
        min_extent = df_2021$flood_extent
    )

plt_floodextent <- ggplot(
  mapping = aes(
    x = month
  )
) +
  geom_ribbon(
    data = df_pre2021,
    mapping = aes(
      ymin = min_extent,
      ymax = max_extent
    ),
    fill = hdx_hex("mint-hdx")
  ) +
  geom_ribbon(
    data = df_2021,
    mapping = aes(
      ymin = min_extent,
      ymax = flood_extent,
      group = year
    ),
    lwd = 0,
    fill = hdx_hex("tomato-light")
  ) +
    geom_ribbon(
        data = df_2022,
        mapping = aes(
            ymin = min_extent,
            ymax = flood_extent,
            group = year
        ),
        lwd = 0,
        fill = hdx_hex("tomato-hdx")
  ) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1)
  ) +
  scale_x_continuous(
    breaks = seq(3, 12, 3),
    labels = month.abb[seq(3, 12, 3)]
  ) +
  geom_text(
    data = data.frame(
      y = c(0.1, 0.16, 0.032),
      x = 9.5,
      label = c(2021:2022, "1998 - 2020")
    ),
    aes(x = x, y = y, label = label),
    fontface = "bold",
    family = "Source Sans Pro",
    size = 6,
    color = "white"
  ) +
  labs(
    x = "Month",
    y = "% of areas flooded",
    title = "Flooded areas around the Sudd wetlands, South Sudan"
  ) +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 16),
    plot.title = element_text(size = 20)
  )

ggsave(file.path(
  data_dir,
  "private",
  "exploration",
  "ssd",
  "floodscan",
  "ssd_floodscan_roi_perc_flooded_timeseries_v2.png"
),plot = plt_floodextent, width=4,height=2)
