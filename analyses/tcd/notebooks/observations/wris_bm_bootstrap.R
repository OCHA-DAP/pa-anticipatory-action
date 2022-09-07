library(tidyverse)
library(ggthemes)
library(lubridate)
library(gridExtra)
library(yardstick)
library(gghdx)
gghdx()

#######################
#### GENERAL SETUP ####
#######################

aa_dir <- Sys.getenv("AA_DATA_DIR")
tcd_dir <- file.path(aa_dir, "public", "processed", "tcd")
ndvi_dir <- file.path(tcd_dir, "ndvi")
biomasse_dir <- file.path(tcd_dir, "biomasse")

######################
#### DATA LOADING ####
######################

ndvi <- read_csv(
  file.path(
    ndvi_dir, "tcd_ndvi_anomaly.csv"
  ), col_types = "_Dcdd"
)

biomasse <- read_csv(
  file.path(
    biomasse_dir, "biomasse_tcd_ADM2_dekad_10.csv"
  )
)

biomasse_drought_years <- read_csv(
  file.path(
    biomasse_dir, "biomasse_impact_years.csv"
  )
) %>% pull(year)

########################
#### DATA WRANGLING ####
########################

df <- ndvi %>%
  transmute(date = date,
            year = year(date),
            dekad = (mday(date) %/% 10) + ((month(date) - 1) * 3) + 1 + 2, # +2 so date publication matches Biomasse
            iso3 = toupper(iso3),
            anomaly_thresholds = anomaly_thresholds,
            ndvi_percent_area = percent_area,
            drought_list = factor(year %in% c(2001, 2004, 2009, 2011, 2017), levels = c(TRUE, FALSE)),
            drought_biom = factor(year %in% biomasse_drought_years, levels = c(TRUE, FALSE))) %>%
  arrange(date) %>%
  filter(anomaly_thresholds == 80, dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  mutate(
    biomasse_pred = factor(biomasse_anomaly <= 80, levels = c(TRUE, FALSE)),
    ndvi_pred = factor(ndvi_percent_area >= 12, levels = c(TRUE, FALSE))
  )

##############
#### PLOT ####
##############

base_plot <- df %>%
  ggplot(aes(x = ndvi_percent_area, y = biomasse_anomaly)) +
  scale_fill_manual(values = c("black", "white"),
                    labels = c("Drought", "No drought")) +
  labs(title = "NDVI anomaly vs. Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "NDVI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)") +
  geom_hline(yintercept = 80,
             color = hdx_hex("gray-dark"),
             size = 1.5) +
  geom_vline(xintercept = 12,
             color = hdx_hex("gray-dark"),
             size = 1.5)

# Basic plot for presentation

p_basic <- base_plot +
  geom_point(
    aes(fill = drought_list),
    shape = 21,
    color = "black"
  )

# add in lines to the plot to show margins
# data frame of values close or beyond the thresholds
df_close <- df %>%
  filter(
    ndvi_percent_area >= 8
  )

# margins

p_ndvi_margins <- base_plot +
  geom_segment(
    data = df_close,
    aes(
      x = ndvi_percent_area,
      y = biomasse_anomaly,
      yend = biomasse_anomaly
    ),
    xend = 12,
    color = hdx_hex("gray-dark")
  ) +
  geom_point(
    aes(fill = drought_list),
    shape = 21,
    color = "black"
  )

# standard deviation
ndvi_sd <- sd(df$ndvi_percent_area)
p_ndvi_sd <- base_plot +
  geom_segment(
    aes(
      x = ndvi_percent_area + ndvi_sd,
      xend = ndvi_percent_area - ndvi_sd,
      y = biomasse_anomaly,
      yend = biomasse_anomaly
    ),
    color = hdx_hex("gray-dark")
  ) +
  geom_point(
    aes(fill = drought_list),
    shape = 21,
    color = "black"
  )

# margins for Biomasse
p_bm_margins <- base_plot +
  geom_segment(
    data = df_close,
    aes(
      x = ndvi_percent_area,
      xend = ndvi_percent_area,
      y = biomasse_anomaly
    ),
    yend = 80,
    color = hdx_hex("gray-dark")
  ) +
  geom_point(
    aes(fill = drought_list),
    shape = 21,
    color = "black"
  )

# sd for Biomasse
bm_sd <- sd(df$biomasse_anomaly)

p_bm_sd <- base_plot +
  geom_segment(
    aes(
      x = ndvi_percent_area,
      xend = ndvi_percent_area,
      y = biomasse_anomaly + bm_sd,
      yend = biomasse_anomaly - bm_sd
    ),
    color = hdx_hex("gray-dark")
  ) +
  geom_point(
    aes(fill = drought_list),
    shape = 21,
    color = "black"
  )

###############################
#### BOOTSTRAP PERFORMANCE ####
###############################

calc_metrics <- function(df, pred, true, group = NULL) {
  df %>%
    group_by({{ group }}) %>%
    summarize(`Valid activation rate` = precision_vec({{ true }}, {{ pred }}),
              `False alarm rate` = 1 - `Valid activation rate`,
              `Detection rate` = recall_vec({{ true }}, {{ pred }}),
              `Miss rate` = 1 - `Detection rate`,
              Accuracy = accuracy_vec({{ true }}, {{ pred }}))
}

# Bootstrap performance metrics for September

set.seed(123)

bm_metrics <- map_dfr(
  1:1000,
  ~ df[sample(1:nrow(df), nrow(df), replace = T),] %>%
    calc_metrics(biomasse_pred, drought_list)
)

# Now do the bootstrap while jittering the values
bm_metrics_jittered <- map_dfr(
  1:1000,
  ~ df[sample(1:nrow(df), nrow(df), replace = T),] %>%
    mutate(
      biomasse_anomaly = biomasse_anomaly + rnorm(n = nrow(.), sd = bm_sd),
      biomasse_pred = factor(biomasse_anomaly <= 80, levels = c(TRUE, FALSE))
    ) %>%
    calc_metrics(biomasse_pred, drought_list)
)

# base for NDVI
ndvi_metrics <- map_dfr(
  1:1000,
  ~ df[sample(1:nrow(df), nrow(df), replace = T),] %>%
    calc_metrics(ndvi_pred, drought_list)
)

# NDVI with jittering for uncertainty
ndvi_metrics_jittered <- map_dfr(
  1:1000,
  ~ df[sample(1:nrow(df), nrow(df), replace = T),] %>%
    mutate(
      ndvi_percent_area = ndvi_percent_area + rnorm(n = nrow(.), sd = ndvi_sd),
      ndvi_pred = factor(ndvi_percent_area >= 12, levels = c(TRUE, FALSE))
    ) %>%
    calc_metrics(ndvi_pred, drought_list)
)

# join together and plot the data

df_metrics <- bind_rows(
  bm_metrics %>% mutate(measure = "Biomasse"),
  bm_metrics_jittered %>% mutate(measure = "Biomasse (with uncertainty)"),
  ndvi_metrics %>% mutate(measure = "NDVI"),
  ndvi_metrics_jittered %>% mutate(measure = "NDVI (with uncertainty)")
) %>%
  mutate(
    measure = factor(measure, levels = c(
      "Biomasse",
      "NDVI",
      "Biomasse (with uncertainty)",
      "NDVI (with uncertainty)"
    ))
  ) %>%
  pivot_longer(
    -measure
  )

df_metrics_mean <- df_metrics %>%
  group_by(
    name,
    measure
  ) %>%
  summarize(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  )

p_uncertainty <- df_metrics %>%
  filter(!is.na(value)) %>%
  ggplot(
    aes(x = value)
  ) +
  stat_density(
    aes(y = ..scaled..),
    fill = hdx_hex("sapphire-ultra-light"),
    color = hdx_hex("sapphire-hdx")
  ) +
  geom_segment(
    data = df_metrics_mean,
    aes(
      x = value,
      xend = value
    ),
    y = 0,
    yend = Inf,
  ) +
  geom_text_hdx(
    data = data.frame(
      value = 0.75,
      name = "Accuracy",
      measure = "Biomasse"
    ),
    y = 0.85,
    label = "Bootstrap mean\nfor metric",
    hjust = 1,
    color = "black"
  ) +
  facet_grid(measure~name) +
  labs(
    x = "Performance metric",
    y = "Density (scaled)",
    title = "Bootstrapping performance metrics with uncertainty",
    subtitle = "1,000 bootstrap samples taken for each predictor"
  )

# plotting the drop in performance for each

p_performance_drop <- df_metrics_mean %>%
  mutate(
    general_measure = ifelse(
      str_detect(measure, "NDVI"),
      "NDVI",
      "Biomasse"
    ),
    uncertainty = ifelse(
      str_detect(measure, "uncertainty"),
      "With uncertainty",
      "No uncertainty"
    )
  ) %>%
  pivot_wider(
    id_cols = c(name, general_measure),
    names_from = uncertainty
  ) %>%
  ggplot(
    aes(
      x = `No uncertainty`,
      xend = `With uncertainty`,
      y = general_measure,
      yend = general_measure
    )
  ) +
  geom_segment(
    color = hdx_hex("gray-medium")
  ) +
  geom_point(size = 3) +
  geom_point(
    aes(
      x = `With uncertainty`
    ),
    color = hdx_hex("tomato-hdx"),
    size = 3
  ) +
  facet_wrap(~name, nrow = 1, scales = "free_x") +
  scale_x_continuous(
    breaks = scales::pretty_breaks()
  ) +
  geom_text_hdx(
    data = tibble(
      `No uncertainty` = c(0.75),
      `With uncertainty` = c(0.74),
      general_measure = "NDVI",
      name = "Accuracy"
    ),
    label = "With\nuncertainty",
    hjust = 0,
    color = hdx_hex("tomato-hdx")
  ) +
  geom_text_hdx(
    data = tibble(
      `No uncertainty` = c(0.84),
      `With uncertainty` = c(0.82),
      general_measure = "NDVI",
      name = "Accuracy"
    ),
    label = "No\nuncertainty",
    hjust = 1,
    color = hdx_hex("sapphire-hdx")
  ) +
  labs(
    x = "Performance metric",
    y = "",
    title = "Drop in performance when accounting for uncertainty during bootstrapping",
    subtitle = "Mean of the metric across 1,000 bootstrap samples"
  )
