###############
#### SETUP ####
###############

library(tidyverse)
library(ggthemes)
library(yardstick)
library(lubridate)
library(ggrepel)
library(ggpubr)

set.seed(1)

aa_dir <- Sys.getenv("AA_DATA_DIR")

metrics_dir <- file.path(
  aa_dir,
  "private",
  "exploration",
  "tcd",
  "trigger_performance"
)

df <- read_csv(file.path(
  aa_dir, "public", "processed", "tcd", "biomasse", "biomasse_tcd_ADM2_dekad_10.csv"
)) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20))

biomasse_years <- read_csv(
  file.path(
    aa_dir, "public", "processed", "tcd", "biomasse", "biomasse_impact_years.csv"
  )
)$year

list_years <- c(2001, 2004, 2009, 2011, 2017)

####################
#### PREDICTION ####
####################

pred_df <- mutate(
  df,
  biomasse_pred = factor(biomasse_anomaly <= 80, levels = c(T,F)),
  bm_end_season = factor(year %in% biomasse_years, levels = c(T,F)),
  drought_list = factor(year %in% list_years, levels = c(T,F))
)

#################
#### METRICS ####
#################

# generate metrics for months July to December
# only looking at data from the 2nd dekad of
# each month, which is roughly available from
# the first of the next month

calc_metrics <- function(df, pred, true, group = NULL) {
  df %>%
    group_by({{ group }}) %>%
    summarize(
      var = precision_vec({{ true }}, {{ pred }}),
      far = 1 - var,
      det = recall_vec({{ true }}, {{ pred }}),
      mis = 1 - det,
      acc = accuracy_vec({{ true }}, {{ pred }}),
      atv = sum({{ pred }} == TRUE) / n()
    )
}

pred_sep_df <- pred_df %>%
  mutate(month_pub = month + 1) %>%
  filter(
    day == 11, 
    month_pub == 9
  ) %>%
  select(
    biomasse_pred, drought_list
  )

# Bootstrap performance metrics for Biomasse

bootstrapped_metrics <- map_dfr(
  1:10000,
  ~ pred_sep_df[sample(1:nrow(pred_sep_df), nrow(pred_sep_df), replace = T),] %>%
    calc_metrics(biomasse_pred, drought_list)
    )

###########################
#### TRIGGER 3 METRICS ####
###########################

df_t3_95 <- bootstrapped_metrics %>%
  summarize(
    across(
      everything(),
      ~quantile(.x, probs = c(0.05, 0.5, 0.95), na.rm = T)
    )
  ) %>%
  mutate(
    point = c("low_end", "central", "high_end"),
    trigger = "Trigger3"
  ) %>%
  pivot_longer(
    var:atv,
    names_to = "metric"
  ) %>%
  select(
    metric, trigger, value, point
  ) 

# Saving in Trigger 1 and 2 metrics
# Calculated in `tcd_iri_seas_forecast_probability.md`

df_t12 <- data.frame(
  metric = "atv",
  trigger = c("Trigger1", "Trigger2", "Trigger1-2"),
  value = c(0.070746, 0.079827, 0.111824),
  point = "central"
)

df_t3_95 %>%
  bind_rows(
    df_t12
  ) %>%
  write_csv(
    file.path(
      metrics_dir,
      "tcd_perf_metrics_table_ci_0.95.csv"
    )
  )

df_t3_68 <- bootstrapped_metrics %>%
  summarize(
    across(
      everything(),
      ~quantile(.x, probs = c(0.32, 0.5, 0.68), na.rm = T)
    )
  ) %>%
  mutate(
    point = c("low_end", "central", "high_end"),
    trigger = "Trigger3"
  ) %>%
  pivot_longer(
    var:atv,
    names_to = "metric"
  ) %>%
  select(
    metric, trigger, value, point
  ) 

df_t3_68 %>%
  write_csv(
    file.path(
      metrics_dir,
      "tcd_perf_metrics_table_ci_0.68.csv"
    )
  )
