library(tidyverse)
library(ggthemes)
library(yardstick)
library(lubridate)

aa_dir <- Sys.getenv("AA_DATA_DIR")

df <- read_csv(file.path(
  aa_dir, "public", "processed", "tcd", "biomasse", "biomasse_tcd_ADM2_dekad_10.csv"
))

# Biomasse anomaly across years, region of interest

df %>%
  filter(between(dekad, 16, 33)) %>%
  ggplot(aes(x = dekad, y = biomasse_anomaly)) +
  geom_area(alpha = 0.75, fill = rgb(238, 88, 89, maxColorValue=255)) +
  facet_wrap(~year) +
  geom_hline(yintercept = 50,
             alpha = 0.5) +
  geom_hline(yintercept = 100,
             alpha = 0.5) +
  theme_minimal() +
  labs(title = "Biomasse anomaly, June to November",
       subtitle = "Central Chad region",
       y = "Biomasse (anomaly)",
       x = "Dekad")

# Biomasse anomaly across years, showing growing season

plt_bm_anom_jjason = df %>%
  mutate(season = (year + (dekad >= 10)) - 1,
         season_order = ifelse(dekad >= 10,
                               dekad - 9,
                               dekad + 27)) %>%
  ggplot(aes(x = season_order, y = biomasse_anomaly)) +
  geom_rect(xmin=7, xmax = 24, ymin = 0, ymax = 150, fill = "grey", alpha = 0.25) +
  geom_area(alpha = 0.75, fill = rgb(238, 88, 89, maxColorValue=255)) +
  facet_wrap(~season) +
  geom_hline(yintercept = 50,
             alpha = 0.5) +
  geom_hline(yintercept = 100,
             alpha = 0.5) +
  theme_minimal() +
  labs(title = "Biomasse anomaly, June to November",
       subtitle = "Central Chad region, season from first Dekad of April to last of March",
       y = "Biomasse (anomaly)",
       x = "Dekad (into the season)")

# August confusion matrix

df %>%
  mutate(drought = factor(year %in% c(2001, 2004, 2009, 2011, 2017))) %>%
  group_by(year) %>%
  filter(dekad == 21) %>%
  mutate(pred = factor(biomasse_anomaly <= 80)) %>%
  distinct(year, drought, pred) %>%
  ungroup() %>%
  conf_mat(drought, pred) %>%
  autoplot(type = "heatmap") +
  labs(title="Confusion matrix, threshold of 80 on August 1")


bm_cm_80_nov <- df %>%
  mutate(drought = factor(year %in% c(2001, 2004, 2009, 2011, 2017))) %>%
  group_by(year) %>%
  filter(dekad == 27) %>%
  mutate(pred = factor(biomasse_anomaly <= 80)) %>%
  distinct(year, drought, pred) %>%
  ungroup() %>%
  conf_mat(drought, pred) %>%
  autoplot(type = "heatmap") +
  labs(title="Confusion matrix, threshold of 80 on November 1")

# # END OF YEAR IMPACT

# What years do we want to define here? 85 seems a reasonable threshold
# but misses out on 2017 as well as 2001, but has enough instances for
# validation against other sources.

end_impact <- df %>%
  filter(dekad == 33,
         biomasse_anomaly <= 85)

end_impact %>%
  write_csv(
    file.path(
      aa_dir, "public", "processed", "tcd", "biomasse", "biomasse_impact_years.csv"
    )
  )

# # TEST IMPACT YEARS ACROSS PUB DATE AND THRESHOLDS

imp_years <- end_impact$year
list_years <- c(2001, 2004, 2009, 2011, 2017)

test_df <- df %>%
  mutate(drought_biom = factor(year %in% imp_years, levels = c(TRUE, FALSE)),
         drought_list = factor(year %in% list_years, levels = c(TRUE, FALSE))) %>%
  filter(between(dekad, 17, 33))

param_test <- function(df) {
  thresholds <- 70:90
  scores <- map_df(thresholds, function(x) {
    df %>%
      mutate(pred = factor(biomasse_anomaly <= x, levels = c(TRUE, FALSE))) %>%
      recall(drought_biom, pred)
  })[[".estimate"]]
  thresholds[which.max(scores)]
}

pred_df <- test_df %>%
  select(dekad, year, biomasse_anomaly, starts_with("drought")) %>%
  nest(data = -dekad) %>%
  mutate(threshold = map_dbl(data, param_test),
         threshold_fixed = 80) %>%
  unnest(cols = c(data)) %>%
  mutate(pred = factor(biomasse_anomaly <= threshold, levels = c(TRUE, FALSE)),
         pred_fixed = factor(biomasse_anomaly <= threshold_fixed, levels = c(TRUE, FALSE))) %>%
  group_by(dekad)

# # COMPARED AGAINST BIOMASSE END OF YEAR

metrics_df <- bind_rows(metrics(pred_df, drought_biom, pred),
                        precision(pred_df, drought_biom, pred),
                        recall(pred_df, drought_biom, pred))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, Biomasse, against end of season Biomasse drought",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

# # COMPARED AGAINST LIST OF DROUGHT YEARS

metrics_df <- bind_rows(metrics(pred_df, drought_list, pred),
                        precision(pred_df, drought_list, pred),
                        recall(pred_df, drought_list, pred))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, Biomasse, against list of drought years",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")


# # PLOTTING THE THRESHOLDS

pred_df %>%
  ungroup() %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20)) %>%
  distinct(date_pub, threshold) %>%
  ggplot(aes(x = date_pub, y = threshold)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Threshold, by date of publication",
       y = "Threshold",
       x = "Date of publication")

# # COMPARED AGAINST BIOMASSE, FIXED THRESHOLD

metrics_df <- bind_rows(metrics(pred_df, drought_biom, pred_fixed),
                        precision(pred_df, drought_biom, pred_fixed),
                        recall(pred_df, drought_biom, pred_fixed))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, Biomasse, against end of season Biomasse drought",
       subtitle = "Fixed threshold of 80% Biomasse anomaly",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

# # COMPARED AGAINST DROUGHT LIST, FIXED THRESHOLD

metrics_df <- bind_rows(metrics(pred_df, drought_list, pred_fixed),
                        precision(pred_df, drought_list, pred_fixed),
                        recall(pred_df, drought_list, pred_fixed))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, Biomasse, against list of drought years",
       subtitle = "Fixed threshold of 80% Biomasse anomaly",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

# # PLOTTING ALL THE SPECIFIC TN, TP, ... VALUES

pred_df %>%
  mutate(across(matches("drought|pred"), as.logical)) %>%
  summarize(tp = sum(drought_list + pred_fixed == 2),
            fp = sum(!(drought_list) & pred_fixed),
            tn = sum(drought_list + pred_fixed == 0),
            fn = sum(drought_list & !(pred_fixed))) %>%
  pivot_longer(-dekad) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20),
         type_group = name %in% c("tp", "tn"),
         pos_group = name %in% c("tp", "fp"),
         name = toupper(name)) %>%
  ggplot(aes(x = date_pub, y = value, group = name)) +
  geom_line(aes(color = name, linetype = name)) +
  theme_minimal() +
  scale_color_manual(values = c("red", "red", "darkgrey", "darkgrey"), name = "Metric") +
  scale_linetype_manual(values = c("dashed", "solid", "dashed", "solid"), name = "Metric") +
  labs(x = "Date of publication",
       y = "Value",
       title = "Prediction outcomes, Biomasse, against list of drought years",
       subtitle = "Fixed threshold of 80% Biomasse anomaly")



pred_df %>%
  mutate(across(matches("drought|pred"), as.logical)) %>%
  summarize(tp = sum(drought_biom + pred_fixed == 2),
            fp = sum(!(drought_biom) & pred_fixed),
            tn = sum(drought_biom + pred_fixed == 0),
            fn = sum(drought_biom & !(pred_fixed))) %>%
  pivot_longer(-dekad) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(20),
         type_group = name %in% c("tp", "tn"),
         pos_group = name %in% c("tp", "fp"),
         name = toupper(name)) %>%
  ggplot(aes(x = date_pub, y = value, group = name)) +
  geom_line(aes(color = name, linetype = name)) +
  theme_minimal() +
  scale_color_manual(values = c("red", "red", "darkgrey", "darkgrey"), name = "Metric") +
  scale_linetype_manual(values = c("dashed", "solid", "dashed", "solid"), name = "Metric") +
  labs(x = "Date of publication",
       y = "Value",
       title = "Prediction outcomes, Biomasse, against end of season Biomasse drought",
       subtitle = "Fixed threshold of 80% Biomasse anomaly")
