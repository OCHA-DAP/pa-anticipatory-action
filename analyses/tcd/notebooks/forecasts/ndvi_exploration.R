library(tidyverse)
library(ggthemes)
library(lubridate)
library(gridExtra)
library(yardstick)

#######################
#### GENERAL SETUP ####
#######################

aa_dir <- Sys.getenv("AA_DATA_DIR")
tcd_dir <- file.path(aa_dir, "public", "processed", "tcd")
wrsi_dir <- file.path(tcd_dir, "wrsi")
biomasse_dir <- file.path(tcd_dir, "biomasse")
ndvi_dir <- file.path(tcd_dir, "ndvi")

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
            ndvi_percent_area = percent_area * 100,
            drought_list = factor(year %in% c(2001, 2004, 2009, 2011, 2017), levels = c(TRUE, FALSE)),
            drought_biom = factor(year %in% biomasse_drought_years, levels = c(TRUE, FALSE))) %>%
  arrange(date)

#########################
#### INITIAL EXPLORE ####
#########################

# compare NDVI to Biomasse

p_ndvi_bm <- df %>%
  filter(anomaly_thresholds == 100) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = ndvi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 10, y = 170, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 10, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("black", "white"),
                    labels = c("Drought", "No drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "NDVI anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "NDVI anomaly (% of target area <= median)",
       fill = "Drought (from list)")

p_ndvi_bm

# Same graph, but setting threshold for NDVI to be 80% of median

p_ndvi_bm_80 <- df %>%
  filter(anomaly_thresholds == 80) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = ndvi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 10, y = 170, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 10, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("black", "white"),
                    labels = c("Drought", "No drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "NDVI anomaly vs. Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "NDVI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

p_ndvi_bm_80

# Problem above is that we are looking across all dekads in the season.
# Let's examine how they look at the beginning of September, which we
# found to be optimal for Biomasse

p_ndvi_bm_fy <- df %>%
  filter(anomaly_thresholds == 100, dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = ndvi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 30, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 30, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("black", "white"),
                    labels = c("Drought", "No drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "NDVI anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "NDVI anomaly (% of target area <= median)",
       fill = "Drought (from list)")

p_ndvi_bm_fy

# Same but with 80% for NDVI

p_ndvi_bm_fy_80 <- df %>%
  filter(anomaly_thresholds == 80, dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = ndvi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 30, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 30, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("black", "white"),
                    labels = c("Drought", "No drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "NDVI anomaly vs. Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "NDVI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

p_ndvi_bm_fy_80
  
# Explore a bit more cropland anom vs biomasse anom

p_ndvi_bm_fy_80 +
  geom_hline(yintercept = 80,
             color = "white") + 
  geom_text(x = .1,
            y = 60,
            label = "Separation based on Biomasse anomaly <= 80% \n gives fairly clear grouping of values.",
            check_overlap = T,
            hjust = 0,
            color = "white")

p_ndvi_bm_fy_80 +
  geom_vline(xintercept = 12,
             color = "white") + 
  geom_text(x = 3,
            y = 80,
            label = "Separation based on WRSI \ncropland anomaly <= 80% in >= 3% of areas\nis less clearly grouped",
            check_overlap = T,
            hjust = 0,
            color = "white")

point_year_df <- df %>%
  filter(anomaly_thresholds == 80, dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  filter(biomasse_anomaly <= 80 | ndvi_percent_area >= 12)

p_ndvi_bm_fy_80 +
  ggrepel::geom_text_repel(
    data = point_year_df,
    aes(label = year),
    color = "white"
  ) +
  geom_vline(xintercept = 12,
             color = "white") +
  geom_hline(yintercept = 80,
             color = "white") +
  geom_text(x = 7.5,
            y = 120,
            label = "Question is, how do we consider\nyears 2008 & 2013 captured by NDVI\nvs 2006 captured by Biomasse?\nDo we trust our list of years?",
            check_overlap = T,
            hjust = 0,
            color = "white")

##########################
#### NDVI PERFORMANCE ####
##########################

# Looking at performance of using NDVI, 80%

ndvi_anom_80_df <- df %>%
  filter(anomaly_thresholds == 80)

plt_ndvi_anom_80 <- ndvi_anom_80_df %>%
  ggplot(aes(x = date, y = ndvi_percent_area, group = year)) +
  geom_area(fill = "#FF8080") +
  facet_wrap(~year, scales = "free_x", ncol = 4) +
  theme_minimal() +
  scale_x_date(date_breaks = "3 month",
               date_labels = "%b") +
  labs(y = "NDVI % of area <= 80% median",
       x = "Month",
       title = "NDVI anomaly, 80% of median",
       subtitle = "Central Chad region, May to November")

param_test <- function(df) {
  thresholds <- 1:20
  scores <- map_df(thresholds, function(x) {
    df %>%
      mutate(pred = factor(ndvi_percent_area >= x, levels = c(TRUE, FALSE))) %>%
      f_meas(drought_biom, pred)
  })[[".estimate"]]
  thresholds[which.max(scores)]
}

pred_df <- ndvi_anom_80_df %>%
  select(dekad, year, date, ndvi_percent_area, starts_with("drought")) %>%
  nest(data = -dekad) %>%
  mutate(threshold_area = map_dbl(data, param_test),
         threshold_area_fixed = 12) %>%
  unnest(cols = data) %>%
  mutate(pred = factor(ndvi_percent_area >= threshold_area, levels = c(TRUE, FALSE)),
         pred_fixed = factor(ndvi_percent_area >= threshold_area_fixed, levels = c(TRUE, FALSE))) %>%
  group_by(dekad)

## COMPARED AGAINST DROUGHT LIST

metrics_df <- bind_rows(metrics(pred_df, drought_list, pred),
                        precision(pred_df, drought_list, pred),
                        recall(pred_df, drought_list, pred))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(30)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  scale_y_continuous(limits = c(0,1)) +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, NDVI anomaly, against list of drought years",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

## COMPARED AGAINST BIOMASSE END OF YEAR

metrics_df <- bind_rows(metrics(pred_df, drought_biom, pred),
                        precision(pred_df, drought_biom, pred),
                        recall(pred_df, drought_biom, pred))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(30)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, NDVI anomaly, against end of season Biomasse drought",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

## COMPARED AGAINST DROUGHT LIST, FIXED THRESHOLD

metrics_df <- bind_rows(metrics(pred_df, drought_list, pred_fixed),
                        precision(pred_df, drought_list, pred_fixed),
                        recall(pred_df, drought_list, pred_fixed))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(30)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  scale_y_continuous(limits = c(0,1)) +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, NDVI anomaly, against list of drought years",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

## PLOTTING ALL THE SPECIFIC TN, TP, ... VALUES

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
  scale_x_date(date_breaks = "3 month",
               date_labels = "%b") +
  labs(x = "Date of publication",
       y = "Value",
       title = "Prediction outcomes, NDVI anomaly, against list of drought years",
       subtitle = "Fixed threshold >= 12% of area with <= 80% anomaly")

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
  scale_x_date(date_breaks = "3 month",
               date_labels = "%b") +
  labs(x = "Date of publication",
       y = "Value",
       title = "Prediction outcomes, NDVI anomaly, against end of season Biomasse drought",
       subtitle = "Fixed threshold >= 12% of area with <= 80% anomaly")

