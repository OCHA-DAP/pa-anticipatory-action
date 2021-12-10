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

######################
#### DATA LOADING ####
######################

df_crop_curr <- read_csv(file.path(
  wrsi_dir, "tcd_wrsi_current_cropland_thresholds.csv"
), col_types = "__dDd") %>%
  mutate(type = "cropland_current")

df_range_curr <- read_csv(file.path(
  wrsi_dir, "tcd_wrsi_current_rangeland_thresholds.csv"
), col_types = "__dDd") %>%
  mutate(type = "rangeland_current")

df_crop_anom <- read_csv(file.path(
  wrsi_dir, "tcd_wrsi_anomaly_cropland_thresholds.csv"
), col_types = "__dDd") %>%
  mutate(type = "cropland_anomaly")

df_range_anom <- read_csv(file.path(
  wrsi_dir, "tcd_wrsi_anomaly_rangeland_thresholds.csv"
), col_types = "__dDd") %>%
  mutate(type = "rangeland_anomaly")

biomasse_drought_years <- read_csv(
  file.path(
    biomasse_dir, "biomasse_impact_years.csv"
  )
) %>% pull(year)

biomasse <- read_csv(
  file.path(
    biomasse_dir, "biomasse_tcd_ADM2_dekad_10.csv"
  )
)

########################
#### DATA WRANGLING ####
########################

df <- bind_rows(
  df_crop_anom,
  df_crop_curr,
  df_range_anom,
  df_range_curr
) %>%
  mutate(year = year(time),
         dekad = (mday(time) %/% 10) + ((month(time) - 1) * 3) + 1,
         wrsi_percent_area = wrsi_percent_area * 100,
         drought_list = factor(year %in% c(2001, 2004, 2009, 2011, 2017), levels = c(TRUE, FALSE)),
         drought_biom = factor(year %in% biomasse_drought_years, levels = c(TRUE, FALSE)))

#########################
#### INITIAL EXPLORE ####
#########################

# First, look at the anomalies and see percent of areas

df %>%
  filter(threshold == 100, str_detect(type, "anomaly")) %>%
  pivot_wider(names_from = type, values_from = wrsi_percent_area) %>%
  ggplot(aes(x = rangeland_anomaly, y = cropland_anomaly)) +
  geom_bin2d(bins=50) + 
  theme_minimal() +
  geom_abline(slope = 1, intercept = 0) +
  labs(title = "WRSI anomaly, May to November",
       subtitle = "Central Chad region, rangeland vs. cropland",
       y = "Cropland anomaly (% of target area <= 100%)",
       x = "Rangeland anomaly (% of target area <= 100%)")

# Then check current

df %>%
  filter(threshold == 70, str_detect(type, "current")) %>%
  pivot_wider(names_from = type, values_from = wrsi_percent_area) %>%
  ggplot(aes(x = rangeland_current, y = cropland_current)) +
  geom_bin2d(bins=50) + 
  theme_minimal() +
  geom_abline(slope = 1, intercept = 0) +
  labs(title = "WRSI anomaly, May to November",
       subtitle = "Central Chad region, rangeland vs. cropland",
       y = "Cropland current (% of target area <= 70%)",
       x = "Rangeland current (% of target area <= 70%)")

# compare WRSI to Biomasse

df %>%
  filter(threshold == 100, type == "rangeland_anomaly") %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 30, y = 155, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 30, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI rangeland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Rangeland WRSI anomaly (% of target area <= median)",
       fill = "Drought (from list)")

# Same graph, but setting threshold for WRSI to be 80% of median

df %>%
  filter(threshold == 80, type == "rangeland_anomaly") %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 9, y = 155, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 9, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI rangeland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Rangeland WRSI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

## Cropland VS Biomasse

df %>%
  filter(threshold == 100, type == "cropland_anomaly") %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 15, y = 155, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 15, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI cropland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Cropland WRSI anomaly (% of target area <= median)",
       fill = "Drought (from list)")


df %>%
  filter(threshold == 80, type == "cropland_anomaly") %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 10, y = 155, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 10, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI cropland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, May to November",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Cropland WRSI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

# Problem above is that we are looking across all dekads in the season.
# Let's examine how they look at the beginning of September, which we
# found to be optimal for Biomasse

df %>%
  filter(threshold == 100, type == "rangeland_anomaly", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
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
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI rangeland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Rangeland WRSI anomaly (% of target area <= median)",
       fill = "Drought (from list)")

# Same but with 80% for rangeland

df %>%
  filter(threshold == 80, type == "rangeland_anomaly", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 7, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 7, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI rangeland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Rangeland WRSI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

# Now back to looking at the WRSI cropland

df %>%
  filter(threshold == 100, type == "cropland_anomaly", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 15, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 15, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI cropland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Cropland WRSI anomaly (% of target area <= median)",
       fill = "Drought (from list)")


df %>%
  filter(threshold == 80, type == "cropland_anomaly", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 7, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 7, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI cropland anomaly vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Cropland WRSI anomaly (% of target area <= 80% median)",
       fill = "Drought (from list)")

# Does using current WRSI show anything different?

df %>%
  filter(threshold == 80, type == "cropland_current", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 10, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 10, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI cropland current vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Cropland WRSI current (% of target area <= 80%)",
       fill = "Drought (from list)")


df %>%
  filter(threshold == 80, type == "rangeland_current", dekad == 24) %>%
  left_join(biomasse, by = c("year", "dekad")) %>%
  arrange(drought_list) %>%
  ggplot(aes(x = wrsi_percent_area, y = biomasse_anomaly)) +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 100, ymax = Inf),
            fill = "#A4D2AC") +
  geom_rect(aes(xmin = 0, xmax = Inf, ymin = 50, ymax = 100),
            fill = "#FF8080") +
  geom_text(x = 20, y = 125, label = "'Good' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_text(x = 20, y = 55, label = "'Bad' Biomasse values", color = "white",
            fontface = "bold", check_overlap = T) +
  geom_point(aes(fill = drought_list),
             shape = 21,
             color = "black") + 
  scale_fill_manual(values = c("white", "black"),
                    labels = c("No drought", "Drought")) +
  theme_minimal() +
  geom_hline(yintercept = 100) +
  labs(title = "WRSI rangeland current vs.Biomasse anomaly",
       subtitle = "Central Chad region, data published beginning of September",
       y = "Biomasse anomaly (% of mean Biomasse)",
       x = "Rangeland WRSI current (% of target area <= 80%)",
       fill = "Drought (from list)")

# Looking at performance of using cropland anomaly, 80%

crop_anom_80_df <- df %>%
  filter(threshold == 80, type == "cropland_anomaly")
  
crop_anom_80_df %>%
  ggplot(aes(x = time, y = wrsi_percent_area, group = year)) +
  geom_area(fill = "#FF8080") +
  facet_wrap(~year, scales = "free_x", ncol = 4) +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month",
               date_labels = "%b") +
  labs(y = "WRSI % of area <= 80% median",
       x = "Month",
       title = "WRSI cropland anomaly, 80% of median",
       subtitle = "Central Chad region, May to November")
  
##########################
#### WRSI PERFORMANCE ####
##########################

param_test <- function(df) {
  thresholds <- 1:20
  scores <- map_df(thresholds, function(x) {
    df %>%
      mutate(pred = factor(wrsi_percent_area >= x, levels = c(TRUE, FALSE))) %>%
      recall(drought_biom, pred)
  })[[".estimate"]]
  thresholds[which.max(scores)]
}

pred_df <- crop_anom_80_df %>%
  select(dekad, year, time, wrsi_percent_area, starts_with("drought")) %>%
  nest(data = -dekad) %>%
  mutate(threshold_area = map_dbl(data, param_test),
         threshold_area_fixed = 2) %>%
  unnest(cols = data) %>%
  mutate(pred = factor(wrsi_percent_area >= threshold_area, levels = c(TRUE, FALSE)),
         pred_fixed = factor(wrsi_percent_area >= threshold_area_fixed, levels = c(TRUE, FALSE))) %>%
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
         date_pub = date + days(10)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  scale_y_continuous(limits = c(0,1)) +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, WRSI cropland anomaly, against list of drought years",
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
         date_pub = date + days(10)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, WRSI cropland anomaly, against end of season Biomasse drought",
       subtitle = "Threshold optimized for each date of publication",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")



## COMPARED AGAINST BIOMASSE END OF YEAR

metrics_df <- bind_rows(metrics(pred_df, drought_biom, pred_fixed),
                        precision(pred_df, drought_biom, pred_fixed),
                        recall(pred_df, drought_biom, pred_fixed))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(10)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, WRSI cropland anomaly, against end of season Biomasse drought",
       subtitle = "Threshold of >= 2% of covered area with <= 80% anomaly",
       y = "Performance",
       x = "Date of publication",
       color = "Metric")

## COMPARED AGAINST BIOMASSE END OF YEAR

metrics_df <- bind_rows(metrics(pred_df, drought_list, pred_fixed),
                        precision(pred_df, drought_list, pred_fixed),
                        recall(pred_df, drought_list, pred_fixed))

metrics_df %>%
  filter(.metric %in% c("precision" , "recall", "accuracy")) %>%
  mutate(month = ((dekad - 1) %/% 3) + 1,
         day = 10 * ((dekad - 1) %% 3) + 1,
         date = ymd(paste(2000, month, day, sep = "-")),
         date_pub = date + days(10)) %>%
  ggplot(aes(x = date_pub, y = `.estimate`, color = `.metric`, group = `.metric`)) +
  geom_line() +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%B") +
  scale_y_continuous(limits = c(0,1)) +
  scale_color_manual(values = c("lightgrey", "darkgrey", "black")) +
  labs(title = "Performance metrics, WRSI cropland anomaly, against list of drought years",
       subtitle = "Threshold of >= 2% of covered area with <= 80% anomaly",
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
  labs(x = "Date of publication",
       y = "Value",
       title = "Prediction outcomes, WRSI cropland anomaly, against list of drought years",
       subtitle = "Fixed threshold >= 2% of area with <= 80% anomaly")

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
       title = "Prediction outcomes, WRSI cropland anomaly, against end of season Biomasse drought",
       subtitle = "Fixed threshold >= 2% of area with <= 80% anomaly")

