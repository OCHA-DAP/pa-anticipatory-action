###############
#### SETUP ####
###############

library(tidyverse)
library(ggthemes)
library(yardstick)
library(lubridate)
library(ggrepel)

aa_dir <- Sys.getenv("AA_DATA_DIR")

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

end_season_metrics <- pred_df %>%
  mutate(month_pub = month + 1) %>%
  filter(day == 11, between(month_pub, 5, 12)) %>%
  group_by(month_pub) %>%
  summarize(
    `Valid activation rate` = precision_vec(bm_end_season, biomasse_pred),
    `False alarm rate` = 1 - `Valid activation rate`,
    `Detection rate` = recall_vec(bm_end_season, biomasse_pred),
    `Miss rate` = 1 - `Detection rate`,
    Accuracy = accuracy_vec(bm_end_season, biomasse_pred)
  )
  
list_metrics <- pred_df %>%
  mutate(month_pub = month + 1) %>%
  filter(day == 11, between(month_pub, 5, 12)) %>%
  group_by(month_pub) %>%
  summarize(
    `Valid activation rate` = precision_vec(drought_list, biomasse_pred),
    `False alarm rate` = 1 - `Valid activation rate`,
    `Detection rate` = recall_vec(drought_list, biomasse_pred),
    `Miss rate` = 1 - `Detection rate`,
    Accuracy = accuracy_vec(drought_list, biomasse_pred)
  )

# Compared to end of season Biomasse (end of November)

bm_end_season_p <- end_season_metrics %>%
  pivot_longer(-month_pub) %>%
  filter(name %in% c("Valid activation rate", "False alarm rate", "Miss rate")) %>%
  ggplot(aes(x = month_pub, y = name)) +
  geom_tile(aes(alpha = value, fill = name), color = "white") +
  geom_text_repel(aes(label = scales::percent(value, accuracy = 1)),
                  color = "#111111",
                  fontface = "bold",
                  size = 3.5,
                  bg.color = "white",
                  bg.r = 0.15,
                  force = 0) +
  scale_alpha(range = c(0.2, 1)) +
  theme_minimal() +
  scale_fill_manual(values = c("#EA4C46", "#EA4C46", "#62BD69")) +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = 5:12,
                     labels = ~ month.abb[.x]) +
  labs(x = "Month of publication",
       y = "",
       title = "Biomasse anomaly, threshold of 80%, against end of season Biomasse, Chad",
       subtitle = "Data published around the 1st of each month")

bm_end_season_p

# Compared to drought list

bm_drought_list_p <- list_metrics %>%
  pivot_longer(-month_pub) %>%
  filter(name %in% c("Valid activation rate", "False alarm rate", "Miss rate")) %>%
  ggplot(aes(x = month_pub, y = name)) +
  geom_tile(aes(alpha = value, fill = name), color = "white") +
  geom_text_repel(aes(label = scales::percent(value, accuracy = 1)),
                  color = "#111111",
                  fontface = "bold",
                  size = 3.5,
                  bg.color = "white",
                  bg.r = 0.15,
                  force = 0) +
  scale_alpha(range = c(0.2, 1)) +
  theme_minimal() +
  scale_fill_manual(values = c("#EA4C46", "#EA4C46", "#62BD69")) +
  theme(legend.position = "none") +
  scale_x_continuous(breaks = 5:12,
                     labels = ~ month.abb[.x]) +
  labs(x = "Month of publication",
       y = "",
       title = "Biomasse anomaly, threshold of 80%, against list of drought years, Chad",
       subtitle = "Data published around the 1st of each month")

bm_drought_list_p
