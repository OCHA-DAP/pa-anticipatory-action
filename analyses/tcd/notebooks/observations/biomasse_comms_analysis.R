###############
#### SETUP ####
###############

library(tidyverse)
library(ggthemes)
library(yardstick)
library(lubridate)
library(ggrepel)
library(ggpubr)

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

calc_metrics <- function(df, pred, true, group = NULL) {
  df %>%
    group_by({{ group }}) %>%
    summarize(`Valid activation rate` = precision_vec({{ true }}, {{ pred }}),
              `False alarm rate` = 1 - `Valid activation rate`,
              `Detection rate` = recall_vec({{ true }}, {{ pred }}),
              `Miss rate` = 1 - `Detection rate`,
              Accuracy = accuracy_vec({{ true }}, {{ pred }}),
              `Return period` = (n() + 1) / sum({{ pred }} == TRUE))
}

pred_jul_dec_df <- pred_df %>%
  mutate(month_pub = month + 1) %>%
  filter(day == 11, between(month_pub, 5, 12))

end_season_metrics <- calc_metrics(pred_jul_dec_df, bm_end_season, biomasse_pred, month_pub)
list_metrics <- calc_metrics(pred_jul_dec_df, drought_list, biomasse_pred, month_pub)

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


# Bootstrap performance metrics for September

pred_sep_df <- pred_jul_dec_df %>%
  filter(month_pub == 9) %>%
  select(biomasse_pred, bm_end_season, drought_list)

bootstrapped_metrics <- map_dfr(
  1:10000,
  ~ pred_sep_df[sample(1:nrow(pred_sep_df), 100, replace = T),] %>%
    calc_metrics(biomasse_pred, drought_list)
    )

bootstrapped_metrics %>%
  summarize(
    across(
      everything(),
      ~quantile(.x, probs = c(0.025, 0.975))
    )
  )

# Historical events graph

pred_sep_df %>%
  summarize(`False activations` = sum(biomasse_pred == TRUE & drought_list == FALSE),
            `Valid activations` = sum(biomasse_pred == TRUE & drought_list == TRUE),
            `Missed events` = sum(biomasse_pred == FALSE & drought_list == TRUE)) %>%
  pivot_longer(everything()) %>%
  mutate(dummy = 1,
         name = factor(name, levels = c("False activations", "Valid activations", "Missed events"))) %>%
  ggplot(aes(fill = name, y = value, x = dummy)) +
  geom_bar(position = "stack", stat = "identity", alpha = 0.9, width = 1) +
  scale_fill_manual(values = c("#f07470", "#1bb580", "#f07470")) +
  coord_flip() +
  geom_segment(y = 0, yend = 5, x = 1.6, xend = 1.6,
             color = "#444444",
             arrow = arrow(angle = 20, length = unit(0.1, "in"), ends = "both", type = "closed")) +
  geom_text(y = 2.5, x = 1.7, label = "Drought events", size = 5, fontface = "bold") +
  geom_segment(y = 2, yend = 7, x = 0.4, xend = 0.4,
             color = "#444444",
             arrow = arrow(angle = 20, length = unit(0.1, "in"), ends = "both", type = "closed")) +
  geom_text(y = 4.5, x = 0.3, label = "Activations", size = 5, fontface = "bold") +
  scale_x_continuous(limits = c(0.2, 1.8)) +
  scale_y_continuous(breaks = 0:7) +
  geom_text(x = 1, y = 1, label = "Missed\nevents", color = "white", size = 5) +
  geom_text(x = 1, y = 3.5, label = "Valid\nactivations", color = "white", size = 5) +
  geom_text(x = 1, y = 6, label = "False\nactivations", color = "white", size = 5) +
  theme_classic() +
  theme(axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position = "none") +
  labs(x = "",
       y = "Total events and activations",
       title = "Historical drought trigger activations and events",
       subtitle = "Threshold of 80% Biomasse anomaly published in September")
