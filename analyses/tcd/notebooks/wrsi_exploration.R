library(tidyverse)
library(ggthemes)
library(lubridate)
library(gridExtra)
library(yardstick)

aa_dir <- Sys.getenv("AA_DATA_DIR")

df <- read_csv(file.path(
  aa_dir, "public", "processed", "general", "wrsi", "range_anom.csv"
))

df <- mutate(df,
             year = year(time),
             wrsi = wrsi*100,
             drought = factor(year %in% c(2001, 2004, 2009, 2011, 2017)))

df %>%
  ggplot(aes(x = time, y = wrsi, group = year)) +
  geom_area(alpha = 0.75, fill = rgb(238, 88, 89, maxColorValue=255)) +
  scale_x_date(date_breaks = "1 month", date_labels = "%m") +
  facet_wrap(~year, scales = "free_x") +
  geom_hline(yintercept = 12,
             alpha = 0.5) +
  theme_minimal() +
  labs(title = "WRSI anomaly, May to November",
       subtitle = "Central Chad region, rangeland",
       y = "WRSI (anomaly)",
       x = "Month")

df %>%
  group_by(year) %>%
  filter(wrsi >= 12) %>%
  summarize(time = min(time)) %>%
  arrange(time) %>%
  grid.table()

pred_df <- df %>%
  group_by(year) %>%
  mutate(pred = factor(any(wrsi >= 12))) %>%
  distinct(year, pred, drought) %>%
  ungroup()

conf_mat(pred_df, drought, pred) %>%
  autoplot(type = "heatmap") +
  labs(title = "WRSI trigger, threshold of 13%")



df %>%
  group_by(year) %>%
  filter(wrsi >= 13) %>%
  summarize(time = min(time)) %>%
  arrange(time) %>%
  grid.table()

pred_df <- df %>%
  group_by(year) %>%
  mutate(pred = factor(any(wrsi >= 13))) %>%
  distinct(year, pred, drought) %>%
  ungroup()

conf_mat(pred_df, drought, pred) %>%
  autoplot(type = "heatmap")
