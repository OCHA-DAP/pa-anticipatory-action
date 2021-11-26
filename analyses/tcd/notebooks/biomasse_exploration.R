library(tidyverse)
library(ggthemes)

aa_dir <- Sys.getenv("AA_DATA_DIR")

df <- read_csv(file.path(
  aa_dir, "public", "processed", "general", "biomasse", "biomasse_tcd_trigger_dekad_10.csv"
))

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

df %>%
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
