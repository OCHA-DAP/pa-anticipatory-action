source(here::here("exploration", "moz_data_prep.R"))

# quick edit of graphs from `moz_exploration_granular_data.Rmd`
# for inclusion in Google Slides presentation to share with
# the government of Mozambique

# general situation

general_p <- overall_df %>%
  filter(epi_type == "cholera") %>%
  ggplot(aes(y = cases, x = year_week)) + 
  geom_area(fill = "#f28080") +
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.ticks.y = element_blank(),
        axis.title = element_text(face = "bold"),
        axis.text.y = element_text(face = "bold")) +
  scale_x_continuous(breaks = seq(1, 51*5, by = 51),
                     labels = ~ 2017 + (.x %/% 51)) +
  labs(y = "Cases",
       x = "Date",
       title = "Cholera cases in Cabo Delgado and Nampula",
       subtitle = "2017 - 2021")

general_p

# Add CERF allocation
cerf_dates <- as.Date(c("2020-04-23", "2020-10-27"))
epi_weeks <- lubridate::epiweek(cerf_dates) 
year_weeks <- overall_df$year_week[match(paste0("2020, ", epi_weeks), overall_df$date)]

general_p +
  geom_segment(
    x = year_weeks[1],
    xend = year_weeks[1],
    y = 0,
    yend = 320
  ) +
  geom_segment(
    x = year_weeks[1] - 40,
    xend = year_weeks[1],
    y = 320,
    yend = 320
  ) +
  geom_text(
    x = year_weeks[1] - 38,
    y = 350,
    label = "Rapid response",
    hjust = 0,
    check_overlap = T
  ) +
  geom_segment(
    x = year_weeks[2],
    xend = year_weeks[2],
    y = 0,
    yend = 400
  )  +
geom_segment(
  x = year_weeks[2] - 64,
  xend = year_weeks[2],
  y = 400,
  yend = 400
) +
  geom_text(
    x = year_weeks[2] - 64,
    y = 430,
    label = "Underfunded emergencies",
    hjust = 0,
    check_overlap = T
  )
  
# look at potential CERF allocation in 2021
general_p +
  geom_segment(
    x = 207,
    xend = 207,
    y = 0,
    yend = 350
  )  +
  geom_segment(
    x = 207 - 70,
    xend = 207,
    y = 350,
    yend = 350
  ) +
  geom_text(
    x = 207-68,
    y = 355,
    label = "Trigger 2 weeks before peak\n350 case threshold",
    hjust = 0,
    check_overlap = T
  )
