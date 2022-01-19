plotCI <- function(dataframe, metric) {

dataframe <- data.frame(status = c('below_ci', 'ci', 'above_ci'), ## REMOVE ME dummy data for now
                   value = c(2, 3, 2),
                   dummy = c(1, 1, 1))

ci_df <- dataframe %>%
  mutate(status = factor(status, levels = c('below_ci', 'ci', 'above_ci')),
         value = round(value, 0))

# compute width of segments
below_width <- ci_df %>%
  filter(status == 'below_ci') %>%
  select(value) %>%
  as.numeric()

ci_width <- ci_df %>%
  filter(status == 'ci') %>%
  select(value) %>%
  as.numeric()

# compute position of value labels (low end, central estimate, high end)
central.x <- ci_df %>% # estimate (midpoint of CI)
  filter(status == 'ci') %>%
  mutate(ci_midpoint = value / 2,
         x_position = below_width + ci_midpoint) %>%
  select(x_position) %>%
  as.numeric()

low.x <- ci_df %>% # position of CI's low end
  filter(status == 'below_ci') %>%
  select(value) %>%
  as.numeric()

high.x <- below_width + ci_width # position of CI's high end

# plot
ci_color <- ifelse(metric %in% c('var_ci', 'det_ci'), "#1bb580", "#FF3333") # select green for detection and valid activation rates, red for the others

metric_plot <- ci_df %>%
  ggplot(aes(fill = status, y = value, x = dummy, labels = "low", "central", "high")) +
  geom_bar(position = "stack", stat = "identity", alpha = 0.9, width = 1) +
  scale_fill_manual(values = c("azure2", ci_color, "azure2")) +
  scale_x_continuous(limits = c(0, 5)) +
  coord_flip() +
  geom_segment(y = central.x, yend = central.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
  geom_segment(y = low.x, yend = low.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.5) + # low ci end line
  geom_segment(y = high.x, yend = high.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.5) + # high ci end line
  geom_label(aes(x = 2,
                 y = central.x,
                 label = central.x,
                 vjust = 1,
                 size = 1), # size refers to font size
             nudge_x = -0.1, # bring label closer to graph
             color = "black",
             fill = "white",
             fontface = 'bold',
             label.size = NA) + # removes border
  geom_label(aes(x = 2,
                 y = low.x,
                 label = low.x,
                 vjust = 1,
                 size = 1),
             nudge_x = -0.1,
             color = "black",
             fill = "white",
             label.size = NA) + # removes border
  geom_label(aes(x = 2,
                 y = high.x,
                 label = high.x,
                 vjust = 1,
                 size = 1),
             nudge_x = -0.1,
             color = "black",
             fill = "white",
             label.size = NA) + # removes border
  theme_economist_white(base_size = 11,
                        base_family = "sans",
                        gray_bg = F,
                        horizontal = F) +
  theme(
    legend.position = "none",
    plot.title = element_blank(),
    axis.line = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    axis.text = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank())

# Save original as png
filename <- paste0(metric, ".png")
png(paste0(plots_path, filename))
print(metric_plot)
dev.off()

# crop plot
metric_magick <- magick::image_read(paste0(plots_path, filename))
trimmed_metric <- magick::image_trim(metric_magick)
magick::image_write(trimmed_metric, path = paste0(plots_path, "trimmed_", metric, ".png"), format = "png")

}
