plotCI <- function(trigger_id, metric_name) {

  # create df to receive individual segment dimensions
  segment_dimensions <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                                   lo_end = c(0,NA,NA,NA,NA),
                                   hi_end = c(NA,NA,NA,NA,100))
  segment_dimensions$segment <- factor(segment_dimensions$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)

  # select performance metrics for this trigger_id and metric_name
  perf_metrics_sub <- perf_metrics_data %>%
    filter(trigger == trigger_id & metric == metric_name))

  # compute segment low/high end values
  segment_dimensions[which(segment_dimensions$segment == 'seg_below_95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_95to68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_95to68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_68to95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_68to95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']
  segment_dimensions[which(segment_dimensions$segment == 'seg_above_95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']

  # plot all segments
  ci_color <- ifelse(metric %in% c('var', 'det'), "#1bb580", ifelse(metric %in% c('min', 'ful', 'atv'), "#007ce1" ,"#FF3333")) # select green for detection and valid activation rates, blue for framework probabilities, red for the others
  ci_color_pale <- alpha(ci_color, 0.7)

  central.x <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] # value of central value of 95% confidence interval

  #p <-
    ggplot(segment_dimensions, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
    geom_rect(aes(fill = segment), colour = NA) +
    scale_fill_manual(values=c('seg_below_95' = 'azure2',
                               'seg_95to68' = ci_color_pale,
                               'seg_68' = ci_color,
                               'seg_68to95' = ci_color_pale,
                               'seg_above_95' = 'azure2')) +
    xlim(0, 100) +
    geom_segment(y = 0, yend = 1, x = central.x, xend = central.x, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
    geom_label(aes(x = central.x,
                   y = 1.2,
                   label = central.x,
                   vjust = 1),
               size = 8, # font size
               nudge_x = 0.1, # bring label closer to graph
               color = "black",
               fill = "white",
               fontface = 'bold',
               label.size = NA)  # removes border


    geom_text(
      aes(x=low_end, y=0, label=low_end),
      size = 4, vjust = 0, hjust = 0, nudge_x = -2) +
    geom_text(
      aes(x=median_x, y=0.5, label=median_x),
      size = 8, vjust = 0, hjust = 0, nudge_x = 0)




# compute labels
low_end_label <- ci_widths_df %>% filter(point == 'low_end') %>% select(value) %>% as.numeric()
central_label <- ci_widths_df %>% filter(point == 'central') %>% select(value) %>% as.numeric()
high_end_label <- ci_widths_df %>% filter(point == 'high_end') %>% select(value) %>% as.numeric()

# plot
ci_color <- ifelse(metric %in% c('var', 'det'), "#1bb580", ifelse(metric %in% c('min', 'ful', 'atv'), "#007ce1" ,"#FF3333")) # select green for detection and valid activation rates, blue for framework probabilities, red for the others

metric_plot <- ci_df %>%
  ggplot(aes(fill = point, y = value, x = dummy, labels = "low", "central", "high")) +
  geom_bar(position = "stack", stat = "identity", alpha = 0.9, width = 1) +
  scale_fill_manual(values = c("azure2", ci_color, "azure2")) +
  scale_x_continuous(limits = c(0, 5)) +
  coord_flip() +
  geom_segment(y = central.x, yend = central.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
  geom_segment(y = low.x, yend = low.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.5) + # low ci end line
  geom_segment(y = high.x, yend = high.x, x = 0.5, xend = 1.6, color = "#444444", size = 0.5) + # high ci end line
  geom_label(aes(x = 2,
                 y = central.x,
                 label = central_label,
                 vjust = 1),
             size = 8, # font size
             nudge_x = 0.1, # bring label closer to graph
             color = "black",
             fill = "white",
             fontface = 'bold',
             label.size = NA) + # removes border
  geom_label(aes(x = 2,
                 y = low.x,
                 label = low_end_label,
                 vjust = 1),
             size = 7,
             nudge_x = 0,
             color = "black",
             fill = "white",
             label.size = NA) + # removes border
  geom_label(aes(x = 2,
                 y = high.x,
                 label = high_end_label,
                 vjust = 1),
             size = 7,
             nudge_x = 0,
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

trigger_label <- ifelse(trigger_id == 'framework', 'framework', paste0('trigger_', trigger_id))

filename <- paste0(trigger_label, "_", metric, "_ci.png")
png(filename = paste0("plots/", filename), width = 815, height = 410, units = "px")
print(metric_plot)
dev.off()

# crop plot
metric_magick <- magick::image_read(paste0("plots/", filename))
trimmed_metric <- magick::image_trim(metric_magick)
magick::image_write(trimmed_metric, path = paste0("plots/trimmed_", trigger_label, "_", metric, "_ci.png"), format = "png")

}
