plotTradeoffCI <- function(trigger_id, metric_name) {

  # subset performance metrics data for the trigger

  #metrics <- if(trigger_id %in% trigger_list) c('var', 'det') else 'atv'

  perf_metrics_sub <- perf_metrics_data %>%
    filter(trigger == trigger_id & metric == metric_name)

   # Create df with segment widths for double-metric bars
  seg_dims <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                             lo_end = c(0,NA,NA,NA,NA),
                             hi_end = c(NA,NA,NA,NA,100))
  seg_dims$segment <- factor(seg_dims$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)

  seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_95to68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_95to68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68to95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68to95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']

  # plot bar
  right_label <- ifelse(metric_name == 'var', 'False alarms',
                        ifelse(metric_name == 'det', 'Misses',
                               ifelse(metric_name == 'atv', 'None', "error")))
  left_label <- ifelse(metric_name == 'var', 'Valid',
                       ifelse(metric_name == 'det', 'Detections',
                              ifelse(metric_name == 'atv', 'Any', "error")))

  tradeoff_bar <- seg_dims %>%
  ggplot(aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
  geom_rect(aes(fill = segment), colour = NA) +
  scale_fill_manual(values=c('seg_below_95' = '#1bb580',
                             'seg_95to68' = alpha('grey', 0.7),
                             'seg_68' = 'grey',
                             'seg_68to95' = alpha('grey', 0.7),
                             'seg_above_95' = '#FF3333')) +
  ylim(0, 10) +
  xlim(0, 100) +
  geom_segment(y = -0.5, # central line
               yend = 1.5,
               x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'],
               xend = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'],
               color = "#444444",
               size = 0.75) +
  geom_segment(y = 0.5, # left arrow
               yend = 0.5,
               x = 0,
               xend = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] - 0.5,
               color = "#1bb580",
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "last",
                             type = "closed")) +
  geom_segment(y = 0.5, # right arrow
               yend = 0.5,
               x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] + 0.5,
               xend = 100,
               color = "#FF3333",
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "first",
                             type = "closed")) +
  geom_text(y = 0.5,
            x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value'] + ((100-perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value'])/2),
            label = right_label,
            size = 4,
            # fontface = "bold",
            color = "white") +
  geom_text(y = 0.5,
            x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']/2,
            label = left_label,
            size = 4,
            # fontface = "bold",
            color = "white") +
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

return(tradeoff_bar)
}
