## Generates plots for each metric against its opposite (VAR vs FAR, DET vs MIS) to show the tradeoff between the pair elements.
## Also plots the framework activation likelihood bar for Any vs None scenarios

plotTradeoffCI <- function(trigger_id, left_metric_name) {

  # subset performance metrics data for the trigger
  perf_metrics_sub <- perf_metrics_data %>%
    filter(trigger == trigger_id & metric == left_metric_name)

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
  right_label <- ifelse(left_metric_name == 'var', 'False alarms',
                        ifelse(left_metric_name == 'det', 'Misses',
                               ifelse(left_metric_name == 'atv', 'None', "error")))
  left_label <- ifelse(left_metric_name == 'var', 'Valid',
                       ifelse(left_metric_name == 'det', 'Detections',
                              ifelse(left_metric_name == 'atv', 'Any', "error")))

  left_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#1bb580', '#007ce1')
  right_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#FF3333', '#007ce1')

  p <- seg_dims %>%
  ggplot(aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
  geom_rect(aes(fill = segment), colour = NA) +
  scale_fill_manual(values=c('seg_below_95' = left_colour,
                             'seg_95to68' = alpha('grey', 0.7),
                             'seg_68' = 'grey',
                             'seg_68to95' = alpha('grey', 0.7),
                             'seg_above_95' = right_colour)) +
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
               color = left_colour,
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "last",
                             type = "closed")) +
  geom_segment(y = 0.5, # right arrow
               yend = 0.5,
               x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] + 0.5,
               xend = 100,
               color = right_colour,
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "first",
                             type = "closed")) +
  geom_label(y = 0.5,
             x = 1,
              hjust = "inward",
              label = left_label,
              size = 4,
              color = left_colour,
              fill = alpha('white', 0.3)) +
  geom_label(y = 0.5,
            x = 99,
            hjust = "inward",
            label = right_label,
            size = 4,
            color = right_colour,
            fill = alpha('white', 0.3)) +
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

  return(list(p = p, seg_dims = seg_dims))
}
