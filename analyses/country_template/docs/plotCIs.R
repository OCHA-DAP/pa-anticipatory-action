plotCI <- function(trigger_id, metric_name) {

  # create df to receive individual segment dimensions
  segment_dimensions <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                                   lo_end = c(0,NA,NA,NA,NA),
                                   hi_end = c(NA,NA,NA,NA,100))
  segment_dimensions$segment <- factor(segment_dimensions$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)

  # select performance metrics for this trigger_id and metric_name
  perf_metrics_sub <- perf_metrics_data %>%
    filter(trigger == trigger_id & metric == metric_name)

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
  ci_color <- ifelse(metric_name %in% c('var', 'det'), "#1bb580", ifelse(metric_name == 'atv', "#007ce1" ,"#FF3333")) # select green for detection and valid activation rates, blue for activation probs, red for error metrics
  ci_color_pale <- alpha(ci_color, 0.7)

  central.x <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] # value of central value of 95% confidence interval

  p <- ggplot(segment_dimensions, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
    geom_rect(aes(fill = segment), colour = NA) +
    scale_fill_manual(values=c('seg_below_95' = 'azure2',
                               'seg_95to68' = ci_color_pale,
                               'seg_68' = ci_color,
                               'seg_68to95' = ci_color_pale,
                               'seg_above_95' = 'azure2')) +
    ylim(0, 10) +
    xlim(0, 100) +
    geom_segment(y = 0, yend = 1, x = central.x, xend = central.x, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
    geom_label(aes(x = segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'lo_end'],
                     y = 1,
                     label = segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'lo_end'],
                     vjust = -0.1),
                 size = 5,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) + # removes border
    geom_label(aes(x = segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'hi_end'],
                     y = 1,
                     label = segment_dimensions[which(segment_dimensions$segment == 'seg_68'), 'hi_end'],
                     vjust = -0.1),
                 size = 5,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) +
    geom_label(aes(x = segment_dimensions[which(segment_dimensions$segment == 'seg_below_95'), 'hi_end'],
                   y = 0,
                   label = segment_dimensions[which(segment_dimensions$segment == 'seg_below_95'), 'hi_end'],
                   vjust = -0.1),
               size = 4,
               nudge_x = 0,
               color = "black",
               fill = alpha('white', 0.2),
               label.size = NA) +
    geom_label(aes(x = segment_dimensions[which(segment_dimensions$segment == 'seg_above_95'), 'lo_end'],
                   y = 0,
                   label = segment_dimensions[which(segment_dimensions$segment == 'seg_above_95'), 'lo_end'],
                   vjust = -0.1),
               size = 4,
               nudge_x = 0,
               color = "black",
               fill = alpha('white', 0.2),
               label.size = NA) +
    geom_label(aes(x = central.x,
                   y = 1,
                   label = central.x,
                   vjust = -0.1),
               size = 6, # font size
               color = "black",
               fill = "white",
               fontface = 'bold',
               label.size = NA) +
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

  return(list(plot = p, segment_dims = segment_dimensions))
}
