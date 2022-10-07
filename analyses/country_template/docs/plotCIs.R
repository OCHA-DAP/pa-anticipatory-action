## Generates the CI plots of VAR, DET, FAR, MIS, and individual activation scenarios to position above/below the tradeoff bars
## The CI plots are white below/above 95th CI endings
## Only requires the LEFT metric name (var, det, or atv) to produce the plots for the top/bottom CIs

plotCI <- function(trigger_id, left_metric_name) {

  # create df to receive individual segment dimensions
  seg_dims <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                                   lo_end = c(0,NA,NA,NA,NA),
                                   hi_end = c(NA,NA,NA,NA,100))
  seg_dims$segment <- factor(seg_dims$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)

  # select performance metrics for this trigger_id and metric_name
  perf_metrics_sub <- perf_metrics_data %>%
    filter(trigger == trigger_id & metric == left_metric_name)

  # compute segment low/high end values
  seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_95to68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_95to68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68to95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_68to95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']
  seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']

  # Plot parameters
  top_ci_color <- ifelse(left_metric_name %in% c('var', 'det'), "#1bb580", ifelse(left_metric_name == 'atv', "#007ce1" ,"white")) # select green for detection and valid activation rates, blue for activation probs, red for error metrics
  top_ci_color_pale <- alpha(top_ci_color, 0.7)

  bottom_ci_color <- ifelse(left_metric_name %in% c('var', 'det'), "#FF3333", ifelse(left_metric_name == 'atv', "#007ce1" ,"white"))
  bottom_ci_color_pale <- alpha(bottom_ci_color, 0.7)

  central.x <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] # value of central value of 95% confidence interval

  # plot according to position vis-a-vis tradeoff bar
top_p <- ggplot(seg_dims, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
    geom_rect(aes(fill = segment), colour = NA) +
    scale_fill_manual(values=c('seg_below_95' = 'white',
                               'seg_95to68' = top_ci_color_pale,
                               'seg_68' = top_ci_color,
                               'seg_68to95' = top_ci_color_pale,
                               'seg_above_95' = 'white')) +
    ylim(0, 10) +
    xlim(0, 100) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) + # removes border
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   y = 1,
                   label = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   vjust = 0),
               size = 3,
               nudge_x = 0,
               color = "black",
               label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   y = 1,
                   label = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   vjust = 0),
               size = 3,
               nudge_x = 0,
               color = "black",
               label.size = NA) +
    geom_label(aes(x = central.x,
                   y = 1.3,
                   label = central.x,
                   vjust = 0.2),
               size = 6,
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


bottom_p <- ggplot(seg_dims, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
      geom_rect(aes(fill = segment), colour = NA) +
      scale_fill_manual(values=c('seg_below_95' = 'white',
                                 'seg_95to68' = bottom_ci_color_pale,
                                 'seg_68' = bottom_ci_color,
                                 'seg_68to95' = bottom_ci_color_pale,
                                 'seg_above_95' = 'white')) +
      ylim(-1.5, 10) +
      xlim(0, 100) +
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     y = -1.1,
                     label = 100 - seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     vjust = -0.2),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) + # removes border
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     y = -1.1,
                     label = 100 -seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     vjust = -0.2),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) +
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                     y = -0.8,
                     label = 100 - seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                     vjust = 0),
                 size = 3,
                 nudge_x = 0,
                 color = "black",
                 label.size = NA) +
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                     y = -0.8,
                     label = 100 - seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                     vjust = 0),
                 size = 3,
                 nudge_x = 0,
                 color = "black",
                 label.size = NA) +
      geom_label(aes(x = central.x,
                     y = -1.3,
                     label = 100 - central.x,
                     vjust = 0),
                 size = 6,
                 color = "black",
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


  return(list(top_p = top_p, bottom_p = bottom_p, seg_dims = seg_dims))
}
