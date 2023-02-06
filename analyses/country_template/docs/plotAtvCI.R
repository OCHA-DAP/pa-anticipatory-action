  ## Script to plot the confidence intervals of probability of activation for individual triggers and activation timepoints
  ## Keeps light blue background to show 0-100 range

    plotAtvCI <- function(trigger_id) {

    # create df to receive individual segment dimensions
    seg_dims <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                                     lo_end = c(0,NA,NA,NA,NA),
                                     hi_end = c(NA,NA,NA,NA,100))
    seg_dims$segment <- factor(seg_dims$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)

    # subset performance metrics for this trigger_id
    perf_metrics_sub <- perf_metrics_data %>%
      filter(trigger == trigger_id & metric == 'atv')

    # compute segment low/high end values
    seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_95to68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_95'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_95to68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'low_end_68'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_68to95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_68'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_68to95'), 'hi_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']
    seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'] <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'high_end_95'), 'value']

    # plot all segments
    ci_color <- "#007ce1"
    ci_color_pale <- alpha(ci_color, 0.7)

    central.x <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] # value of central value of 95% confidence interval

p <- ggplot(seg_dims, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
    geom_rect(aes(fill = segment), colour = NA) +
    scale_fill_manual(values=c('seg_below_95' = 'azure2',
                               'seg_95to68' = ci_color_pale,
                               'seg_68' = ci_color,
                               'seg_68to95' = ci_color_pale,
                               'seg_above_95' = 'azure2')) +
    ylim(0, 10) +
    xlim(0, 100) +
    geom_segment(y = 0, yend = 1, x = central.x, xend = central.x, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     y = 1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     vjust = -0.1),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) + # removes border
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     y = 1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     vjust = -0.1),
                 size = 4,
                 nudge_x = 0,
                 color = "black",
                 fill = "white",
                 label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   y = 1,
                   label = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   vjust = -0.1),
               size = 4,
               nudge_x = 0,
               color = "black",
               fill = 'white',
               label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   y = 1,
                   label = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   vjust = -0.1),
               size = 4,
               nudge_x = 0,
               color = "black",
               fill = 'white',
               label.size = NA) +
    geom_label(aes(x = central.x,
                   y = 1,
                   label = central.x,
                   vjust = -0.1),
               size = 5,
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

  return(list(p = p, seg_dims = seg_dims))
}
