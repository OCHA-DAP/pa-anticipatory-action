## Generates plots for each metric against its opposite (VAR vs FAR, DET vs MIS) to show the tradeoff between the paired elements.
## Also plots the framework activation likelihood bar for Any vs None scenarios.

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

  # Create labels
  right_label <- ifelse(left_metric_name == 'var', 'False alarms',
                        ifelse(left_metric_name == 'det', 'Misses',
                               ifelse(left_metric_name == 'atv', 'None', "error")))
  left_label <- ifelse(left_metric_name == 'var', 'Valid',
                       ifelse(left_metric_name == 'det', 'Detections',
                              ifelse(left_metric_name == 'atv', 'At least one', "error")))

  left_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#1bb580', '#007ce1')
  top_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#1bb580', '#007ce1')

  right_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#FF3333', 'black')
  bottom_colour <- ifelse(left_metric_name %in% c('var', 'det'), '#FF3333', 'black')

  central.x <- perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'] # value of central value of 95% confidence interval

  # plot
p <-  seg_dims %>%
  ggplot(aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
  geom_rect(aes(fill = segment), colour = NA) +
  scale_fill_manual(values= c('seg_below_95' = 'azure2',
                             'seg_95to68' = alpha('grey', 0.6),
                             'seg_68' = 'grey',
                             'seg_68to95' = alpha('grey', 0.6),
                             'seg_above_95' = 'azure2')) +
  ylim(-1, 9) +
  xlim(0, 100) +
  # left arrow
  geom_segment(y = 1.2,
               yend = 1.2,
               x = 0,
               xend = 5,
               color = left_colour,
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "last",
                             type = "closed")) +
    # right arrow
    geom_segment(y = -0.2,
               yend = -0.2,
               x = 95,
               xend = 100,
               color = right_colour,
               size = 1,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "first",
                             type = "closed")) +
    # left metric name
    geom_label(y = 1.7,
             x = 1,
              hjust = "inward",
              label = left_label,
              size = 4,
              color = left_colour,
              fill = alpha('white', 0.3)) +
    # right metric name
  geom_label(y = -0.7,
            x = 99,
            hjust = "inward",
            label = right_label,
            size = 4,
            color = right_colour,
            fill = alpha('white', 0.3)) +

  # bottom values
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                   y = -0.5,
                   label = 100 - seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                   vjust = 0),
               size = 4,
               nudge_x = 0,
               color = bottom_colour,
               fontface = 'bold',
               fill = NA,
               label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                   y = -0.5,
                   label = 100 -seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                   vjust = 0),
               size = 4,
               nudge_x = 0,
               color = bottom_colour,
               fontface = 'bold',
               fill = NA,
               label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   y = -0.5,
                   label = 100 - seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                   vjust = 0),
               size = 4,
               nudge_x = 0,
               color = bottom_colour,
               label.size = NA) +
    geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   y = -0.5,
                   label = 100 - seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                   vjust = 0),
               size = 4,
               nudge_x = 0,
               color = bottom_colour,
               label.size = NA) +
    geom_label(aes(x = central.x,
                   y = -0.5,
                   label = 100 - central.x,
                   vjust = 0.25),
               size = 5,
               color = bottom_colour,
               fontface = 'bold',
               fill = "white",
               label.size = NA) +

  # top values
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'lo_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = top_colour,
                 fontface = 'bold',
                 label.size = NA) + # removes border
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_68'), 'hi_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = top_colour,
                 fontface = 'bold',
                 label.size = NA) +
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_below_95'), 'hi_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = top_colour,
                 label.size = NA) +
      geom_label(aes(x = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                     y = 1.1,
                     label = seg_dims[which(seg_dims$segment == 'seg_above_95'), 'lo_end'],
                     vjust = 0),
                 size = 4,
                 nudge_x = 0,
                 color = top_colour,
                 label.size = NA) +
      geom_label(aes(x = central.x,
                     y = 1.1,
                     label = central.x,
                     vjust = -0.2),
                 size = 5,
                 color = top_colour,
                 fontface = 'bold',
                 fill = "white",
                 label.size = NA) +
  # central line
  geom_segment(y = -0.2, # central line
               yend = 1.2,
               x = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'],
               xend = perf_metrics_sub[which(perf_metrics_sub$upoint == 'central_95'), 'value'],
               color = "#444444",
               size = 0.75) +
  # theme
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
