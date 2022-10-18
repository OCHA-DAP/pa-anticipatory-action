## Example of CI for explanatory purposes
## Mock data

# generate df
seg_dims <- data.frame(segment = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'),
                       lo_end = c(0,35,42,58,65),
                       hi_end = c(35,42,58,65,100))
seg_dims$segment <- factor(seg_dims$segment, levels = c('seg_below_95', 'seg_95to68', 'seg_68', 'seg_68to95', 'seg_above_95'), ordered = TRUE)
central.x <- 50

# plot
p <- ggplot(seg_dims, aes(xmin = lo_end, xmax = hi_end, ymin = 0, ymax = 1)) +
  geom_rect(aes(fill = segment), colour = NA) +
  scale_fill_manual(values=c('seg_below_95' = 'azure2',
                             'seg_95to68' = alpha('grey', 0.6),
                             'seg_68' = 'grey',
                             'seg_68to95' = alpha('grey', 0.6),
                             'seg_above_95' = 'azure2')) +
  ylim(-5, 10) +
  xlim(0, 100) +
  geom_segment(y = 0, yend = 1, x = central.x, xend = central.x, color = "#444444", size = 0.75) + # central estimate line. Size refers to line thickness
  geom_label(y = -0.3,
             x = 75,
             label = "Most likely",
             size = 3,
             color = 'black',
             fill = alpha('white', 0.3),
             label.size = NA) +
  geom_label(y = -0.8,
             x = 75,
             label = "68% of the time",
             size = 3,
             color = 'black',
             fill = alpha('white', 0.3),
             label.size = NA) +
  geom_label(y = -1.3,
             x = 75,
             label = "95% of the time",
             size = 3,
             color = 'black',
             fill = alpha('white', 0.3),
             label.size = NA) +
  # central point
  geom_point(x = central.x,
             y = -0.3) +
  # arrows
  geom_segment(y = -0.8,
               yend = -0.8,
               x = 42,
               xend = 58,
               size = 0.5,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "both",
                             type = "closed")) +
  geom_segment(y = -1.3,
               yend = -1.3,
               x = 35,
               xend = 65,
               size = 0.5,
               arrow = arrow(length = unit(0.1, "in"),
                             angle = 20,
                             ends = "both",
                             type = "closed")) +
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

# save plot
filename <- paste0("explanatory_ci.png")
png(filename = paste0("../country_template/docs/", filename), width = 500, height = 400, units = "px")
print(p)
dev.off()

# trim and save trimmed plot
original_plot <- magick::image_read(paste0("../country_template/docs/", filename))
trimmed_plot <- magick::image_trim(original_plot)

# save trimmed plot
magick::image_write(trimmed_plot, path = paste0("../country_template/docs/", filename), format = "png")

