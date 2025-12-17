library(reticulate)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(gridExtra)
library(grid)
library(viridis)
library(patchwork)
args = commandArgs(trailingOnly = TRUE)
dat_name = args[1]
t = as.numeric(args[2])
if (dat_name == "precip"){
  date_in = "15th Nov 2023"
  date_out <- list(
  "7" = "15th Nov 2023",
  "8" = "25th Nov 2023",
  "9" = "5th Dec 2023",
  "10" = "15th Dec 2023",
  "11" = "25th Dec 2023",
  "12" = "5th Jan 2024"
)
  lower_bound = 0
  upper_bound = 14
  lb_resi = -4
  ub_resi = 7
  lb_se = 0
  ub_se = 4
  denormalize <- function(x, mn = 3.037, var = 2.514){
    return((x*var) + mn)
  }
  sd <- 2.514
  val_name <- "mm."
}else{
  date_in = "14th Dec 2017"
  date_out <- list(
  "41" = "17th Dec 2017",
  "42" = "18th Dec 2017",
  "43" = "19th Dec 2017",
  "44" = "20th Dec 2017",
  "45" = "21th Dec 2017",
  "46" = "22th Dec 2017",
  "47" = "23th Dec 2017",
  "48" = "24th Dec 2017",
  "49" = "25th Dec 2017",
  "50" = "26th Dec 2017"
)
  lower_bound = 0
  upper_bound = 30
  lb_resi = -10
  ub_resi = 9
  lb_se = 0
  ub_se = 9
  denormalize <- function(x, mn = 23, var = 4.11){
    return((x*var) + mn)
  }
  sd <- 4.11
  val_name <- "degC"
}
# dat_name = "sst"
# t = 1
# Load Python modules via reticulate
# scipy_stats <- import("scipy.stats")
np <- import("numpy")

nasa_palette <- c("#03006d","#02008f","#0000b6","#0001ef","#0000f6",
                  "#0428f6","#0b53f7","#0f81f3",
                  "#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b",
                  "#85fd4e","#aefc2a","#e9fc0d","#f6da0c","#f5a009",
                  "#f6780a","#f34a09","#f2210a","#f50008","#d90009",
                  "#a80109","#730005")
bwr <- colorRampPalette(c("blue","white","red"))
wr <- colorRampPalette(c("brown","blue","green"))
color_array <- bwr(100)
color_array.red <- wr(100)
mdl_name <- c("CDNN","FNO")
l_s <- 13
l_t <- 10.2
a_s <- 13
p_t <- 13
plot_saving_width <- 3
plot_saving_height <- 2.5
base_size <- 1.2
bar_width <- 0.5
bar_height <- 5
S<- 64
z <- 1.96
x <- seq(0,1,length.out = S)
y <- seq(0,1,length.out = S)
grid <- expand.grid(Var1 = x, Var2 = y)

data_test <- py_to_r(np$load(paste0("datasets/", dat_name, "_data-test-6t.npy")))

persistence <- data_test[,,,1]
persistence <- denormalize(persistence)
out <- data_test[,,,6]
out <- denormalize(out)
r_diff_per <- persistence - out
mse <- mean(r_diff_per^2)
print(paste0("mse for persistence model:",mse))


### plot for true output

test_mat_in <- out[t,,]
data_for_plot <- grid
data_for_plot1 <- as.data.frame(as.table(test_mat_in))
data_for_plot$obs <- data_for_plot1$Freq
obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
g1 <- ggplot(data_for_plot) +
  geom_tile(aes(x = Var1, y = Var2, fill = obs)) +
  scale_fill_gradientn(colours = nasa_palette,     limits = c(lower_bound, upper_bound), 
                       guide = guide_colorbar(barwidth = bar_width, 
                                              barheight = bar_height,
                                              title.hjust = 1,
                                              title.vjust = 3.5),
                       breaks = round(seq(lower_bound, upper_bound, length.out = 3), digits = 0),
                       name = val_name) +
  theme_bw(base_size = base_size) +
  scale_x_continuous(expand = c(0, 0)) +
scale_y_continuous(expand = c(0, 0)) +
  xlab("") +
  ylab("") +
  coord_fixed() + ggtitle(paste0("True at ", date_out[[as.character(t)]]))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 2)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
# ggsave(file = paste0("anim/",dat_name,"-true-",t,".png"), 
#        plot = g, width = plot_saving_width ,
#        height = plot_saving_height, units = "in")

##### plots for other comparing models

  r_pred <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[1],"-pred.npy")))  
  r_pred <- denormalize(r_pred)
  r_test <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[1],"-test.npy")))
  r_test <- denormalize(r_test)
  r_cov <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[1],"-cov.npy")))
  r_cov <- sd*r_cov
  r_diff <- r_test - r_pred
  mse <- mean(r_diff^2)
  mpiw <- mean(2*z*r_cov)
  print(paste0("mse for ",mdl_name[1],":",mse))
  print(paste0("mpiw for ",mdl_name[1],":",mpiw))
  
  
##########################################################
###### creating plots ####################################
##########################################################
  ### mean 
  
  test_mat_in <- r_pred[t,,]
  data_for_plot <- grid
  data_for_plot1 <- as.data.frame(as.table(test_mat_in))
  data_for_plot$obs <- data_for_plot1$Freq
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g2 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = obs)) +
    scale_fill_gradientn(colours = nasa_palette,     limits = c(lower_bound, upper_bound), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                              title.hjust = 1,
                                                title.vjust = 3.5),
                         breaks = round(seq(lower_bound, upper_bound, length.out = 3), digits = 0),
                         name = val_name) +
    theme_bw(base_size = base_size) +
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste("CNN-IDE"))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5), 
          plot.background = element_rect(fill = "white", color = "white", size = 3))
  
  ### difference 
  
  test_mat_in <- r_diff[t,,]
  data_for_plot <- grid
  data_for_plot1 <- as.data.frame(as.table(test_mat_in))
  data_for_plot$obs <- data_for_plot1$Freq
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g3 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = obs)) +
    scale_fill_gradientn(colours = color_array, limits = c(lb_resi,ub_resi), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                                title.hjust = 1,
                                                title.vjust = 3.5),
                         breaks = round(seq(lb_resi, ub_resi, length.out = 3), digits = 0),
                         name = val_name) +
    theme_bw(base_size = base_size) + 
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste(" "))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  
  ### standard error
  
  test_mat_in <- r_pred[t,,]
  data_for_plot <- grid
  data_for_plot$obs <- r_cov[t,]
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g4 <- ggplot(data = data_for_plot, 
              aes(x = Var2, 
                  y = Var1)) +
#     geom_raster(data = subset(data_for_plot, !is.na(obs)), 
#                 aes(fill=obs)) +
       geom_tile(aes(x = Var2, y = Var1, fill = obs)) +
    scale_fill_gradientn(colours = color_array.red, limits = c(lb_se,ub_se), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                              title.hjust = 2,
                                                title.vjust = 3.5),
                         breaks = round(seq(lb_se, ub_se, length.out = 3), digits = 0),
                         name = "se.") +
    theme_bw(base_size = base_size) + 
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste(" "))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))



r_pred <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[2],"-pred.npy")))  
  r_pred <- denormalize(r_pred)
  r_test <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[2],"-test.npy")))
  r_test <- denormalize(r_test)
  r_cov <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[2],"-cov.npy")))
  r_cov <- sd*r_cov
  r_diff <- r_test - r_pred
  mse <- mean(r_diff^2)
  mpiw <- mean(2*z*r_cov)
  print(paste0("mse for ",mdl_name[1],":",mse))
  print(paste0("mpiw for ",mdl_name[1],":",mpiw))
  
  
##########################################################
###### creating plots ####################################
##########################################################
  ### mean 
  
  test_mat_in <- r_pred[t,,]
  data_for_plot <- grid
  data_for_plot1 <- as.data.frame(as.table(test_mat_in))
  data_for_plot$obs <- data_for_plot1$Freq
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g5 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = obs)) +
    scale_fill_gradientn(colours = nasa_palette,     limits = c(lower_bound, upper_bound), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                              title.hjust = 1,
                                                title.vjust = 3.5),
                         breaks = round(seq(lower_bound, upper_bound, length.out = 3), digits = 0),
                         name = val_name) +
    theme_bw(base_size = base_size) +
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste("FNO-DST"))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5), 
          plot.background = element_rect(fill = "white", color = "white", size = 3))
  
  ### difference 
  
  test_mat_in <- r_diff[t,,]
  data_for_plot <- grid
  data_for_plot1 <- as.data.frame(as.table(test_mat_in))
  data_for_plot$obs <- data_for_plot1$Freq
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g6 <- ggplot(data_for_plot) +
    geom_tile(aes(x = Var1, y = Var2, fill = obs)) +
    scale_fill_gradientn(colours = color_array, limits = c(lb_resi,ub_resi), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                                title.hjust = 1,
                                                title.vjust = 3.5),
                         breaks = round(seq(lb_resi, ub_resi, length.out = 3), digits = 0),
                         name = val_name) +
    theme_bw(base_size = base_size) + 
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste(" "))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  
  ### standard error
  
  test_mat_in <- r_pred[t,,]
  data_for_plot <- grid
  data_for_plot$obs <- r_cov[t,]
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g7 <- ggplot(data = data_for_plot, 
              aes(x = Var2, 
                  y = Var1)) +
#     geom_raster(data = subset(data_for_plot, !is.na(obs)), 
#                 aes(fill=obs)) +
       geom_tile(aes(x = Var2, y = Var1, fill = obs)) +
    scale_fill_gradientn(colours = color_array.red, limits = c(lb_se,ub_se), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                              title.hjust = 2,
                                                title.vjust = 3.5),
                         breaks = round(seq(lb_se, ub_se, length.out = 3), digits = 0),
                         name = "se.") +
    theme_bw(base_size = base_size) + 
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste(" "))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))

empty_plot <- ggplot() +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 40, 0, 0)
  ) +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  annotate(
    "text",
    x = 0.5, y = 0.5,
    label = "Prediction error",
    size = 5,
    color = "black",
    hjust = 0.5, vjust = 0.5
  )

empty_plot1 <- ggplot() +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    plot.margin = margin(0, 40, 0, 0)
  ) +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  annotate(
    "text",
    x = 0.5, y = 0.5,
    label = "Standard error",
    size = 5,
    color = "black",
    hjust = 0.5, vjust = 0.5
  )
# plots <- lapply(list(g1, g2, g3, g4, g5, g6, g7),
#                 \(p) p + theme(plot.margin = margin(0, 0, 0, 0)))

# # Combine plots tightly (no extra arguments in ggsave!)
# final <- ggarrange(
#   plots[[1]], plots[[2]], plots[[5]],
#   empty_plot, plots[[3]], plots[[6]], 
#   empty_plot, plots[[4]],plots[[7]],
#   ncol = 3, nrow = 3,
#   align = "hv",
#   widths = c(1, 1, 1),
#   heights = c(1, 1, 1),
#   padding = 0
# )

# #  final <- ggarrange(
# #   g1, g2, g5,
# #   NULL, g3, g6,
# #   NULL, g4, g7,
# #   ncol = 3, nrow = 3,
# #   # common.legend = TRUE,
# #   # # legend = "bottom",
# #   align = "hv"
# # )

# ggsave(
#   file = paste0("anim/", dat_name, "-", t, ".png"),
#   plot = final,
#   width = 10,   # inches → fills Beamer slide nicely
#   height = 5.625, # 10 * (9/16) → matches 16:9 aspect ratio
#   units = "in",
#   dpi = 300)

# Remove margins from each plot
plots <- list(g1, g2, g3, g4, g5, g6, g7, empty_plot, empty_plot1)

# Construct the 3x3 layout with plot_spacer() for empty cells
final <- (
  (plots[[1]] | plots[[2]] | plots[[5]]) /
  (plots[[8]] | plots[[3]] | plots[[6]]) /
  (plots[[9]] | plots[[4]] | plots[[7]])
) 

# Save for Beamer
ggsave(
  filename = paste0("anim/", dat_name, "-", t, ".png"),
  plot = final,
  width = 8, height = 5.625, units = "in",
  dpi = 300, bg = "white"
)