library(reticulate)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(grid)
library(viridis)
args = commandArgs(trailingOnly = TRUE)
dat_name = args[1]
t = as.numeric(args[2])
if (dat_name == "precip"){
  date_in = "Precipitation on 15 Nov 2023"
  date_out = "Precipitation on 18 Nov 2023"
  lower_bound = 0
  upper_bound = 14
  lb_resi = -5
  ub_resi = 9
  lb_se = 3
  ub_se = 20
  denormalize <- function(x, mn = 3.037, var = 2.514){
    return((x*var) + mn)
  }
  sd <- 2.514
  val_name <- "mm."
}else{
  date_in = "SST data on 14 Dec 2017"
  date_out = "SST data on 17 Dec 2017"
  lower_bound = 0
  upper_bound = 30
  lb_resi = -8
  ub_resi = 7
  lb_se = 3
  ub_se = 28
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
mdl_name <- c("CDNN","ConvLSTM","FNO","STDK")
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

### plot for true input/persistence

test_mat_in <- persistence[t,,]
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
g <- ggplot(data_for_plot) +
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
  coord_fixed() + ggtitle(paste0(date_in))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 2)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
ggsave(file = paste0("plots/",dat_name,"-persistence.png"), 
       plot = g, width = plot_saving_width ,
       height = plot_saving_height, units = "in")

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
g <- ggplot(data_for_plot) +
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
  coord_fixed() + ggtitle(paste0(date_out))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 2)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
ggsave(file = paste0("plots/",dat_name,"-true.png"), 
       plot = g, width = plot_saving_width ,
       height = plot_saving_height, units = "in")

##### plots for other comparing models

for(i in 1:3){

  r_pred <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[i],"-pred.npy")))  
  r_pred <- denormalize(r_pred)
  r_test <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[i],"-test.npy")))
  r_test <- denormalize(r_test)
  r_cov <- py_to_r(np$load(paste0("pred/",dat_name,"-",mdl_name[i],"-cov.npy")))
  r_cov <- sd*r_cov
  r_diff <- r_test - r_pred
  mse <- mean(r_diff^2)
  mpiw <- mean(2*z*r_cov)
  print(paste0("mse for ",mdl_name[i],":",mse))
  print(paste0("mpiw for ",mdl_name[i],":",mpiw))
  
  
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
  g <- ggplot(data_for_plot) +
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
    coord_fixed() + ggtitle(paste("Pred. mean"))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5), 
          plot.background = element_rect(fill = "white", color = "white", size = 3))
  ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-pred.png"), 
         plot = g, width = plot_saving_width ,
         height = plot_saving_height, units = "in")
  
  ### difference 
  
  test_mat_in <- r_diff[t,,]
  data_for_plot <- grid
  data_for_plot1 <- as.data.frame(as.table(test_mat_in))
  data_for_plot$obs <- pmax(lb_resi, pmin(data_for_plot1$Freq, ub_resi))
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g <- ggplot(data_for_plot) +
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
    coord_fixed() + ggtitle(paste("Pred. error"))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-residual.png"), 
         plot = g, width = plot_saving_width ,
         height = plot_saving_height, units = "in")
  
  ### standard error
  
  test_mat_in <- r_pred[t,,]
  data_for_plot <- grid
  data_for_plot$obs <- pmax(lb_se, pmin(2 * 1.96 * r_cov[t, ], ub_se))
  obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
  g <- ggplot(data = data_for_plot, 
              aes(x = Var2, 
                  y = Var1)) +
#     geom_raster(data = subset(data_for_plot, !is.na(obs)), 
#                 aes(fill=obs)) +
       geom_tile(aes(x = Var2, y = Var1, fill = obs)) +
    scale_fill_gradientn(colours = color_array.red, limits = c(lb_se,ub_se), 
                         guide = guide_colorbar(barwidth = bar_width, 
                                                barheight = bar_height,
                                                title.hjust = 0.5,
                                                title.vjust = 3.5),
                         breaks = round(seq(lb_se, ub_se, length.out = 3), digits = 0),
                         name = val_name) +
    theme_bw(base_size = base_size) + 
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
    xlab("") +
    ylab("") +
    coord_fixed() + ggtitle(paste("95% pred. interval"))+
    theme(legend.text=element_text(size=rel(l_t)), 
          legend.title = element_text(size=l_s, margin = margin(l = 2)),
          axis.title=element_text(size=a_s), 
          axis.text=element_blank(),
          plot.title = element_text(size = p_t, hjust = 0.5))
  ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-se.png"), 
         plot = g, width = plot_saving_width ,
         height = plot_saving_height, units = "in")
}

r_pred <- py_to_r(np$load(paste0("pred/",dat_name,"-STDK_med.npy")))  
r_pred <- denormalize(r_pred)
coords <- py_to_r(np$load(paste0("pred/",dat_name,"-coords-STDK.npy")))
r_lb <- py_to_r(np$load(paste0("pred/",dat_name,"-STDK_lb.npy")))
r_lb <- denormalize(r_lb)
r_ub <- py_to_r(np$load(paste0("pred/",dat_name,"-STDK_ub.npy")))
r_ub <- denormalize(r_ub)
r_test <- py_to_r(np$load(paste0("pred/",dat_name,"-STDK_testy.npy")))
r_test <- denormalize(r_test)
r_diff <- r_test - r_pred
r_bound <- r_ub - r_lb
mse <- mean(r_diff^2)
mpiw <- mean(r_bound)
print(paste0("mse for STDK:",mse))
print(paste0("mpiw for STDK:",mpiw))

##########################################################
###### creating plots for STDK ###########################
##########################################################
### mean 
i<- 4
data_for_plot <- as.data.frame(coords)
data_for_plot$obs <- r_pred[t,]
obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
g <- ggplot(data = data_for_plot, 
            aes(x = V1, 
                y = V2)) +
  geom_raster(data = subset(data_for_plot, !is.na(obs)), 
              aes(fill=obs))+
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
  coord_fixed() + ggtitle(paste("Pred. median"))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 2)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-pred.png"), 
       plot = g, width = plot_saving_width ,
       height = plot_saving_height, units = "in")

### difference 

data_for_plot$obs <- r_diff[t,]
obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
g <- ggplot(data = data_for_plot, 
            aes(x = V1, 
                y = V2)) +
  geom_raster(data = subset(data_for_plot, !is.na(obs)), 
              aes(fill=obs))+
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
  coord_fixed() + ggtitle(paste("Pred. error"))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 2)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-residual.png"), 
       plot = g, width = plot_saving_width ,
       height = plot_saving_height, units = "in")

### bound

data_for_plot$obs <- r_bound[t,]
obs_range <- range(data_for_plot$obs, na.rm = TRUE)
  print(obs_range)
  # breaks_manual <- round(seq(obs_range[1], obs_range[2], length.out = 4))
  min_point <- floor(obs_range[1])
  interval <- (obs_range[2] - obs_range[1])/4
  breaks_manual <- round(c(
                    min_point + interval,
                    min_point + 2*interval,
                    min_point + 3*interval), digits = 0)
g <- ggplot(data = data_for_plot, 
            aes(x = V1, 
                y = V2)) +
  geom_raster(data = subset(data_for_plot, !is.na(obs)), 
              aes(fill=obs))+
  scale_fill_gradientn(colours = color_array.red,
                       guide = guide_colorbar(barwidth = bar_width, 
                                              barheight = bar_height,
                                              title.hjust = 0.5,
                                              title.vjust = 3.5),
                       breaks = breaks_manual,
                       name = val_name) +
  theme_bw(base_size = base_size) +
  scale_x_continuous(expand = c(0, 0)) +
       scale_y_continuous(expand = c(0, 0)) +
  xlab("") +
  ylab("") +
  coord_fixed() + ggtitle(paste("95% pred. interval"))+
  theme(legend.text=element_text(size=rel(l_t)), 
        legend.title = element_text(size=l_s, margin = margin(l = 1)),
        axis.title=element_text(size=a_s), 
        axis.text=element_blank(),
        plot.title = element_text(size = p_t, hjust = 0.5))
ggsave(file = paste0("plots/",dat_name,"-",mdl_name[i],"-se.png"), 
       plot = g, width = plot_saving_width ,
       height = plot_saving_height, units = "in")





