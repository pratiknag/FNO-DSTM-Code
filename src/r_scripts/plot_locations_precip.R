# !/usr/bin/env Rscript

# rm(list = ls())
# args = commandArgs(trailingOnly=TRUE)
# file_path = args[1]
# setwd(file_path)
# setwd("/home/praktik/Desktop/Spatial_norm_flows/")
# library(geoR)
# library(MASS)
# library(fields)
library(ggplot2)
library(grid)
# library(rjson)
library(viridis)
# library(latex2exp)

plot_saving_width <- 5.5
plot_saving_height <- 4
base_size <- 12
l_s <- 13
l_t <- 10.2
a_s <- 10
p_t <- 13
bar_width <- 0.5
bar_height <- 5

## train-test data for single location
new_df = read.csv("datasets/dataset-10DAvg-nonnormalized.csv", header = T)
new_df = new_df[,c(2,3,4)]
new_df <- na.omit(new_df)
print(head(new_df))
unique.lonlat = unique(new_df[,c(1,2)])
print(head(unique.lonlat))
min_lon = min(unique.lonlat$LONGITUDE)
max_lon = max(unique.lonlat$LONGITUDE)

min_lat = min(unique.lonlat$LATITUDE)
max_lat = max(unique.lonlat$LATITUDE)

print(min_lon)
print(max_lon)
print(min_lat)
print(max_lat)
#### choice of 4 points for testing 

### plotting the points on map

world1 = map_data("world")
p1 <- ggplot() +
  geom_polygon(data = world1, aes(x = long, y = lat, group = group),
               colour = "darkgrey", fill = "grey", alpha = 1) +
  coord_cartesian(xlim = c(min_lon, max_lon), ylim = c(min_lat, max_lat)) +
  geom_point(data = unique.lonlat, aes(x = LONGITUDE, y = LATITUDE, colour = "Station locations"), 
             pch = 20, size = 1) +
  geom_rect(aes(xmin = 15, xmax = 25, ymin = 45, ymax = 54, color = "Forecasting region"),
            fill = NA, size = 2, inherit.aes = FALSE) +
  scale_color_manual(values = c("Station locations" = "steelblue4", "Forecasting region" = "red3")) +
  theme_bw(base_size = base_size) +
  labs(x = "Longitude (deg)", 
       y = "Latitude (deg)") +
  theme(
    legend.title = element_blank(),
    legend.position = "bottom", 
    legend.text = element_text(size = l_t),
    axis.title = element_text(size = a_s), 
    axis.text = element_text(size = a_s)
  )
ggsave("plots/observed_area-precip.png", 
       plot = p1, width = plot_saving_width,
       height = plot_saving_height, units = "in")