library(reticulate)
library(ggplot2)
library(dplyr)
library(cowplot)
library(reshape2)

# Load Python modules via reticulate
np <- import("numpy")
# scipy_stats <- import("scipy.stats")
ntest <- 50

# Define paths
plots_dir <- "plots"
burger_dir <- file.path(plots_dir, "burger_plots")

# Check and create 'plots' if it doesn't exist
if (!dir.exists(plots_dir)) {
  dir.create(plots_dir)
  cat("Created directory:", plots_dir, "\n")
}

# Check and create 'burger_plots' if it doesn't exist
if (!dir.exists(burger_dir)) {
  dir.create(burger_dir)
  cat("Created directory:", burger_dir, "\n")
}

data_nonlcl <- np$load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
data_lcl <- np$load("datasets/generated_1d_data_Burger-04_matern.npy")
a_text=10
a_title=10
plot_saving_height <- 1.6
plot_saving_width <- 2.8
base_size <- 7.5
legend_text <- 6
legend_key <- 0.17
legend_spacing <- 0.05
# Load data
z <- 1.96
pred_name <- c("pred/burger-FNO-lcl_dat-lcl", "pred/burger-FNO-lcl_dat-nonlcl",
                    "pred/burger-FNO-nonlcl_dat-lcl", "pred/burger-FNO-nonlcl_dat-nonlcl")
data_name <- c(0,1,0,1)
index <- c(1,37,45)  # Time steps to plot
generate_plots <- function(i,j){
    x <- (0:(2^11 - 1)) / 2^11
    print(length(x))
    if(data_name[i] == 0){
        y_test <- data_lcl[(dim(data_lcl)[1] - ntest + 1):dim(data_lcl)[1], , 10]
        y_inputs <- data_lcl[(dim(data_lcl)[1] - ntest + 1):dim(data_lcl)[1], , 1:5]
    }
    else{
        # y_test <- data_nonlcl[(dim(data_nonlcl)[1] - ntest + 1):dim(data_nonlcl)[1], , 10]
        y_test <- np$load("pred/burger_test_nonlcl.npy")
        y_inputs <- data_nonlcl[(dim(data_nonlcl)[1] - ntest + 1):dim(data_nonlcl)[1], , 1:5]
    }
    
    if(pred_name[i] == "pred/burger-STDK_med-nonlcl_dat-nonlcl"){
        pred <- np$load(sprintf("%s.npy",pred_name[i]))
        lb <- np$load("pred/burger-STDK_lb-nonlcl_dat-nonlcl.npy")
        ub <- np$load("pred/burger-STDK_ub-nonlcl_dat-nonlcl.npy")
        # Create grid for spatial axis
        sub("pred/", "", pred_name[i])
        # Compute MSE and MPIW
        mse <- mean((y_test - pred)^2)
        mpi_width <- mean(ub-lb)
        cat(sprintf("%s MSPE: ",sub("pred/", "", pred_name[i])), mse, "\n")
        cat("MPIW:", mpi_width, "\n")
        lower <- lb
        upper <- ub
        breaks <- 4
    }
    else{
        pred <- np$load(sprintf("%s.npy",pred_name[i]))
        var <- np$load(sprintf("%s_cov.npy",pred_name[i]))
        # Compute MSE and MPIW
        mse <- mean((y_test - pred)^2)
        mpi_width <- mean(2 * z * var)
        cat(sprintf("%s MSPE: ",sub("pred/", "", pred_name[i])), mse, "\n")
        cat("MPIW:", mpi_width, "\n")

        lb <- pred - z * var
        ub <- pred + z * var
        breaks <- 3
    }
    df <- data.frame(
        x = x,
        True = y_test[j + 1, ],
        Pred = pred[j+1,],
        Lower = lb[j+1,],
        Upper = ub[j+1,],
        Diff = pred[j+1,] - y_test[j + 1, ]
    )
    inputs_mat <- y_inputs[j + 1, , 1:5]   # dimensions: [space, 5]
    
    # Convert to long format
    inputs_df <- data.frame(x = x, inputs_mat)
    inputs_df <- melt(inputs_df, id.vars = "x", variable.name = "Input", value.name = "value")
    
    inputs_df <- inputs_df[order(inputs_df$x), ]
    # Main prediction plot with uncertainty band
    breaks <- 4
    limits <- c(-1.5,2)
    p1 <- ggplot(df, aes(x = x)) +
      # Input lines (faint, blue)
        geom_line(
          data = inputs_df,
          aes(y = value, group = Input, color = "Inputs"),
          linewidth = 0.2,
          alpha = 0.3
        ) +
        geom_line(aes(y = True, color = "True"), linetype = "dotted", linewidth = 0.6) +
        geom_line(aes(y = Pred, color = "Pred."),  linewidth = 0.6) +
        geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.5, fill = "gray60") +
      # geom_line(data = inputs_df, aes(y = value, color = "Inputs"), linewidth = 0.1) +
      scale_color_manual(values = c("True" = "gray40", "Pred." = "red", "Inputs" = "blue")) +
        labs(x = expression(s)) +
        scale_y_continuous(limits = limits, n.breaks = breaks) +
        scale_x_continuous(breaks = c(0, 1))+
        theme_bw(base_size = base_size) +
        theme(legend.position = "inside",
        legend.position.inside = c(0.55, 0.95),  # top-left corner inside plot
        legend.justification = c("left", "top"),
        legend.title = element_blank(),
        legend.key.size = unit(legend_key, "cm"),
        legend.spacing = unit(legend_spacing, "cm"),
        legend.background = element_rect(fill = "white", color = "black"),
        legend.text = element_text(size = legend_text),  
        axis.text = element_text(size = a_text),
        axis.title = element_text(size = a_title),
        #   axis.line = element_line(linewidth = 0.8),
        axis.text.x = element_text(size = a_text),
        axis.title.y = element_blank()
        )
    breaks <- 4
    limits <- c(-0.75,0.1)
    p2 <- ggplot(df, aes(x = x, y = Diff)) +
    geom_hline(yintercept = 0, color = "grey60", linewidth = 0.8, linetype = "dashed") +
    geom_line(color = "black", linewidth = 0.8) +
    labs(x = expression(s), y = NULL) +
    scale_y_continuous(limits = limits, n.breaks = breaks) +  
    scale_x_continuous(breaks = c(0, 1)) +
    theme_bw(base_size = base_size) +
    theme(
      axis.text = element_text(size = a_text),
      axis.title = element_text(size = a_title),
      axis.text.x = element_text(size = a_text)
    )
    return(list(p1=p1,p2=p2))
}

p <- list()
for(i in 1:length(data_name)){
  for(j in 1:length(index)){
    p[[(i-1)*3 + j ]] <- generate_plots(i,index[j])
  }
}
print(length(p))
#### Persistence plot 

flat_p <- unlist(p, recursive = FALSE)
aligned <- do.call(align_plots, c(flat_p, list(align = "v", axis = "l")))
## saving plots
k <- 0
for(i in 1:length(data_name)){
  for(j in 1:length(index)){
    
    inserted_path <- sub("^pred/", "plots/",pred_name[i])
    ggsave(sprintf("%s-%d.png", inserted_path,index[j]), plot = aligned[[(i-1)*3 + j +k]], width = plot_saving_width, 
           height = plot_saving_height, units = "in")
    
    k <- k+1
    inserted_path <- sub("^pred/", "plots/diff_",pred_name[i])
    ggsave(sprintf("%s-%d.png", inserted_path,index[j]), plot = aligned[[(i-1)*3 + j +k]], width = plot_saving_width, 
           height = plot_saving_height, units = "in")
  }
}




######################################################################################
############################## comparisons ###########################################
######################################################################################

data_nonlcl <- np$load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
data_lcl <- np$load("datasets/generated_1d_data_Burger-04_matern.npy")
a_text=10
a_title=10
plot_saving_height <- 1.1
plot_saving_width <- 1.8
base_size <- 7.5
legend_text <- 6
legend_key <- 0.17
legend_spacing <- 0.05
# Load data
z <- 1.96
pred_name <- c("pred/burger-FNO-nonlcl_dat-nonlcl",
                    "pred/burger-ConvLSTM-nonlcl_dat-nonlcl", "pred/burger-STDK_med-nonlcl_dat-nonlcl")
data_name <- c(1,1,1)
index <- c(1, 37, 45)  # Time steps to plot


generate_plots <- function(i,j){
    x <- (0:(2^11 - 1)) / 2^11
    print(length(x))
    if(data_name[i] == 0){
      y_test <- data_lcl[(dim(data_lcl)[1] - ntest + 1):dim(data_lcl)[1], , 10]
      y_inputs <- data_lcl[(dim(data_lcl)[1] - ntest + 1):dim(data_lcl)[1], , 1:5]
    }
    else{
      # y_test <- data_nonlcl[(dim(data_nonlcl)[1] - ntest + 1):dim(data_nonlcl)[1], , 10]
      y_test <- np$load("pred/burger_test_nonlcl.npy")
      y_inputs <- data_nonlcl[(dim(data_nonlcl)[1] - ntest + 1):dim(data_nonlcl)[1], , 1:5]
    }
    
    if(pred_name[i] == "pred/burger-STDK_med-nonlcl_dat-nonlcl"){
        pred <- np$load(sprintf("%s.npy",pred_name[i]))
        lb <- np$load("pred/burger-STDK_lb-nonlcl_dat-nonlcl.npy")
        ub <- np$load("pred/burger-STDK_ub-nonlcl_dat-nonlcl.npy")
        # Create grid for spatial axis
        sub("pred/", "", pred_name[i])
        # Compute MSE and MPIW
        mse <- mean((y_test - pred)^2)
        mpi_width <- mean(ub-lb)
        cat(sprintf("%s MSPE: ",sub("pred/", "", pred_name[i])), mse, "\n")
        cat("MPIW:", mpi_width, "\n")
        lower <- lb
        upper <- ub
        breaks <- 5
    }
    else{
        pred <- np$load(sprintf("%s.npy",pred_name[i]))
        var <- np$load(sprintf("%s_cov.npy",pred_name[i]))
        # Compute MSE and MPIW
        mse <- mean((y_test - pred)^2)
        mpi_width <- mean(2 * z * var)
        cat(sprintf("%s MSPE: ",sub("pred/", "", pred_name[i])), mse, "\n")
        cat("MPIW:", mpi_width, "\n")

        lb <- pred - z * var
        ub <- pred + z * var
        breaks <- 3
    }
    df <- data.frame(
        x = x,
        True = y_test[j + 1, ],
        Pred = pred[j+1,],
        Lower = lb[j+1,],
        Upper = ub[j+1,],
        Diff = pred[j+1,] - y_test[j + 1, ]
    )
    
    # Combine inputs into long format for ggplot
    inputs_mat <- y_inputs[j + 1, , 1:5]   # dimensions: [space, 5]
    
    # Convert to long format
    inputs_df <- data.frame(x = x, inputs_mat)
    inputs_df <- melt(inputs_df, id.vars = "x", variable.name = "Input", value.name = "value")
    # Main prediction plot with uncertainty band
    inputs_df <- inputs_df[order(inputs_df$x), ]
    # Main prediction plot with uncertainty band
    breaks <- 4
    limits <- c(-1.3,1.1)
    p1 <- ggplot(df, aes(x = x)) +
      # Input lines (faint, blue)
        geom_line(
          data = inputs_df,
          aes(y = value, group = Input, color = "Inputs"),
          linewidth = 0.2,
          alpha = 0.3
        ) +
        geom_line(aes(y = True, color = "True"), linetype = "dotted", linewidth = 0.6) +
        geom_line(aes(y = Pred, color = "Pred."), linewidth = 0.6) +
        geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.5, fill = "gray60") +
      # geom_line(data = inputs_df, aes(y = value, color = "Inputs"), linewidth = 0.1) +
      scale_color_manual(values = c("True" = "gray40", "Pred." = "red", "Inputs" = "blue")) +
        labs(x = expression(s)) +
        scale_y_continuous(limits = limits, n.breaks = breaks) +
        scale_x_continuous(breaks = c(0, 1))+
        theme_bw(base_size = base_size) +
        theme(legend.position = "inside",
        legend.position.inside = c(0.55, 0.95),  # top-left corner inside plot
        legend.justification = c("left", "top"),
        legend.title = element_blank(),
        legend.key.size = unit(legend_key, "cm"),
        legend.spacing = unit(legend_spacing, "cm"),
        legend.background = element_rect(fill = "white", color = "black"),
        legend.text = element_text(size = legend_text),  
        axis.text = element_text(size = a_text),
        axis.title = element_text(size = a_title),
        #   axis.line = element_line(linewidth = 0.8),
        axis.text.x = element_text(size = a_text),
        axis.title.y = element_blank()
        )
    breaks <- 4
    limits <- c(-0.5,0.5)
    # Difference plot
    p2 <- ggplot(df, aes(x = x, y = Diff)) +
    geom_hline(yintercept = 0, color = "grey60", linewidth = 0.8, linetype = "dashed") +
    geom_line(color = "black", linewidth = 0.8) +
    labs(x = expression(s), y = NULL) +
    scale_y_continuous(limits = limits, n.breaks = breaks) +  
    scale_x_continuous(breaks = c(0, 1)) +
    theme_bw(base_size = base_size) +
    theme(
      axis.text = element_text(size = a_text),
      axis.title = element_text(size = a_title),
      axis.text.x = element_text(size = a_text)
    )
    return(list(p1=p1,p2=p2))
}

p <- list()
for(i in 1:length(data_name)){
  for(j in 1:length(index)){
    p[[(i-1)*3 + j ]] <- generate_plots(i,index[j])
  }
}
print(length(p))

#### Persistence plot 

data <- np$load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
y_test <- data[(dim(data)[1] - ntest + 1):dim(data)[1], , 10]
# y_inputs <- data[(dim(data)[1] - ntest + 1):dim(data)[1], , 1:5]
ntest <- 50
pred <- data[(dim(data)[1] - ntest + 1):dim(data)[1], , 5]
diff <- pred - y_test

print(paste("Persistence MSPE: ",mean((y_test - pred)^2)))
x <- seq(0, 1, length.out = 2^11)

for(i in index){

df <- data.frame(
  x = x,
  True = y_test[i + 1, ],
  Pred = pred[i+1,],
  Diff = pred[i+1,] - y_test[i + 1, ]
)

# # Combine inputs into long format for ggplot
# inputs_mat <- y_inputs[j + 1, , 1:5]   # dimensions: [space, 5]
# 
# # Convert to long format
# inputs_df <- data.frame(x = x, inputs_mat)
# inputs_df <- melt(inputs_df, id.vars = "x", variable.name = "Input", value.name = "value")
# 
breaks <- 4
limits <- c(-1.3,1.1)
p1 <- ggplot(df, aes(x = x)) +
  # geom_point(
  #   data = inputs_df,
  #   aes(x = x, y = value, color = "Inputs"),
  #   size = 0.05,        
  #   alpha = 0.1       
  # ) +
        geom_line(aes(y = True, color = "True"), linetype = "dotted", linewidth = 0.6) +
        geom_line(aes(y = Pred, color = "Pred."), linewidth = 0.6) +
  # geom_line(data = inputs_df, aes(y = value, color = "Inputs"), linewidth = 0.5) +
  scale_color_manual(values = c("True" = "gray40", "Pred." = "red")) +
        labs(x = expression(s)) +
            scale_y_continuous(limits = limits, n.breaks = breaks) +
            scale_x_continuous(breaks = c(0, 1))+
        theme_bw(base_size = base_size) +
        theme(legend.position = "inside",
        legend.position.inside = c(0.55, 0.95),  # top-left corner inside plot
        legend.justification = c("left", "top"),
        legend.title = element_blank(),
        legend.key.size = unit(legend_key, "cm"),
        legend.spacing = unit(legend_spacing, "cm"),
        legend.background = element_rect(fill = "white", color = "black"),
        legend.text = element_text(size = legend_text),  
        axis.text = element_text(size = a_text),
        axis.title = element_text(size = a_title),
        #   axis.line = element_line(linewidth = 0.8),
        axis.text.x = element_text(size = a_text),
        axis.title.y = element_blank()
        )
breaks <- 4
limits <- c(-0.5,0.5)
    # Difference plot
    p2 <- ggplot(df, aes(x = x, y = Diff)) +
      geom_hline(yintercept = 0, color = "grey60", linewidth = 0.8, linetype = "dashed") +
      geom_line(color = "black", linewidth = 0.8) +
      labs(x = expression(s), y = NULL) +
      scale_y_continuous(limits = limits, n.breaks = breaks) +  
      scale_x_continuous(breaks = c(0, 1)) +
      theme_bw(base_size = base_size) +
      theme(
        axis.text = element_text(size = a_text),
        axis.title = element_text(size = a_title),
        axis.text.x = element_text(size = a_text)
      )
p[[9+i]] <- list(p1=p1,p2=p2)
}

flat_p <- unlist(p, recursive = FALSE)
aligned <- do.call(align_plots, c(flat_p, list(align = "v", axis = "l")))
## saving plots
k <- 0
for(i in 1:length(data_name)){
  for(j in 1:length(index)){
    
    inserted_path <- sub("^pred/", "plots/burger_plots/",pred_name[i])
    ggsave(sprintf("%s-%d.png", inserted_path,index[j]), plot = aligned[[(i-1)*3 + j +k]], width = plot_saving_width, 
           height = plot_saving_height, units = "in")
    
    k <- k+1
    inserted_path <- sub("^pred/", "plots/burger_plots/diff_",pred_name[i])
    ggsave(sprintf("%s-%d.png", inserted_path,index[j]), plot = aligned[[(i-1)*3 + j +k]], width = plot_saving_width, 
           height = plot_saving_height, units = "in")
  }
}

for(j in 1:length(index)){
  ## saving plots 
  ggsave(sprintf("plots/burger_plots/burger-persistance-nonlcl-%d.png",index[j]), plot = aligned[[9 +j +k]], 
         width = plot_saving_width, 
         height = plot_saving_height, units = "in")
  k <- k+1
  ggsave(sprintf("plots/burger_plots/diff_burger-burger-persistance-nonlcl_%d.png",index[j]), plot = aligned[[9 +j +k]], 
         width = plot_saving_width, 
         height = plot_saving_height, units = "in")
}
