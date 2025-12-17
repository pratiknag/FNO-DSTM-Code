# Example: 3 replicates at each of 10 locations
library(devtools)
library(GpGp)
n_sites <- 10
n_rep <- 3
total_obs <- n_sites * n_rep

# Spatio-temporal locations
set.seed(1)
base_coords <- matrix(runif(n_sites * 3), ncol = 3)  # 3 = x, y, t
coords <- base_coords[rep(1:n_sites, each = n_rep), ]

# Simulated data
signal <- sin(coords[,1]*10) + cos(coords[,2]*10)
y <- signal + rnorm(total_obs, 0, 0.1)

# Replication indices: same for each set of 3 observations
rep_inds <- rep(1:n_sites, each = n_rep)

# Fit model with replication info
fit <- fit_model(y = y,
                 locs = coords,
                 rep_inds = rep_inds,
                 covfun_name = "matern32",
                 isotropic = FALSE)